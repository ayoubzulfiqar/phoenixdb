//! Change notifications: reactive queries over committed writes.
//!
//! A [`Watcher`] receives every committed change to keys under a prefix, so a
//! UI can rebuild a list when the data behind it moves instead of polling the
//! database. Only committed writes are published — a rolled-back or
//! conflicted transaction produces nothing — and they are published after the
//! write-ahead log is durable, so a consumer never sees a change that a crash
//! could take back.
//!
//! # Delivery
//!
//! Each subscription owns a bounded queue. A consumer that keeps up sees every
//! change in commit order; one that falls behind loses the *oldest* changes
//! and learns how many from [`Watcher::dropped`], which is the right trade for
//! a UI: the newest state matters, and a lost prefix is a signal to re-read.
//!
//! Queues hold keys, and values only when [`WatchOptions::values`] is set, so
//! a watcher over large values costs a pointer per change rather than a copy.
//!
//! # Example
//!
//! ```no_run
//! use phoenixdb::{Database, Options};
//! use phoenixdb::watch::WatchOptions;
//! use std::time::Duration;
//!
//! # fn main() -> phoenixdb::Result<()> {
//! let db = Database::open("app.pdb", Options::default())?;
//! let watcher = db.watch(b"user:", WatchOptions::default());
//! db.put_auto(b"user:1", b"ada")?;
//!
//! for change in watcher.poll(Duration::from_millis(100)) {
//!     println!("{:?} {:?}", change.kind, change.key);
//! }
//! # Ok(())
//! # }
//! ```

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// What happened to a key.
///
/// The discriminants are part of the C ABI encoding and must not be
/// renumbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChangeKind {
    /// The key was written (inserted or overwritten).
    Put = 1,
    /// The key was deleted.
    Delete = 2,
    /// Everything changed at once: the database was restored from a backup.
    /// Re-read whatever you are displaying; no per-key changes follow for it.
    Reset = 3,
}

/// One committed change.
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    /// What happened.
    pub kind: ChangeKind,
    /// The key, empty for [`ChangeKind::Reset`].
    pub key: Vec<u8>,
    /// The value written, when the subscription asked for values and this is
    /// a [`ChangeKind::Put`].
    pub value: Option<Vec<u8>>,
    /// Commit timestamp; changes are published in this order.
    pub commit_ts: u64,
}

/// What a subscription keeps and how much.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatchOptions {
    /// Changes buffered before the oldest are dropped.
    pub capacity: usize,
    /// Deliver written values as well as keys.
    pub values: bool,
}

impl Default for WatchOptions {
    fn default() -> Self {
        WatchOptions {
            capacity: 1024,
            values: false,
        }
    }
}

struct State {
    queue: VecDeque<Change>,
    dropped: u64,
    closed: bool,
    /// Bumped by [`Watcher::wake`] so a blocked poll returns promptly.
    woken: bool,
}

struct Shared {
    prefix: Vec<u8>,
    options: WatchOptions,
    state: Mutex<State>,
    signal: Condvar,
}

impl Shared {
    fn push(&self, change: &Change) {
        let mut state = self.state.lock();
        if state.closed {
            return;
        }
        if state.queue.len() >= self.options.capacity {
            state.queue.pop_front();
            state.dropped += 1;
        }
        let mut change = change.clone();
        if !self.options.values {
            change.value = None;
        }
        state.queue.push_back(change);
        self.signal.notify_all();
    }

    fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        self.signal.notify_all();
    }
}

/// A subscription to committed changes. Dropping it unsubscribes.
pub struct Watcher {
    id: u64,
    shared: Arc<Shared>,
    registry: Arc<Registry>,
}

impl Watcher {
    /// The prefix this watcher observes.
    #[must_use]
    pub fn prefix(&self) -> &[u8] {
        &self.shared.prefix
    }

    /// Takes every buffered change without waiting.
    #[must_use]
    pub fn try_poll(&self) -> Vec<Change> {
        let mut state = self.shared.state.lock();
        state.woken = false;
        state.queue.drain(..).collect()
    }

    /// Takes every buffered change, waiting up to `timeout` for the first one.
    ///
    /// Returns empty when nothing arrives in time, when [`Watcher::wake`] is
    /// called, or when the database is gone (see [`Watcher::is_closed`]).
    #[must_use]
    pub fn poll(&self, timeout: Duration) -> Vec<Change> {
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.state.lock();
        while state.queue.is_empty() && !state.closed && !state.woken {
            if self
                .shared
                .signal
                .wait_until(&mut state, deadline)
                .timed_out()
            {
                break;
            }
        }
        state.woken = false;
        state.queue.drain(..).collect()
    }

    /// Changes dropped because the queue was full, and resets the count.
    pub fn dropped(&self) -> u64 {
        std::mem::take(&mut self.shared.state.lock().dropped)
    }

    /// Unblocks a [`Watcher::poll`] in progress, so a consumer on another
    /// thread can shut down without waiting out the timeout.
    pub fn wake(&self) {
        let mut state = self.shared.state.lock();
        state.woken = true;
        self.shared.signal.notify_all();
    }

    /// Whether the database this watcher belongs to has closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.shared.state.lock().closed
    }

    /// Buffered changes not yet taken.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.shared.state.lock().queue.len()
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.registry.unsubscribe(self.id);
    }
}

impl std::fmt::Debug for Watcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Watcher")
            .field("prefix", &self.shared.prefix)
            .field("pending", &self.pending())
            .finish()
    }
}

/// Every live subscription on one database.
pub(crate) struct Registry {
    subs: Mutex<Vec<(u64, Arc<Shared>)>>,
    next_id: AtomicU64,
    /// Mirrors `subs.is_empty()` so the commit path can skip its work with an
    /// atomic load instead of taking the lock.
    active: AtomicU64,
}

impl Registry {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Registry {
            subs: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            active: AtomicU64::new(0),
        })
    }

    /// Whether any watcher is listening.
    pub(crate) fn is_empty(&self) -> bool {
        self.active.load(Ordering::Acquire) == 0
    }

    /// Live subscriptions.
    pub(crate) fn len(&self) -> usize {
        self.active.load(Ordering::Acquire) as usize
    }

    /// Whether any watcher asked for values, so the commit path knows whether
    /// to clone them.
    pub(crate) fn wants_values(&self) -> bool {
        self.subs.lock().iter().any(|(_, s)| s.options.values)
    }

    pub(crate) fn subscribe(self: &Arc<Self>, prefix: &[u8], options: WatchOptions) -> Watcher {
        let shared = Arc::new(Shared {
            prefix: prefix.to_vec(),
            options: WatchOptions {
                capacity: options.capacity.max(1),
                ..options
            },
            state: Mutex::new(State {
                queue: VecDeque::new(),
                dropped: 0,
                closed: false,
                woken: false,
            }),
            signal: Condvar::new(),
        });
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut subs = self.subs.lock();
        subs.push((id, shared.clone()));
        self.active.store(subs.len() as u64, Ordering::Release);
        drop(subs);
        Watcher {
            id,
            shared,
            registry: self.clone(),
        }
    }

    fn unsubscribe(&self, id: u64) {
        let mut subs = self.subs.lock();
        subs.retain(|(other, _)| *other != id);
        self.active.store(subs.len() as u64, Ordering::Release);
    }

    /// Publishes committed changes to every matching subscription.
    pub(crate) fn publish(&self, changes: &[Change]) {
        if changes.is_empty() {
            return;
        }
        let subs = self.subs.lock();
        for (_, shared) in subs.iter() {
            for change in changes {
                if change.kind == ChangeKind::Reset || change.key.starts_with(&shared.prefix) {
                    shared.push(change);
                }
            }
        }
    }

    /// Tells every watcher that the whole database was replaced.
    pub(crate) fn publish_reset(&self, commit_ts: u64) {
        self.publish(&[Change {
            kind: ChangeKind::Reset,
            key: Vec::new(),
            value: None,
            commit_ts,
        }]);
    }

    /// Closes every subscription; blocked polls return immediately.
    pub(crate) fn close_all(&self) {
        let mut subs = self.subs.lock();
        for (_, shared) in subs.drain(..) {
            shared.close();
        }
        self.active.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn change(kind: ChangeKind, key: &[u8], value: Option<&[u8]>) -> Change {
        Change {
            kind,
            key: key.to_vec(),
            value: value.map(<[u8]>::to_vec),
            commit_ts: 7,
        }
    }

    #[test]
    fn delivers_only_matching_keys() {
        let registry = Registry::new();
        let users = registry.subscribe(b"user:", WatchOptions::default());
        let all = registry.subscribe(b"", WatchOptions::default());
        assert!(!registry.is_empty());

        registry.publish(&[
            change(ChangeKind::Put, b"user:1", Some(b"ada")),
            change(ChangeKind::Delete, b"post:9", None),
        ]);

        let seen = users.try_poll();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].key, b"user:1");
        assert_eq!(seen[0].kind, ChangeKind::Put);
        assert_eq!(seen[0].value, None, "values are opt-in");
        assert_eq!(all.try_poll().len(), 2);
        assert!(users.try_poll().is_empty(), "a poll drains the queue");
    }

    #[test]
    fn values_are_delivered_when_requested() {
        let registry = Registry::new();
        let watcher = registry.subscribe(
            b"",
            WatchOptions {
                values: true,
                ..WatchOptions::default()
            },
        );
        registry.publish(&[change(ChangeKind::Put, b"k", Some(b"v"))]);
        assert_eq!(watcher.try_poll()[0].value.as_deref(), Some(&b"v"[..]));
        assert!(registry.wants_values());
    }

    #[test]
    fn a_full_queue_drops_the_oldest_and_counts_it() {
        let registry = Registry::new();
        let watcher = registry.subscribe(
            b"",
            WatchOptions {
                capacity: 2,
                values: false,
            },
        );
        for i in 0..5u8 {
            registry.publish(&[change(ChangeKind::Put, &[i], None)]);
        }
        let seen = watcher.try_poll();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].key, vec![3], "the newest survive");
        assert_eq!(seen[1].key, vec![4]);
        assert_eq!(watcher.dropped(), 3);
        assert_eq!(watcher.dropped(), 0, "reading resets the count");
    }

    #[test]
    fn poll_waits_then_times_out() {
        let registry = Registry::new();
        let watcher = registry.subscribe(b"", WatchOptions::default());
        let started = Instant::now();
        assert!(watcher.poll(Duration::from_millis(60)).is_empty());
        assert!(started.elapsed() >= Duration::from_millis(50), "it waited");
    }

    #[test]
    fn poll_returns_as_soon_as_a_change_arrives() {
        let registry = Registry::new();
        let watcher = registry.subscribe(b"", WatchOptions::default());
        let publisher = Arc::clone(&registry);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            publisher.publish(&[change(ChangeKind::Put, b"k", None)]);
        });
        let seen = watcher.poll(Duration::from_secs(5));
        assert_eq!(seen.len(), 1);
    }

    #[test]
    fn wake_and_close_unblock_a_poll() {
        let registry = Registry::new();
        let watcher = registry.subscribe(b"", WatchOptions::default());
        let woken = Arc::new(Mutex::new(false));
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(20));
                watcher.wake();
                *woken.lock() = true;
            });
            assert!(watcher.poll(Duration::from_secs(5)).is_empty());
        });
        assert!(*woken.lock());
        assert!(!watcher.is_closed());

        registry.close_all();
        assert!(watcher.is_closed());
        assert!(watcher.poll(Duration::from_secs(5)).is_empty());
        registry.publish(&[change(ChangeKind::Put, b"k", None)]);
        assert!(
            watcher.try_poll().is_empty(),
            "a closed queue takes nothing"
        );
    }

    #[test]
    fn dropping_a_watcher_unsubscribes() {
        let registry = Registry::new();
        {
            let _watcher = registry.subscribe(b"", WatchOptions::default());
            assert!(!registry.is_empty());
        }
        assert!(registry.is_empty());
        registry.publish(&[change(ChangeKind::Put, b"k", None)]);
    }

    #[test]
    fn a_reset_reaches_every_prefix() {
        let registry = Registry::new();
        let watcher = registry.subscribe(b"user:", WatchOptions::default());
        registry.publish_reset(42);
        let seen = watcher.try_poll();
        assert_eq!(seen[0].kind, ChangeKind::Reset);
        assert!(seen[0].key.is_empty());
        assert_eq!(seen[0].commit_ts, 42);
    }
}
