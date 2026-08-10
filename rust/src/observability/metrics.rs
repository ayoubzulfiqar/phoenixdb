//! Metrics: counters, gauges and latency histograms.
//!
//! # Dependency-free by design
//!
//! The registry compiles into every build, including the lean embedded one,
//! because it has no dependencies beyond `std`. The optional `metrics` feature
//! adds only the Prometheus *exposition format*; the instrumentation itself is
//! always available, so a Flutter app can read its own cache hit rate without
//! pulling in a metrics stack.
//!
//! # Histogram design
//!
//! Latencies are recorded in **fixed logarithmic buckets** rather than by
//! keeping every sample. That bounds memory at a few hundred bytes per
//! histogram regardless of traffic, which matters for a database that may run
//! for months. The cost is that quantiles are approximate: a reported p99 is
//! the upper bound of the bucket containing the true p99. Bucket boundaries
//! double, so the error is at most 2x — sufficient for spotting a regression,
//! not for billing.
//!
//! The required percentiles (p50, p99, p999) are computed by walking the
//! cumulative bucket counts.
//!
//! # Concurrency
//!
//! Every counter is an [`AtomicU64`] updated with `Relaxed` ordering: metrics
//! are statistics, not synchronisation, and `Relaxed` keeps the instrumentation
//! off the critical path. A reader may observe a slightly stale total; it will
//! never observe a torn value.

use std::sync::atomic::{AtomicU64, Ordering};

/// Number of buckets in a [`Histogram`].
///
/// With a 1 µs base and doubling boundaries, 32 buckets span 1 µs to ~1.2 hours,
/// which comfortably covers an fsync that has gone pathologically wrong.
pub const HISTOGRAM_BUCKETS: usize = 32;

/// A monotonically increasing counter.
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Creates a counter at zero.
    #[must_use]
    pub const fn new() -> Self {
        Counter {
            value: AtomicU64::new(0),
        }
    }

    /// Adds one.
    #[inline]
    pub fn increment(&self) {
        self.add(1);
    }

    /// Adds `n`.
    #[inline]
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Current value.
    #[inline]
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Resets to zero, returning the previous value.
    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }
}

/// A value that can go up or down.
#[derive(Debug, Default)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Creates a gauge at zero.
    #[must_use]
    pub const fn new() -> Self {
        Gauge {
            value: AtomicU64::new(0),
        }
    }

    /// Overwrites the value.
    #[inline]
    pub fn set(&self, v: u64) {
        self.value.store(v, Ordering::Relaxed);
    }

    /// Adds `n`.
    #[inline]
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Subtracts `n`, saturating at zero.
    #[inline]
    pub fn sub(&self, n: u64) {
        let _ = self
            .value
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(n))
            });
    }

    /// Current value.
    #[inline]
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// A latency histogram with logarithmic buckets.
///
/// Bucket `i` covers `[2^i, 2^(i+1))` microseconds; bucket 0 also absorbs
/// sub-microsecond samples, and the top bucket absorbs everything above its
/// lower bound.
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; HISTOGRAM_BUCKETS],
    count: AtomicU64,
    sum_micros: AtomicU64,
    min_micros: AtomicU64,
    max_micros: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Creates an empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Histogram {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
            min_micros: AtomicU64::new(u64::MAX),
            max_micros: AtomicU64::new(0),
        }
    }

    /// Bucket index for `micros`.
    #[inline]
    #[must_use]
    fn bucket_for(micros: u64) -> usize {
        if micros < 2 {
            return 0;
        }
        // floor(log2(micros)), clamped to the last bucket.
        let idx = (63 - micros.leading_zeros()) as usize;
        idx.min(HISTOGRAM_BUCKETS - 1)
    }

    /// Inclusive upper bound of bucket `i`, in microseconds.
    #[must_use]
    pub fn bucket_upper_bound(i: usize) -> u64 {
        if i >= HISTOGRAM_BUCKETS - 1 {
            return u64::MAX;
        }
        1u64 << (i + 1)
    }

    /// Records a sample.
    pub fn record_micros(&self, micros: u64) {
        self.buckets[Self::bucket_for(micros)].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add(micros, Ordering::Relaxed);
        let _ = self
            .min_micros
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |m| {
                (micros < m).then_some(micros)
            });
        let _ = self
            .max_micros
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |m| {
                (micros > m).then_some(micros)
            });
    }

    /// Records the elapsed time of `duration`.
    pub fn record(&self, duration: std::time::Duration) {
        self.record_micros(duration.as_micros().min(u64::MAX as u128) as u64);
    }

    /// Number of samples recorded.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Sum of every sample, in microseconds.
    #[must_use]
    pub fn sum_micros(&self) -> u64 {
        self.sum_micros.load(Ordering::Relaxed)
    }

    /// Arithmetic mean in microseconds, or 0 when empty.
    #[must_use]
    pub fn mean_micros(&self) -> f64 {
        let n = self.count();
        if n == 0 {
            return 0.0;
        }
        self.sum_micros() as f64 / n as f64
    }

    /// Smallest sample, or 0 when empty. Exact, not bucketed.
    #[must_use]
    pub fn min_micros(&self) -> u64 {
        let m = self.min_micros.load(Ordering::Relaxed);
        if m == u64::MAX { 0 } else { m }
    }

    /// Largest sample. Exact, not bucketed.
    #[must_use]
    pub fn max_micros(&self) -> u64 {
        self.max_micros.load(Ordering::Relaxed)
    }

    /// Approximate quantile in microseconds.
    ///
    /// `q` is clamped to `[0, 1]`. Returns the upper bound of the bucket that
    /// contains the requested quantile, so the result is an over-estimate by at
    /// most one bucket width (2x).
    #[must_use]
    pub fn quantile_micros(&self, q: f64) -> u64 {
        let total = self.count();
        if total == 0 {
            return 0;
        }
        let q = q.clamp(0.0, 1.0);
        // Rank of the sample we want, 1-based.
        let target = ((total as f64) * q).ceil().max(1.0) as u64;
        let mut cumulative = 0u64;
        for i in 0..HISTOGRAM_BUCKETS {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            if cumulative >= target {
                // Never report above the true maximum.
                return Self::bucket_upper_bound(i).min(self.max_micros().max(1));
            }
        }
        self.max_micros()
    }

    /// p50 in microseconds.
    #[must_use]
    pub fn p50_micros(&self) -> u64 {
        self.quantile_micros(0.50)
    }

    /// p99 in microseconds.
    #[must_use]
    pub fn p99_micros(&self) -> u64 {
        self.quantile_micros(0.99)
    }

    /// p999 in microseconds.
    #[must_use]
    pub fn p999_micros(&self) -> u64 {
        self.quantile_micros(0.999)
    }

    /// Snapshot of the bucket counts.
    #[must_use]
    pub fn buckets(&self) -> [u64; HISTOGRAM_BUCKETS] {
        std::array::from_fn(|i| self.buckets[i].load(Ordering::Relaxed))
    }
}

/// Times a block and records it into a histogram on drop.
///
/// ```no_run
/// # use phoenixdb::observability::metrics::{Histogram, Timer};
/// # let hist = Histogram::new();
/// {
///     let _t = Timer::start(&hist);
///     // ... work being measured ...
/// } // recorded here
/// ```
pub struct Timer<'a> {
    histogram: &'a Histogram,
    started: std::time::Instant,
}

impl<'a> Timer<'a> {
    /// Starts timing into `histogram`.
    #[must_use]
    pub fn start(histogram: &'a Histogram) -> Self {
        Timer {
            histogram,
            started: std::time::Instant::now(),
        }
    }

    /// Elapsed time so far.
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        self.started.elapsed()
    }
}

impl Drop for Timer<'_> {
    fn drop(&mut self) {
        self.histogram.record(self.started.elapsed());
    }
}

/// The engine-wide metric set.
///
/// Covers every quantity named in the observability directive: WAL fsync
/// latency percentiles, compaction throughput, per-page cache hit ratio, and
/// Raft commit-index lag.
#[derive(Debug, Default)]
pub struct EngineMetrics {
    // ---- durability ----
    /// WAL `fsync` latency. p50/p99/p999 come from this.
    pub wal_fsync_latency: Histogram,
    /// WAL bytes appended.
    pub wal_bytes_written: Counter,
    /// WAL `fsync` calls.
    pub wal_fsyncs: Counter,

    // ---- page cache ----
    /// Reads served from the page cache.
    pub cache_hits: Counter,
    /// Reads that had to touch the mapping or disk.
    pub cache_misses: Counter,
    /// Pages currently resident.
    pub cache_resident_pages: Gauge,
    /// Pages evicted.
    pub cache_evictions: Counter,

    // ---- LSM ----
    /// MemTable flushes completed.
    pub flushes: Counter,
    /// Compactions completed.
    pub compactions: Counter,
    /// Bytes read by compaction.
    pub compaction_bytes_in: Counter,
    /// Bytes written by compaction.
    pub compaction_bytes_out: Counter,
    /// Wall-clock time per compaction.
    pub compaction_duration: Histogram,
    /// Live SSTables.
    pub sstable_count: Gauge,

    // ---- transactions ----
    /// Transactions committed.
    pub txn_commits: Counter,
    /// Transactions rolled back.
    pub txn_rollbacks: Counter,
    /// Write-write conflicts detected.
    pub txn_conflicts: Counter,
    /// End-to-end commit latency.
    pub txn_commit_latency: Histogram,

    // ---- queries ----
    /// Point reads served.
    pub reads: Counter,
    /// Writes applied.
    pub writes: Counter,
    /// Read latency.
    pub read_latency: Histogram,

    // ---- raft ----
    /// Leader commit index minus this node's applied index.
    pub raft_commit_lag: Gauge,
    /// Raft log entries appended.
    pub raft_entries_appended: Counter,
    /// Leader elections observed.
    pub raft_elections: Counter,
}

impl EngineMetrics {
    /// Creates a zeroed metric set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cache hit ratio in `[0, 1]`; 0 when no reads have happened.
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.get();
        let total = hits + self.cache_misses.get();
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    /// Compaction throughput in MB/s, measured over recorded compaction time.
    ///
    /// Returns 0 when no compaction has run.
    #[must_use]
    pub fn compaction_throughput_mbps(&self) -> f64 {
        let micros = self.compaction_duration.sum_micros();
        if micros == 0 {
            return 0.0;
        }
        let bytes = self.compaction_bytes_in.get() as f64;
        let seconds = micros as f64 / 1_000_000.0;
        (bytes / (1024.0 * 1024.0)) / seconds
    }

    /// Write amplification: compaction bytes out per byte in.
    ///
    /// Returns 1.0 when nothing has been compacted.
    #[must_use]
    pub fn write_amplification(&self) -> f64 {
        let input = self.compaction_bytes_in.get();
        if input == 0 {
            return 1.0;
        }
        self.compaction_bytes_out.get() as f64 / input as f64
    }

    /// Renders a human-readable report.
    #[must_use]
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("PhoenixDB metrics\n");
        s.push_str("=================\n");
        s.push_str(&format!(
            "wal    fsyncs={} bytes={} p50={}us p99={}us p999={}us\n",
            self.wal_fsyncs.get(),
            self.wal_bytes_written.get(),
            self.wal_fsync_latency.p50_micros(),
            self.wal_fsync_latency.p99_micros(),
            self.wal_fsync_latency.p999_micros(),
        ));
        s.push_str(&format!(
            "cache  hits={} misses={} ratio={:.3} resident={} evictions={}\n",
            self.cache_hits.get(),
            self.cache_misses.get(),
            self.cache_hit_ratio(),
            self.cache_resident_pages.get(),
            self.cache_evictions.get(),
        ));
        s.push_str(&format!(
            "lsm    flushes={} compactions={} tables={} throughput={:.2}MB/s amp={:.2}x\n",
            self.flushes.get(),
            self.compactions.get(),
            self.sstable_count.get(),
            self.compaction_throughput_mbps(),
            self.write_amplification(),
        ));
        s.push_str(&format!(
            "txn    commits={} rollbacks={} conflicts={} p99={}us\n",
            self.txn_commits.get(),
            self.txn_rollbacks.get(),
            self.txn_conflicts.get(),
            self.txn_commit_latency.p99_micros(),
        ));
        s.push_str(&format!(
            "query  reads={} writes={} read_p50={}us read_p99={}us\n",
            self.reads.get(),
            self.writes.get(),
            self.read_latency.p50_micros(),
            self.read_latency.p99_micros(),
        ));
        s.push_str(&format!(
            "raft   commit_lag={} appended={} elections={}\n",
            self.raft_commit_lag.get(),
            self.raft_entries_appended.get(),
            self.raft_elections.get(),
        ));
        s
    }

    /// Renders the metric set in Prometheus text exposition format.
    ///
    /// Available without the `metrics` feature: the format is simple enough to
    /// emit directly, and doing so keeps the scrape endpoint dependency-free.
    #[must_use]
    pub fn prometheus(&self) -> String {
        let mut s = String::new();
        let counter = |s: &mut String, name: &str, help: &str, v: u64| {
            s.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n"
            ));
        };
        let gauge = |s: &mut String, name: &str, help: &str, v: f64| {
            s.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}\n"
            ));
        };

        counter(
            &mut s,
            "phoenixdb_wal_fsyncs_total",
            "WAL fsync calls",
            self.wal_fsyncs.get(),
        );
        counter(
            &mut s,
            "phoenixdb_wal_bytes_total",
            "WAL bytes written",
            self.wal_bytes_written.get(),
        );
        counter(
            &mut s,
            "phoenixdb_cache_hits_total",
            "Page cache hits",
            self.cache_hits.get(),
        );
        counter(
            &mut s,
            "phoenixdb_cache_misses_total",
            "Page cache misses",
            self.cache_misses.get(),
        );
        counter(
            &mut s,
            "phoenixdb_flushes_total",
            "MemTable flushes",
            self.flushes.get(),
        );
        counter(
            &mut s,
            "phoenixdb_compactions_total",
            "Compactions run",
            self.compactions.get(),
        );
        counter(
            &mut s,
            "phoenixdb_txn_commits_total",
            "Transactions committed",
            self.txn_commits.get(),
        );
        counter(
            &mut s,
            "phoenixdb_txn_conflicts_total",
            "Write-write conflicts",
            self.txn_conflicts.get(),
        );
        counter(
            &mut s,
            "phoenixdb_reads_total",
            "Point reads",
            self.reads.get(),
        );
        counter(
            &mut s,
            "phoenixdb_writes_total",
            "Writes applied",
            self.writes.get(),
        );

        gauge(
            &mut s,
            "phoenixdb_cache_hit_ratio",
            "Cache hit ratio",
            self.cache_hit_ratio(),
        );
        gauge(
            &mut s,
            "phoenixdb_sstable_count",
            "Live SSTables",
            self.sstable_count.get() as f64,
        );
        gauge(
            &mut s,
            "phoenixdb_raft_commit_lag",
            "Raft commit index lag",
            self.raft_commit_lag.get() as f64,
        );
        gauge(
            &mut s,
            "phoenixdb_compaction_throughput_mbps",
            "Compaction MB/s",
            self.compaction_throughput_mbps(),
        );

        // Latency summaries carry their quantiles as labels.
        for (name, help, h) in [
            (
                "phoenixdb_wal_fsync_micros",
                "WAL fsync latency",
                &self.wal_fsync_latency,
            ),
            (
                "phoenixdb_commit_micros",
                "Commit latency",
                &self.txn_commit_latency,
            ),
            ("phoenixdb_read_micros", "Read latency", &self.read_latency),
        ] {
            s.push_str(&format!("# HELP {name} {help}\n# TYPE {name} summary\n"));
            s.push_str(&format!("{name}{{quantile=\"0.5\"}} {}\n", h.p50_micros()));
            s.push_str(&format!("{name}{{quantile=\"0.99\"}} {}\n", h.p99_micros()));
            s.push_str(&format!(
                "{name}{{quantile=\"0.999\"}} {}\n",
                h.p999_micros()
            ));
            s.push_str(&format!("{name}_count {}\n", h.count()));
            s.push_str(&format!("{name}_sum {}\n", h.sum_micros()));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_accumulates_and_resets() {
        let c = Counter::new();
        assert_eq!(c.get(), 0);
        c.increment();
        c.increment();
        c.add(10);
        assert_eq!(c.get(), 12);
        assert_eq!(c.reset(), 12);
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn gauge_goes_up_and_down_and_saturates_at_zero() {
        let g = Gauge::new();
        g.set(10);
        assert_eq!(g.get(), 10);
        g.add(5);
        assert_eq!(g.get(), 15);
        g.sub(3);
        assert_eq!(g.get(), 12);
        g.sub(100);
        assert_eq!(g.get(), 0, "must saturate, not wrap");
    }

    #[test]
    fn bucket_assignment_is_logarithmic() {
        assert_eq!(Histogram::bucket_for(0), 0);
        assert_eq!(Histogram::bucket_for(1), 0);
        assert_eq!(Histogram::bucket_for(2), 1);
        assert_eq!(Histogram::bucket_for(3), 1);
        assert_eq!(Histogram::bucket_for(4), 2);
        assert_eq!(Histogram::bucket_for(7), 2);
        assert_eq!(Histogram::bucket_for(8), 3);
        assert_eq!(Histogram::bucket_for(1024), 10);
        // Anything enormous lands in the final bucket rather than panicking.
        assert_eq!(Histogram::bucket_for(u64::MAX), HISTOGRAM_BUCKETS - 1);
    }

    #[test]
    fn empty_histogram_reports_zeroes() {
        let h = Histogram::new();
        assert_eq!(h.count(), 0);
        assert_eq!(h.p50_micros(), 0);
        assert_eq!(h.p99_micros(), 0);
        assert_eq!(h.min_micros(), 0);
        assert_eq!(h.max_micros(), 0);
        assert_eq!(h.mean_micros(), 0.0);
    }

    #[test]
    fn min_max_mean_are_exact() {
        let h = Histogram::new();
        for v in [10u64, 20, 30, 40] {
            h.record_micros(v);
        }
        assert_eq!(h.count(), 4);
        assert_eq!(h.min_micros(), 10);
        assert_eq!(h.max_micros(), 40);
        assert_eq!(h.sum_micros(), 100);
        assert_eq!(h.mean_micros(), 25.0);
    }

    #[test]
    fn quantiles_track_a_uniform_distribution() {
        let h = Histogram::new();
        // 1000 samples spread over 1..=1000 microseconds.
        for v in 1..=1000u64 {
            h.record_micros(v);
        }
        let p50 = h.p50_micros();
        let p99 = h.p99_micros();
        // Bucketing over-estimates by at most one bucket (2x), never under.
        assert!(
            (500..=1024).contains(&p50),
            "p50 {p50} outside the plausible band"
        );
        assert!(
            (990..=1024).contains(&p99),
            "p99 {p99} outside the plausible band"
        );
        assert!(p99 >= p50, "quantiles must be monotonic");
    }

    #[test]
    fn quantiles_catch_a_heavy_tail() {
        // 980 fast samples and 20 very slow ones. The tail must be >1% of the
        // population for p99 to land inside it: with exactly 1% slow, the 990th
        // sorted sample is still a fast one and a low p99 would be *correct*.
        let h = Histogram::new();
        for _ in 0..980 {
            h.record_micros(10);
        }
        for _ in 0..20 {
            h.record_micros(1_000_000);
        }
        assert!(h.p50_micros() <= 16, "p50 must ignore the tail");
        assert!(
            h.p99_micros() >= 500_000,
            "p99 {} must reflect the slow tail",
            h.p99_micros()
        );
    }

    #[test]
    fn a_tail_of_exactly_one_percent_does_not_move_p99() {
        // The boundary case that makes the test above subtle, pinned so the
        // quantile rank arithmetic cannot silently drift.
        let h = Histogram::new();
        for _ in 0..990 {
            h.record_micros(10);
        }
        for _ in 0..10 {
            h.record_micros(1_000_000);
        }
        assert!(
            h.p99_micros() <= 16,
            "the 990th of 1000 samples is still fast; p99 {} should be low",
            h.p99_micros()
        );
        // p999 reaches into the tail, though.
        assert!(h.p999_micros() >= 500_000, "p999 must see the tail");
    }

    #[test]
    fn quantile_never_exceeds_the_observed_maximum() {
        let h = Histogram::new();
        h.record_micros(5);
        h.record_micros(7);
        // Bucket 2 has an upper bound of 8, but the real max is 7.
        assert!(h.p99_micros() <= 7, "must not report above the true max");
        assert!(h.quantile_micros(1.0) <= 7);
    }

    #[test]
    fn quantile_argument_is_clamped() {
        let h = Histogram::new();
        h.record_micros(100);
        assert_eq!(h.quantile_micros(-5.0), h.quantile_micros(0.0));
        assert_eq!(h.quantile_micros(50.0), h.quantile_micros(1.0));
    }

    #[test]
    fn single_sample_reports_itself_at_every_quantile() {
        let h = Histogram::new();
        h.record_micros(42);
        assert_eq!(h.p50_micros(), 42);
        assert_eq!(h.p99_micros(), 42);
        assert_eq!(h.p999_micros(), 42);
    }

    #[test]
    fn timer_records_on_drop() {
        let h = Histogram::new();
        {
            let _t = Timer::start(&h);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(h.count(), 1);
        assert!(h.max_micros() >= 1_000, "should have measured ~2ms");
    }

    #[test]
    fn cache_hit_ratio_is_correct() {
        let m = EngineMetrics::new();
        assert_eq!(m.cache_hit_ratio(), 0.0, "no data means no ratio");
        m.cache_hits.add(75);
        m.cache_misses.add(25);
        assert!((m.cache_hit_ratio() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn compaction_throughput_and_amplification() {
        let m = EngineMetrics::new();
        assert_eq!(m.compaction_throughput_mbps(), 0.0);
        assert_eq!(m.write_amplification(), 1.0);

        // 10 MiB read in exactly 1 second.
        m.compaction_bytes_in.add(10 * 1024 * 1024);
        m.compaction_bytes_out.add(5 * 1024 * 1024);
        m.compaction_duration.record_micros(1_000_000);
        assert!((m.compaction_throughput_mbps() - 10.0).abs() < 0.01);
        assert!((m.write_amplification() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn report_covers_every_required_metric() {
        let m = EngineMetrics::new();
        m.wal_fsyncs.add(3);
        m.wal_fsync_latency.record_micros(150);
        m.cache_hits.add(9);
        m.cache_misses.add(1);
        m.compactions.increment();
        m.raft_commit_lag.set(7);

        let r = m.report();
        // The directive names these four explicitly.
        assert!(r.contains("p50="), "WAL fsync percentiles");
        assert!(r.contains("p999="));
        assert!(r.contains("MB/s"), "compaction throughput");
        assert!(r.contains("ratio="), "cache hit ratio");
        assert!(r.contains("commit_lag=7"), "raft commit index lag");
    }

    #[test]
    fn prometheus_output_is_well_formed() {
        let m = EngineMetrics::new();
        m.reads.add(100);
        m.cache_hits.add(80);
        m.cache_misses.add(20);
        m.read_latency.record_micros(250);

        let p = m.prometheus();
        assert!(p.contains("# TYPE phoenixdb_reads_total counter"));
        assert!(p.contains("phoenixdb_reads_total 100"));
        assert!(p.contains("# TYPE phoenixdb_cache_hit_ratio gauge"));
        assert!(p.contains("phoenixdb_read_micros{quantile=\"0.99\"}"));
        assert!(p.contains("phoenixdb_read_micros_count 1"));

        // Every HELP must be followed by a TYPE, and no line may be blank.
        for line in p.lines() {
            assert!(
                !line.trim().is_empty(),
                "no blank lines in exposition format"
            );
        }
        let helps = p.matches("# HELP").count();
        let types = p.matches("# TYPE").count();
        assert_eq!(helps, types, "each metric needs both HELP and TYPE");
    }

    #[test]
    fn metrics_are_safe_across_threads() {
        use std::sync::Arc;
        let m = Arc::new(EngineMetrics::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let m = Arc::clone(&m);
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    m.reads.increment();
                    m.cache_hits.increment();
                    m.read_latency.record_micros(10);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.reads.get(), 8000, "counter lost updates under contention");
        assert_eq!(m.read_latency.count(), 8000);
    }
}
