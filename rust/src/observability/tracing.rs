//! Structured tracing: hierarchical spans with timing.
//!
//! # Model
//!
//! A [`Span`] measures one unit of work — a transaction, a query, a compaction.
//! Spans carry a trace id (shared by everything in one logical operation), a
//! span id, an optional parent id, key/value attributes, and a duration. That
//! is the OpenTelemetry data model minus the wire format, so mapping these
//! records onto OTLP later is mechanical.
//!
//! # Export
//!
//! Completed spans are handed to a [`SpanExporter`]. The engine ships two:
//! [`NullExporter`] (drops everything, the default, zero overhead) and
//! [`CollectingExporter`] (keeps a bounded ring in memory, for tests and
//! `EXPLAIN`-style debugging). A Jaeger/OTLP exporter implements the same
//! two-method trait.
//!
//! # Overhead
//!
//! When no tracer is installed, [`Tracer::span`] returns a span whose `Drop`
//! does nothing but read a clock. Instrumentation can therefore stay in hot
//! paths; the directive's "span for every transaction and query" does not cost
//! an allocation unless an exporter is actually attached.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A finished span, ready for export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanRecord {
    /// Identifies the whole logical operation.
    pub trace_id: u64,
    /// Identifies this span.
    pub span_id: u64,
    /// Enclosing span, when nested.
    pub parent_id: Option<u64>,
    /// Operation name, e.g. `"txn.commit"`.
    pub name: String,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Structured key/value context.
    pub attributes: Vec<(String, String)>,
    /// Whether the operation failed.
    pub error: bool,
}

impl SpanRecord {
    /// Duration in microseconds.
    #[must_use]
    pub fn duration_micros(&self) -> u64 {
        self.duration.as_micros().min(u64::MAX as u128) as u64
    }

    /// Value of `key`, if present.
    #[must_use]
    pub fn attribute(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Receives completed spans.
///
/// Implementations must be cheap and must not panic: a span is exported from
/// `Drop`, which may run on any thread and inside error paths.
pub trait SpanExporter: Send + Sync {
    /// Called once per completed span.
    fn export(&self, span: SpanRecord);

    /// Flushes any buffered spans. Default: nothing to do.
    fn flush(&self) {}
}

/// Discards every span. The default, and free.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullExporter;

impl SpanExporter for NullExporter {
    fn export(&self, _span: SpanRecord) {}
}

/// Keeps the most recent spans in memory.
///
/// Bounded so a long-running process cannot leak: once `capacity` is reached
/// the oldest span is dropped.
#[derive(Debug)]
pub struct CollectingExporter {
    spans: Mutex<Vec<SpanRecord>>,
    capacity: usize,
}

impl CollectingExporter {
    /// Creates an exporter retaining at most `capacity` spans.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        CollectingExporter {
            spans: Mutex::new(Vec::new()),
            capacity: capacity.max(1),
        }
    }

    /// Snapshot of the retained spans, oldest first.
    #[must_use]
    pub fn spans(&self) -> Vec<SpanRecord> {
        self.spans.lock().clone()
    }

    /// Number of retained spans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.lock().len()
    }

    /// True when nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.lock().is_empty()
    }

    /// Discards everything retained.
    pub fn clear(&self) {
        self.spans.lock().clear();
    }

    /// Retained spans whose name is `name`.
    #[must_use]
    pub fn named(&self, name: &str) -> Vec<SpanRecord> {
        self.spans
            .lock()
            .iter()
            .filter(|s| s.name == name)
            .cloned()
            .collect()
    }
}

impl SpanExporter for CollectingExporter {
    fn export(&self, span: SpanRecord) {
        let mut spans = self.spans.lock();
        if spans.len() >= self.capacity {
            spans.remove(0); // bounded ring: drop the oldest
        }
        spans.push(span);
    }
}

/// Creates spans and routes them to an exporter.
pub struct Tracer {
    exporter: Arc<dyn SpanExporter>,
    next_id: AtomicU64,
    enabled: bool,
}

impl Tracer {
    /// Creates a tracer exporting to `exporter`.
    #[must_use]
    pub fn new(exporter: Arc<dyn SpanExporter>) -> Self {
        Tracer {
            exporter,
            next_id: AtomicU64::new(1),
            enabled: true,
        }
    }

    /// Creates a disabled tracer: spans are created but never exported.
    #[must_use]
    pub fn disabled() -> Self {
        Tracer {
            exporter: Arc::new(NullExporter),
            next_id: AtomicU64::new(1),
            enabled: false,
        }
    }

    /// Whether spans are exported.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Allocates a unique id.
    fn allocate_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Starts a root span for a new trace.
    #[must_use]
    pub fn span(&self, name: impl Into<String>) -> Span<'_> {
        let id = self.allocate_id();
        Span {
            tracer: self,
            trace_id: id,
            span_id: id,
            parent_id: None,
            name: name.into(),
            started: Instant::now(),
            attributes: Vec::new(),
            error: false,
        }
    }

    /// Starts a span nested inside `parent`, sharing its trace id.
    #[must_use]
    pub fn child_of(&self, parent: &Span<'_>, name: impl Into<String>) -> Span<'_> {
        Span {
            tracer: self,
            trace_id: parent.trace_id,
            span_id: self.allocate_id(),
            parent_id: Some(parent.span_id),
            name: name.into(),
            started: Instant::now(),
            attributes: Vec::new(),
            error: false,
        }
    }

    /// Flushes the exporter.
    pub fn flush(&self) {
        self.exporter.flush();
    }
}

impl std::fmt::Debug for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracer")
            .field("enabled", &self.enabled)
            .finish()
    }
}

/// An in-flight span. Exports itself when dropped.
pub struct Span<'a> {
    tracer: &'a Tracer,
    trace_id: u64,
    span_id: u64,
    parent_id: Option<u64>,
    name: String,
    started: Instant,
    attributes: Vec<(String, String)>,
    error: bool,
}

impl Span<'_> {
    /// This span's trace id.
    #[must_use]
    pub fn trace_id(&self) -> u64 {
        self.trace_id
    }

    /// This span's id.
    #[must_use]
    pub fn span_id(&self) -> u64 {
        self.span_id
    }

    /// Attaches a key/value attribute.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.push((key.into(), value.into()));
    }

    /// Builder form of [`Span::set_attribute`].
    #[must_use]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set_attribute(key, value);
        self
    }

    /// Marks the operation as failed.
    pub fn set_error(&mut self, message: impl Into<String>) {
        self.error = true;
        self.set_attribute("error", message);
    }

    /// Time elapsed so far.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        if !self.tracer.enabled {
            return; // disabled tracer: no allocation, no export
        }
        self.tracer.exporter.export(SpanRecord {
            trace_id: self.trace_id,
            span_id: self.span_id,
            parent_id: self.parent_id,
            name: std::mem::take(&mut self.name),
            duration: self.started.elapsed(),
            attributes: std::mem::take(&mut self.attributes),
            error: self.error,
        });
    }
}

impl std::fmt::Debug for Span<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Span")
            .field("trace_id", &self.trace_id)
            .field("span_id", &self.span_id)
            .field("name", &self.name)
            .finish()
    }
}

/// Groups spans by trace id, for rendering a call tree.
#[must_use]
pub fn group_by_trace(spans: &[SpanRecord]) -> HashMap<u64, Vec<SpanRecord>> {
    let mut out: HashMap<u64, Vec<SpanRecord>> = HashMap::new();
    for s in spans {
        out.entry(s.trace_id).or_default().push(s.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracer_with_collector() -> (Tracer, Arc<CollectingExporter>) {
        let exporter = Arc::new(CollectingExporter::new(100));
        let tracer = Tracer::new(exporter.clone());
        (tracer, exporter)
    }

    #[test]
    fn a_span_is_exported_when_dropped() {
        let (tracer, exporter) = tracer_with_collector();
        assert!(exporter.is_empty());
        {
            let _s = tracer.span("txn.commit");
        }
        let spans = exporter.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "txn.commit");
        assert!(spans[0].parent_id.is_none(), "root span has no parent");
    }

    #[test]
    fn span_measures_elapsed_time() {
        let (tracer, exporter) = tracer_with_collector();
        {
            let _s = tracer.span("slow");
            std::thread::sleep(Duration::from_millis(3));
        }
        let s = &exporter.spans()[0];
        assert!(
            s.duration_micros() >= 2_000,
            "expected >=2ms, measured {}us",
            s.duration_micros()
        );
    }

    #[test]
    fn child_spans_share_the_trace_and_link_to_the_parent() {
        let (tracer, exporter) = tracer_with_collector();
        {
            let parent = tracer.span("query");
            {
                let _c1 = tracer.child_of(&parent, "index.scan");
                let _c2 = tracer.child_of(&parent, "page.read");
            }
        }
        let spans = exporter.spans();
        assert_eq!(spans.len(), 3);
        let trace = spans[0].trace_id;
        assert!(
            spans.iter().all(|s| s.trace_id == trace),
            "one logical operation must share one trace id"
        );
        let children: Vec<&SpanRecord> = spans.iter().filter(|s| s.parent_id.is_some()).collect();
        assert_eq!(children.len(), 2);
        // Children close before the parent, so the parent is exported last.
        assert!(spans.last().unwrap().parent_id.is_none());
    }

    #[test]
    fn span_ids_are_unique() {
        let (tracer, exporter) = tracer_with_collector();
        for i in 0..50 {
            let _s = tracer.span(format!("op{i}"));
        }
        let spans = exporter.spans();
        let mut ids: Vec<u64> = spans.iter().map(|s| s.span_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 50, "span ids collided");
    }

    #[test]
    fn attributes_are_recorded() {
        let (tracer, exporter) = tracer_with_collector();
        {
            let mut s = tracer.span("get");
            s.set_attribute("key", "user:42");
            s.set_attribute("bytes", "128");
        }
        let s = &exporter.spans()[0];
        assert_eq!(s.attribute("key"), Some("user:42"));
        assert_eq!(s.attribute("bytes"), Some("128"));
        assert_eq!(s.attribute("missing"), None);
        assert!(!s.error);
    }

    #[test]
    fn builder_style_attributes_work() {
        let (tracer, exporter) = tracer_with_collector();
        {
            let _s = tracer.span("op").with_attribute("k", "v");
        }
        assert_eq!(exporter.spans()[0].attribute("k"), Some("v"));
    }

    #[test]
    fn errors_are_flagged_and_described() {
        let (tracer, exporter) = tracer_with_collector();
        {
            let mut s = tracer.span("commit");
            s.set_error("write-write conflict");
        }
        let s = &exporter.spans()[0];
        assert!(s.error);
        assert_eq!(s.attribute("error"), Some("write-write conflict"));
    }

    #[test]
    fn a_disabled_tracer_exports_nothing() {
        let exporter = Arc::new(CollectingExporter::new(10));
        let tracer = Tracer::disabled();
        assert!(!tracer.is_enabled());
        {
            let mut s = tracer.span("op");
            s.set_attribute("k", "v");
        }
        assert!(exporter.is_empty(), "disabled tracer must not export");
    }

    #[test]
    fn collector_is_bounded_and_drops_the_oldest() {
        let exporter = Arc::new(CollectingExporter::new(3));
        let tracer = Tracer::new(exporter.clone());
        for i in 0..10 {
            let _s = tracer.span(format!("op{i}"));
        }
        assert_eq!(exporter.len(), 3, "must not grow without bound");
        let names: Vec<String> = exporter.spans().iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, vec!["op7", "op8", "op9"], "oldest are evicted");
    }

    #[test]
    fn spans_can_be_filtered_by_name() {
        let (tracer, exporter) = tracer_with_collector();
        {
            let _a = tracer.span("read");
        }
        {
            let _b = tracer.span("write");
        }
        {
            let _c = tracer.span("read");
        }
        assert_eq!(exporter.named("read").len(), 2);
        assert_eq!(exporter.named("write").len(), 1);
        assert_eq!(exporter.named("nope").len(), 0);
    }

    #[test]
    fn grouping_reconstructs_the_call_tree() {
        let (tracer, exporter) = tracer_with_collector();
        {
            let p = tracer.span("txn");
            let _c = tracer.child_of(&p, "insert");
        }
        {
            let p2 = tracer.span("txn");
            let _c2 = tracer.child_of(&p2, "insert");
        }
        let grouped = group_by_trace(&exporter.spans());
        assert_eq!(grouped.len(), 2, "two independent traces");
        for spans in grouped.values() {
            assert_eq!(spans.len(), 2, "each trace has a parent and a child");
        }
    }

    #[test]
    fn tracing_is_thread_safe() {
        let exporter = Arc::new(CollectingExporter::new(1000));
        let tracer = Arc::new(Tracer::new(exporter.clone()));
        let mut handles = Vec::new();
        for t in 0..8 {
            let tracer = Arc::clone(&tracer);
            handles.push(std::thread::spawn(move || {
                for i in 0..50 {
                    let _s = tracer.span(format!("t{t}-op{i}"));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(exporter.len(), 400);
        let mut ids: Vec<u64> = exporter.spans().iter().map(|s| s.span_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 400, "concurrent span ids must stay unique");
    }
}
