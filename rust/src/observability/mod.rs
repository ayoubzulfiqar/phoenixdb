//! Observability: metrics and tracing.
//!
//! # Split from the directive
//!
//! The design asked for OpenTelemetry with an OTLP exporter. That pulls in
//! `tonic`/gRPC, whose transitive `cc`-based dependencies do not build in this
//! toolchain (and would bloat an embedded Flutter build considerably). The
//! split here keeps the *instrumentation* dependency-free and leaves the
//! *export* pluggable:
//!
//! * [`metrics`] — counters, gauges and latency histograms, always compiled in.
//!   Renders Prometheus text format directly, so a scrape endpoint needs no
//!   extra crates.
//! * [`tracing`] — hierarchical spans with timing, exported through a
//!   [`SpanExporter`](tracing::SpanExporter) trait. An OTLP exporter is a
//!   ~100-line implementation of that trait once a gRPC stack is available;
//!   nothing in the engine changes.
//!
//! This keeps the promise of the directive (every transaction and query emits a
//! structured span with timing) without hard-wiring a transport that cannot be
//! built or verified here.

pub mod metrics;
pub mod tracing;

pub use metrics::{Counter, EngineMetrics, Gauge, Histogram, Timer};
pub use tracing::{Span, SpanExporter, SpanRecord, Tracer};
