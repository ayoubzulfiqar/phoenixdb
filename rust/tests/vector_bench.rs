//! Vector search benchmark.
//!
//! Not a criterion harness: this is a `#[test]` behind `--ignored` so it needs
//! no dev-dependency and runs on demand with
//!
//! ```sh
//! cargo test --release --test vector_bench -- --ignored --nocapture
//! ```
//!
//! It reports build throughput, query latency, and — the number that actually
//! matters for a UI — the recall the approximate index achieves against an
//! exhaustive scan.

use phoenixdb::vector::{Metric, VectorEngine, VectorOptions};
use std::time::Instant;

/// Deterministic pseudo-random vectors, so runs are comparable.
fn cloud(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / 8_388_608.0) - 1.0
                })
                .collect()
        })
        .collect()
}

/// Exhaustive nearest neighbours, used as the recall oracle.
fn exact_neighbours(points: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let distance: f32 = point
                .iter()
                .zip(query)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (index, distance)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn benchmark_build_search_and_recall() {
    const COUNT: usize = 20_000;
    const DIM: usize = 384; // all-MiniLM-L6-v2
    const K: usize = 10;
    const QUERIES: usize = 200;

    println!("\nPhoenixDB vector benchmark");
    println!("  kernel     : {}", VectorEngine::kernel());
    println!("  vectors    : {COUNT}");
    println!("  dimensions : {DIM}");
    println!("  k          : {K}\n");

    let points = cloud(COUNT, DIM, 0xC0FFEE);
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = VectorEngine::open(
        dir.path().join("bench.pvec"),
        DIM,
        Metric::Euclidean,
        VectorOptions::default(),
    )
    .expect("open");

    // --- build ------------------------------------------------------------
    let ids: Vec<String> = (0..COUNT).map(|i| format!("v{i}")).collect();
    let batch: Vec<(&str, &[f32])> = ids
        .iter()
        .zip(&points)
        .map(|(id, p)| (id.as_str(), p.as_slice()))
        .collect();

    let started = Instant::now();
    engine.insert_many(&batch).expect("insert_many");
    let build = started.elapsed();
    println!(
        "build      : {:>8.2?} total, {:>8.1} vectors/s",
        build,
        COUNT as f64 / build.as_secs_f64()
    );

    // --- snapshot ---------------------------------------------------------
    let started = Instant::now();
    engine.save(None).expect("save");
    println!("save       : {:>8.2?}", started.elapsed());

    // --- query ------------------------------------------------------------
    // The exhaustive oracle is computed up front, outside every timer: it is
    // O(N) per query and would otherwise dominate the measurement it exists to
    // check.
    let queries = cloud(QUERIES, DIM, 0xBEEF);
    let oracle: Vec<Vec<String>> = queries
        .iter()
        .map(|q| {
            exact_neighbours(&points, q, K)
                .into_iter()
                .map(|i| format!("v{i}"))
                .collect()
        })
        .collect();

    for ef in [32usize, 64, 128, 256] {
        // Time the searches and nothing else.
        let started = Instant::now();
        let results: Vec<Vec<String>> = queries
            .iter()
            .map(|query| {
                engine
                    .search(query, K, Some(ef))
                    .expect("search")
                    .into_iter()
                    .map(|m| m.id)
                    .collect()
            })
            .collect();
        let elapsed = started.elapsed();

        let mut hits = 0usize;
        let mut total = 0usize;
        for (found, exact) in results.iter().zip(&oracle) {
            hits += found.iter().filter(|id| exact.contains(id)).count();
            total += exact.len();
        }

        let per_query = elapsed / QUERIES as u32;
        let recall = hits as f64 / total as f64;
        println!(
            "ef={ef:<4}   : {:>10.2?}/query, {:>8.0} qps, recall@{K} {:.3}",
            per_query,
            QUERIES as f64 / elapsed.as_secs_f64(),
            recall
        );

        // A frame at 120 FPS is 8.3 ms. A single search must be a small
        // fraction of that, or the "never blocks the UI" claim is hollow even
        // on a worker isolate.
        assert!(
            per_query.as_micros() < 8_300,
            "ef={ef}: {per_query:?} per query exceeds a 120 FPS frame budget"
        );
    }

    // Recall is asserted only at a wide beam. These are uniform random points
    // in 384 dimensions — the adversarial case for any graph index, since the
    // curse of dimensionality makes every point nearly equidistant. Real
    // embeddings sit on a much lower-dimensional manifold and recall far
    // higher at the same `ef`.
    let wide: Vec<Vec<String>> = queries
        .iter()
        .map(|q| {
            engine
                .search(q, K, Some(512))
                .expect("search")
                .into_iter()
                .map(|m| m.id)
                .collect()
        })
        .collect();
    let hits: usize = wide
        .iter()
        .zip(&oracle)
        .map(|(found, exact)| found.iter().filter(|id| exact.contains(id)).count())
        .sum();
    let recall = hits as f64 / (QUERIES * K) as f64;
    println!("ef=512    : recall@{K} {recall:.3}");
    assert!(recall >= 0.70, "recall {recall:.3} at ef=512 is below 0.70");

    // --- brute-force baseline, for scale ----------------------------------
    let started = Instant::now();
    for query in queries.iter().take(20) {
        std::hint::black_box(exact_neighbours(&points, query, K));
    }
    println!(
        "exhaustive : {:>8.2?}/query (baseline)",
        started.elapsed() / 20
    );
    println!();
}
