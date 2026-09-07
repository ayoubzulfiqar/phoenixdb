# PhoenixDB Audit & Roadmap

## Executive Summary

- **Architecture Health Score:** 7/10
- **Production Readiness:** Core ACID guarantees hold under current tests, but the global lock and allocation behavior limit real-world concurrency and memory footprint.
- **Top 3 Critical Risks:**
  1. All operations, including reads, serialize on one `RwLock` write guard.
  2. `sync_on_commit: false` removes the WAL fsync durability guarantee.
  3. Version store clones entire value vectors on every read.

## Verified Baseline

- Rust: 417+ tests passed
- Dart: 142 tests passed
- Added `rust/tests/audit_dynamic.rs` with 11 targeted dynamic tests

## Improvement Areas

### 1. Concurrency Model (HIGH)
- Split monolithic `RwLock<Inner>` into component locks
- Enable true concurrent reads
- Background merge without blocking writers

### 2. Memory Management (HIGH)
- Page pool to reduce `Box::new([0u8; 4096])` allocations
- `Arc<Page>` in cache/dirty set
- `Arc<[u8]>` for version store values

### 3. Scan Performance (MEDIUM)
- Streaming/cursor-based scan API
- Avoid full `BTreeMap` materialization

### 4. WAL Durability (MEDIUM)
- Always fsync Commit record
- Separate WAL fsync from data-page fsync

### 5. Vector Features (LOW)
- Hybrid search (vector + keyword)
- Quantization options
- Runtime HNSW parameter adjustment

## Implementation Plan

### Phase 1: Critical Fixes
- [ ] Split global lock into component locks
- [ ] Implement page pool
- [ ] Add streaming scan API
- [ ] Clarify WAL durability semantics

### Phase 2: Performance
- [ ] Lock-free reads with `RwLock<VersionStore>`
- [ ] `Arc<Page>` in cache
- [ ] Incremental background merge
- [ ] Vector filtering/hybrid search

### Phase 3: Features
- [ ] Multi-index support
- [ ] Backup/restore API
- [ ] Compression
- [ ] Batch operations

### Phase 4: Hardening
- [ ] Fuzz testing
- [ ] Fault injection
- [ ] Performance benchmarks
