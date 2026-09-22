/// Isolate-based async vector search, so k-NN never blocks a Flutter frame.
///
/// A single long-lived worker isolate owns the native handle. Because the
/// handle never crosses an isolate boundary there is no shared-memory hazard:
/// requests and responses are plain data sent over ports, and [Float32List]
/// is transferred efficiently by the Dart runtime.
///
/// ```dart
/// final db = await AsyncPhoenixVectorDB.open('vectors.pvec', dimensions: 384);
/// await db.insert('doc-1', embedding);
/// final hits = await db.search(VectorQuery(query, k: 5));
/// await db.close();
/// ```
///
/// ## Why a dedicated isolate rather than `compute`
///
/// `compute` spawns a fresh isolate per call, which would reopen the index —
/// re-reading the graph snapshot — on every search. A long-lived worker opens
/// it once and keeps the memory-mapped vectors warm, which is the difference
/// between a sub-millisecond query and a filesystem round trip at a 120 FPS
/// frame budget of 8.3 ms.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

// `VectorMetric` arrives via phoenix_vector_db.dart, which re-exports it.
import 'phoenix_vector_db.dart';
import 'worker.dart';

/// Operations the worker understands.
enum _VectorOp {
  insert,
  insertAll,
  search,
  searchIds,
  searchBatch,
  get,
  remove,
  contains,
  count,
  stats,
  save,
  flush,
  compact,
  close,
}

/// A request sent to the worker isolate.
class _VectorRequest {
  final _VectorOp op;
  final String? key;
  final Float32List? vector;
  final int k;
  final int? ef;
  final String? path;
  final Object? args;

  const _VectorRequest(
    this.op, {
    this.key,
    this.vector,
    this.k = 10,
    this.ef,
    this.path,
    this.args,
  });
}

/// A match flattened to a transferable record.
///
/// [VectorMatch] itself is a plain object and would send fine, but keeping the
/// wire format to primitives means the worker never has to agree with the host
/// on class identity across a hot reload.
typedef _WireMatch = (String, double, double);

List<_WireMatch> _wire(List<VectorMatch> matches) => matches
    .map<_WireMatch>((m) => (m.id, m.distance, m.score))
    .toList(growable: false);

List<VectorMatch> _unwire(Object? raw) => (raw! as List<Object?>)
    .cast<_WireMatch>()
    .map((m) => VectorMatch(id: m.$1, distance: m.$2, score: m.$3))
    .toList(growable: false);

/// Startup payload for the worker isolate.
class _VectorBoot {
  final SendPort ready;
  final String path;
  final int dimensions;
  final int metricCode;
  final int maxElements;
  final String? libraryPath;

  const _VectorBoot(
    this.ready,
    this.path,
    this.dimensions,
    this.metricCode,
    this.maxElements,
    this.libraryPath,
  );
}

Object? _executeVector(PhoenixVectorDB db, _VectorRequest r) {
  switch (r.op) {
    case _VectorOp.insert:
      db.insert(r.key!, r.vector!);
      return null;
    case _VectorOp.insertAll:
      db.insertAll(r.args! as Map<String, Float32List>);
      return null;
    case _VectorOp.search:
      return _wire(db.search(VectorQuery(r.vector!, k: r.k, efSearch: r.ef)));
    case _VectorOp.searchIds:
      return _wire(
        db.searchIds(r.vector!, r.args! as List<String>, k: r.k, ef: r.ef),
      );
    case _VectorOp.searchBatch:
      return db
          .searchBatch(r.args! as List<Float32List>, k: r.k, ef: r.ef)
          .map(_wire)
          .toList(growable: false);
    case _VectorOp.get:
      return db.get(r.key!);
    case _VectorOp.remove:
      return db.remove(r.key!);
    case _VectorOp.contains:
      return db.contains(r.key!);
    case _VectorOp.count:
      return db.count();
    case _VectorOp.stats:
      final s = db.stats();
      return [s.live, s.total, s.deleted];
    case _VectorOp.save:
      db.save(path: r.path);
      return null;
    case _VectorOp.flush:
      db.flush();
      return null;
    case _VectorOp.compact:
      return db.compact();
    case _VectorOp.close:
      db.close();
      return null;
  }
}

/// Worker entry point: opens the index, then serves requests until closed.
void _vectorWorkerMain(_VectorBoot boot) => serveWorker<PhoenixVectorDB>(
  boot.ready,
  () => PhoenixVectorDB.open(
    boot.path,
    dimensions: boot.dimensions,
    metric: VectorMetric.values[boot.metricCode],
    maxElements: boot.maxElements,
    libraryPath: boot.libraryPath,
  ),
  (db, request) => _executeVector(db, request! as _VectorRequest),
  (request) => request is _VectorRequest && request.op == _VectorOp.close,
);

/// Asynchronous vector-search client backed by a dedicated worker isolate.
///
/// Every call returns a `Future` and runs entirely off the calling isolate, so
/// a search over a large index cannot drop a frame. The worker is supervised:
/// if it dies, pending and later calls fail instead of hanging.
class AsyncPhoenixVectorDB {
  final WorkerClient _worker;
  final int _dimensions;
  final VectorMetric _metric;

  AsyncPhoenixVectorDB._(this._worker, this._dimensions, this._metric);

  /// Spawns the worker isolate and opens the index at [path].
  ///
  /// The arguments mirror [PhoenixVectorDB.open]; [dimensions] and [metric]
  /// are fixed for the life of the index.
  static Future<AsyncPhoenixVectorDB> open(
    String path, {
    required int dimensions,
    VectorMetric metric = VectorMetric.cosine,
    int maxElements = 0,
    String? libraryPath,
  }) async {
    if (dimensions <= 0) {
      throw ArgumentError.value(
        dimensions,
        'dimensions',
        'must be greater than zero',
      );
    }
    final worker = await WorkerClient.spawn<_VectorBoot>(
      _vectorWorkerMain,
      (ready) => _VectorBoot(
        ready,
        path,
        dimensions,
        metric.index,
        maxElements,
        libraryPath,
      ),
      debugName: 'phoenixdb-vector-worker',
      what: 'vector index',
    );
    return AsyncPhoenixVectorDB._(worker, dimensions, metric);
  }

  /// Dimensionality of every vector in this index.
  int get dimensions => _dimensions;

  /// Metric this index orders by.
  VectorMetric get metric => _metric;

  /// Whether [close] has already run (or the worker died).
  bool get isClosed => _worker.isClosed;

  Future<Object?> _send(_VectorRequest request) => _worker.call(request);

  void _checkVector(Float32List vector, String parameter) {
    if (vector.length != _dimensions) {
      throw ArgumentError.value(
        vector.length,
        parameter,
        'index holds $_dimensions-dimensional vectors',
      );
    }
  }

  /// Inserts or replaces [id] with [vector].
  Future<void> insert(String id, Float32List vector) {
    // Validated on this side too, so an obvious mistake fails at the call site
    // rather than as an isolate round trip.
    _checkVector(vector, 'vector');
    return _send(_VectorRequest(_VectorOp.insert, key: id, vector: vector));
  }

  /// Inserts every entry of [vectors], keyed by id, as one atomic batch in a
  /// single round trip.
  Future<void> insertAll(Map<String, Float32List> vectors) async {
    if (vectors.isEmpty) return;
    for (final entry in vectors.entries) {
      _checkVector(entry.value, 'vectors["${entry.key}"]');
    }
    await _send(
      _VectorRequest(
        _VectorOp.insertAll,
        args: Map<String, Float32List>.of(vectors),
      ),
    );
  }

  /// Returns the [VectorQuery.k] nearest neighbours of [query], nearest first.
  Future<List<VectorMatch>> search(VectorQuery query) async {
    _checkVector(query.vector, 'query.vector');
    return _unwire(
      await _send(
        _VectorRequest(
          _VectorOp.search,
          vector: query.vector,
          k: query.k,
          ef: query.efSearch,
        ),
      ),
    );
  }

  /// Convenience wrapper: searches for the [k] nearest neighbours of [vector].
  Future<List<VectorMatch>> searchVector(
    Float32List vector, {
    int k = 10,
    int? ef,
  }) => search(VectorQuery(vector, k: k, efSearch: ef));

  /// Nearest neighbours of [vector] among [ids] only; see
  /// [PhoenixVectorDB.searchIds].
  Future<List<VectorMatch>> searchIds(
    Float32List vector,
    Iterable<String> ids, {
    int k = 10,
    int? ef,
  }) async {
    _checkVector(vector, 'vector');
    return _unwire(
      await _send(
        _VectorRequest(
          _VectorOp.searchIds,
          vector: vector,
          k: k,
          ef: ef,
          args: ids.toList(growable: false),
        ),
      ),
    );
  }

  /// One search per entry of [vectors], in one round trip.
  Future<List<List<VectorMatch>>> searchBatch(
    List<Float32List> vectors, {
    int k = 10,
    int? ef,
  }) async {
    for (var i = 0; i < vectors.length; i++) {
      _checkVector(vectors[i], 'vectors[$i]');
    }
    final raw =
        await _send(
              _VectorRequest(
                _VectorOp.searchBatch,
                k: k,
                ef: ef,
                args: List<Float32List>.of(vectors),
              ),
            )
            as List<Object?>;
    return raw.map(_unwire).toList(growable: false);
  }

  /// Fetches the vector stored under [id], or `null` when it is absent.
  Future<Float32List?> get(String id) async =>
      await _send(_VectorRequest(_VectorOp.get, key: id)) as Float32List?;

  /// True when [id] is stored and has not been removed.
  Future<bool> contains(String id) async =>
      await _send(_VectorRequest(_VectorOp.contains, key: id)) as bool;

  /// Removes [id], returning `false` when it was not present.
  Future<bool> remove(String id) async =>
      await _send(_VectorRequest(_VectorOp.remove, key: id)) as bool;

  /// Number of live vectors.
  Future<int> count() async =>
      await _send(const _VectorRequest(_VectorOp.count)) as int;

  /// Live, total and deleted record counts.
  Future<VectorStats> stats() async {
    final raw =
        await _send(const _VectorRequest(_VectorOp.stats)) as List<Object?>;
    return VectorStats(
      live: raw[0]! as int,
      total: raw[1]! as int,
      deleted: raw[2]! as int,
    );
  }

  /// Syncs the vector file and writes the HNSW graph snapshot.
  Future<void> save({String? path}) =>
      _send(_VectorRequest(_VectorOp.save, path: path));

  /// Syncs the vector file without writing a snapshot.
  Future<void> flush() => _send(const _VectorRequest(_VectorOp.flush));

  /// Rewrites the index without tombstoned records, returning how many were
  /// reclaimed.
  Future<int> compact() async =>
      await _send(const _VectorRequest(_VectorOp.compact)) as int;

  /// Saves the index and shuts the worker isolate down. Idempotent.
  Future<void> close() => _worker.close(const _VectorRequest(_VectorOp.close));
}
