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
import 'phoenixdb_base.dart';

/// Operations the worker understands.
enum _VectorOp {
  insert,
  search,
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
  final int id;
  final _VectorOp op;
  final String? key;
  final Float32List? vector;
  final int k;
  final int? ef;
  final String? path;

  const _VectorRequest(
    this.id,
    this.op, {
    this.key,
    this.vector,
    this.k = 10,
    this.ef,
    this.path,
  });
}

/// A response returned by the worker isolate.
class _VectorResponse {
  final int id;
  final Object? result;
  final String? error;
  final int? status;

  const _VectorResponse(this.id, {this.result, this.error, this.status});
}

/// A match flattened to a transferable record.
///
/// [VectorMatch] itself is a plain object and would send fine, but keeping the
/// wire format to primitives means the worker never has to agree with the host
/// on class identity across a hot reload.
typedef _WireMatch = (String, double, double);

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

/// Worker entry point: opens the index, then serves requests until closed.
void _vectorWorkerMain(_VectorBoot boot) {
  final commands = ReceivePort();
  PhoenixVectorDB db;
  try {
    db = PhoenixVectorDB.open(
      boot.path,
      dimensions: boot.dimensions,
      metric: VectorMetric.values[boot.metricCode],
      maxElements: boot.maxElements,
      libraryPath: boot.libraryPath,
    );
  } catch (e) {
    boot.ready.send('error: $e');
    commands.close();
    return;
  }
  boot.ready.send(commands.sendPort);

  commands.listen((message) {
    if (message is! List || message.length != 2) return;
    final request = message[0] as _VectorRequest;
    final reply = message[1] as SendPort;

    try {
      final Object? result = switch (request.op) {
        _VectorOp.insert => () {
          db.insert(request.key!, request.vector!);
          return null;
        }(),
        _VectorOp.search =>
          db
              .search(
                VectorQuery(
                  request.vector!,
                  k: request.k,
                  efSearch: request.ef,
                ),
              )
              .map<_WireMatch>((m) => (m.id, m.distance, m.score))
              .toList(growable: false),
        _VectorOp.get => db.get(request.key!),
        _VectorOp.remove => db.remove(request.key!),
        _VectorOp.contains => db.contains(request.key!),
        _VectorOp.count => db.count(),
        _VectorOp.stats => () {
          final s = db.stats();
          return [s.live, s.total, s.deleted];
        }(),
        _VectorOp.save => () {
          db.save(path: request.path);
          return null;
        }(),
        _VectorOp.flush => () {
          db.flush();
          return null;
        }(),
        _VectorOp.compact => db.compact(),
        _VectorOp.close => () {
          db.close();
          return null;
        }(),
      };
      reply.send(_VectorResponse(request.id, result: result));
      if (request.op == _VectorOp.close) {
        commands.close();
      }
    } on PhoenixException catch (e) {
      reply.send(
        _VectorResponse(request.id, error: e.message, status: e.status),
      );
    } catch (e) {
      reply.send(_VectorResponse(request.id, error: e.toString()));
    }
  });
}

/// Asynchronous vector-search client backed by a dedicated worker isolate.
///
/// Every call returns a `Future` and runs entirely off the calling isolate, so
/// a search over a large index cannot drop a frame.
class AsyncPhoenixVectorDB {
  final SendPort _commands;
  final ReceivePort _responses;
  final Map<int, Completer<Object?>> _pending = {};
  final int _dimensions;
  final VectorMetric _metric;
  int _nextId = 1;
  bool _closed = false;

  AsyncPhoenixVectorDB._(
    this._commands,
    this._responses,
    this._dimensions,
    this._metric,
  ) {
    _responses.listen((message) {
      if (message is! _VectorResponse) return;
      final completer = _pending.remove(message.id);
      if (completer == null) return;
      if (message.error != null) {
        completer.completeError(
          PhoenixException(message.status ?? -1, message.error!),
        );
      } else {
        completer.complete(message.result);
      }
    });
  }

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
    final ready = ReceivePort();
    await Isolate.spawn(
      _vectorWorkerMain,
      _VectorBoot(
        ready.sendPort,
        path,
        dimensions,
        metric.index,
        maxElements,
        libraryPath,
      ),
      debugName: 'phoenixdb-vector-worker',
    );
    final first = await ready.first;
    ready.close();
    if (first is String) {
      throw PhoenixException(-1, 'failed to open vector index: $first');
    }
    return AsyncPhoenixVectorDB._(
      first as SendPort,
      ReceivePort(),
      dimensions,
      metric,
    );
  }

  /// Dimensionality of every vector in this index.
  int get dimensions => _dimensions;

  /// Metric this index orders by.
  VectorMetric get metric => _metric;

  /// Whether [close] has already run.
  bool get isClosed => _closed;

  Future<Object?> _send(_VectorRequest request) {
    if (_closed) {
      return Future.error(const PhoenixException(-2, 'vector index is closed'));
    }
    final completer = Completer<Object?>();
    _pending[request.id] = completer;
    _commands.send([request, _responses.sendPort]);
    return completer.future;
  }

  int get _id => _nextId++;

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
    return _send(
      _VectorRequest(_id, _VectorOp.insert, key: id, vector: vector),
    );
  }

  /// Inserts every entry of [vectors], keyed by id.
  ///
  /// Requests are pipelined rather than serialised: the worker processes them
  /// in order while the caller waits once at the end.
  Future<void> insertAll(Map<String, Float32List> vectors) async {
    for (final entry in vectors.entries) {
      _checkVector(entry.value, 'vectors["${entry.key}"]');
    }
    await Future.wait([
      for (final entry in vectors.entries) insert(entry.key, entry.value),
    ]);
  }

  /// Returns the [VectorQuery.k] nearest neighbours of [query], nearest first.
  Future<List<VectorMatch>> search(VectorQuery query) async {
    _checkVector(query.vector, 'query.vector');
    final raw =
        await _send(
              _VectorRequest(
                _id,
                _VectorOp.search,
                vector: query.vector,
                k: query.k,
                ef: query.efSearch,
              ),
            )
            as List<Object?>;
    return raw
        .cast<_WireMatch>()
        .map((m) => VectorMatch(id: m.$1, distance: m.$2, score: m.$3))
        .toList(growable: false);
  }

  /// Convenience wrapper: searches for the [k] nearest neighbours of [vector].
  Future<List<VectorMatch>> searchVector(
    Float32List vector, {
    int k = 10,
    int? ef,
  }) => search(VectorQuery(vector, k: k, efSearch: ef));

  /// Fetches the vector stored under [id], or `null` when it is absent.
  Future<Float32List?> get(String id) async =>
      await _send(_VectorRequest(_id, _VectorOp.get, key: id)) as Float32List?;

  /// True when [id] is stored and has not been removed.
  Future<bool> contains(String id) async =>
      await _send(_VectorRequest(_id, _VectorOp.contains, key: id)) as bool;

  /// Removes [id], returning `false` when it was not present.
  Future<bool> remove(String id) async =>
      await _send(_VectorRequest(_id, _VectorOp.remove, key: id)) as bool;

  /// Number of live vectors.
  Future<int> count() async =>
      await _send(_VectorRequest(_id, _VectorOp.count)) as int;

  /// Live, total and deleted record counts.
  Future<VectorStats> stats() async {
    final raw =
        await _send(_VectorRequest(_id, _VectorOp.stats)) as List<Object?>;
    return VectorStats(
      live: raw[0]! as int,
      total: raw[1]! as int,
      deleted: raw[2]! as int,
    );
  }

  /// Syncs the vector file and writes the HNSW graph snapshot.
  Future<void> save({String? path}) =>
      _send(_VectorRequest(_id, _VectorOp.save, path: path));

  /// Syncs the vector file without writing a snapshot.
  Future<void> flush() => _send(_VectorRequest(_id, _VectorOp.flush));

  /// Rewrites the index without tombstoned records, returning how many were
  /// reclaimed.
  Future<int> compact() async =>
      await _send(_VectorRequest(_id, _VectorOp.compact)) as int;

  /// Saves the index and shuts the worker isolate down.
  Future<void> close() async {
    if (_closed) return;
    try {
      await _send(_VectorRequest(_id, _VectorOp.close));
    } finally {
      _closed = true;
      _responses.close();
      for (final completer in _pending.values) {
        if (!completer.isCompleted) {
          completer.completeError(
            const PhoenixException(
              -1,
              'vector index closed while a call was in flight',
            ),
          );
        }
      }
      _pending.clear();
    }
  }
}
