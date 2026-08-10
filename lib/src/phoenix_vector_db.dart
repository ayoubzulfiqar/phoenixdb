/// Ergonomic, memory-safe vector search on top of the PhoenixDB C ABI.
///
/// ```dart
/// final db = PhoenixVectorDB.open('vectors.pvec', dimensions: 384);
/// db.insert('doc-1', embedding);
/// for (final match in db.search(query, k: 5)) {
///   print('${match.id}: ${match.score}');
/// }
/// db.close();
/// ```
///
/// Every method here blocks the calling isolate for the duration of the native
/// call. Use `AsyncPhoenixVectorDB` to keep search off a Flutter UI thread.
library;

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'native/vector_bindings.dart';
import 'phoenixdb_base.dart';

export 'native/vector_bindings.dart' show VectorMetric;

/// One search result.
///
/// Ordering is by [distance] ascending, which is the order [PhoenixVectorDB.search]
/// returns. [score] is the same information inverted so that larger is better,
/// which is usually what a caller wants to display or threshold on.
class VectorMatch implements Comparable<VectorMatch> {
  /// The id supplied at insert time.
  final String id;

  /// Metric distance from the query; smaller is nearer.
  ///
  /// * cosine — `1 - cos(theta)`, in `[0, 2]`
  /// * euclidean — the true L2 distance, in `[0, infinity)`
  /// * dot product — the negated inner product
  final double distance;

  /// Similarity score; larger is better.
  ///
  /// * cosine — the cosine similarity, in `[-1, 1]`
  /// * euclidean — `1 / (1 + d)`, in `(0, 1]`
  /// * dot product — the raw inner product
  final double score;

  /// Creates a match. Normally produced by [PhoenixVectorDB.search].
  const VectorMatch({
    required this.id,
    required this.distance,
    required this.score,
  });

  @override
  int compareTo(VectorMatch other) => distance.compareTo(other.distance);

  @override
  bool operator ==(Object other) =>
      other is VectorMatch &&
      other.id == id &&
      other.distance == distance &&
      other.score == score;

  @override
  int get hashCode => Object.hash(id, distance, score);

  @override
  String toString() =>
      'VectorMatch($id, distance: ${distance.toStringAsFixed(6)}, '
      'score: ${score.toStringAsFixed(6)})';
}

/// A k-NN query.
///
/// Bundling the parameters keeps the async and isolate APIs to a single
/// transferable argument, and gives [efSearch] somewhere to live without
/// cluttering every call site.
class VectorQuery {
  /// The query vector. Its length must equal the index's dimensionality.
  final Float32List vector;

  /// How many neighbours to return.
  final int k;

  /// Search beam width, or `null` for the engine default.
  ///
  /// Higher values raise recall at the cost of latency. The engine clamps it
  /// to at least [k], since a beam narrower than the result count cannot fill
  /// it. A value around `2 * k` is a reasonable starting point; the default
  /// (64) is already generous for typical `k <= 10` queries.
  final int? efSearch;

  /// Creates a query for the [k] nearest neighbours of [vector].
  const VectorQuery(this.vector, {this.k = 10, this.efSearch});

  /// Creates a query from any list of numbers, copying it into a
  /// [Float32List].
  ///
  /// Prefer the main constructor when the caller already holds a
  /// [Float32List] — the FFI layer passes that straight through with no copy.
  factory VectorQuery.fromList(
    List<double> values, {
    int k = 10,
    int? efSearch,
  }) => VectorQuery(Float32List.fromList(values), k: k, efSearch: efSearch);

  /// Dimensionality of the query vector.
  int get dimensions => vector.length;

  @override
  String toString() =>
      'VectorQuery(dim: ${vector.length}, k: $k, ef: $efSearch)';
}

/// Live, total and deleted record counts.
class VectorStats {
  /// Vectors that a search can return.
  final int live;

  /// Records on disk, tombstones included.
  final int total;

  /// Tombstoned records awaiting [PhoenixVectorDB.compact].
  final int deleted;

  /// Creates a statistics snapshot.
  const VectorStats({
    required this.live,
    required this.total,
    required this.deleted,
  });

  /// Fraction of records that are tombstoned, in `[0, 1]`.
  ///
  /// A useful compaction trigger: above roughly 0.3, searches are traversing
  /// substantially more graph than they need to.
  double get deletedRatio => total == 0 ? 0 : deleted / total;

  @override
  String toString() =>
      'VectorStats(live: $live, total: $total, deleted: $deleted)';
}

/// Owns the native handle and releases it if the Dart object is collected.
///
/// [NativeFinalizer] guarantees `phoenix_vector_free` runs even when a caller
/// forgets [PhoenixVectorDB.close]; the explicit path detaches the finalizer
/// first so the handle is never freed twice.
class _VectorHandleOwner implements Finalizable {
  final Pointer<PhoenixVectorEngine> pointer;

  _VectorHandleOwner(this.pointer);
}

/// A synchronous handle to an open vector index.
class PhoenixVectorDB implements Finalizable {
  final PhoenixVectorBindings _b;
  final _VectorHandleOwner _owner;
  final NativeFinalizer _finalizer;
  final int _dimensions;
  final VectorMetric _metric;
  bool _closed = false;

  PhoenixVectorDB._(
    this._b,
    this._owner,
    this._finalizer,
    this._dimensions,
    this._metric,
  ) {
    // `externalSize` lets the GC account for the native footprint, so an index
    // holding tens of megabytes of vectors creates real collection pressure.
    _finalizer.attach(
      this,
      _owner.pointer.cast(),
      detach: this,
      externalSize: 1 << 22,
    );
  }

  /// Opens (or creates) the vector index at [path].
  ///
  /// * [dimensions] — vector width, `1..=65536`. Must match an existing index;
  ///   reopening with a different width is an error rather than a silent
  ///   reinterpretation of the stored bytes.
  /// * [metric] — how neighbours are ordered. Also fixed at creation time.
  /// * [maxElements] — capacity hint; `0` means unknown. The index still grows
  ///   without bound.
  /// * [libraryPath] — overrides native library discovery.
  ///
  /// The HNSW graph snapshot lives beside the vectors at `<path>.hnsw`.
  static PhoenixVectorDB open(
    String path, {
    required int dimensions,
    VectorMetric metric = VectorMetric.cosine,
    int maxElements = 0,
    String? libraryPath,
  }) {
    if (dimensions <= 0) {
      throw ArgumentError.value(
        dimensions,
        'dimensions',
        'must be greater than zero',
      );
    }
    final bindings = PhoenixVectorBindings.load(path: libraryPath);
    final pathPtr = path.toNativeUtf8();
    final outHandle = calloc<Pointer<PhoenixVectorEngine>>();
    try {
      final status = bindings.init(
        pathPtr,
        dimensions,
        metric.code,
        maxElements,
        outHandle,
      );
      if (status != PhoenixStatus.ok) {
        throw _errorFor(bindings, status, 'open("$path")');
      }
      final handle = outHandle.value;
      if (handle == nullptr) {
        throw const PhoenixException(
          PhoenixStatus.error,
          'phoenix_vector_init returned a null handle',
        );
      }
      return PhoenixVectorDB._(
        bindings,
        _VectorHandleOwner(handle),
        NativeFinalizer(bindings.freePtr.cast()),
        dimensions,
        metric,
      );
    } finally {
      calloc.free(pathPtr);
      calloc.free(outHandle);
    }
  }

  /// Dimensionality of every vector in this index.
  int get dimensions => _dimensions;

  /// Metric this index orders by.
  VectorMetric get metric => _metric;

  /// Whether [close] has already run.
  bool get isClosed => _closed;

  /// Largest dimensionality the native layer accepts.
  int get maxDimensions => _b.maxDim();

  /// Largest `k` a single search may request.
  int get maxK => _b.maxK();

  /// Largest vector id, in bytes of UTF-8.
  int get maxIdLength => _b.maxIdLen();

  /// Name of the SIMD kernel this CPU selected: `avx2+fma`, `neon` or
  /// `portable`.
  ///
  /// Diagnostic only — every kernel computes the same distances.
  String get kernel {
    final ptr = _b.kernel();
    // A `'static` string owned by the native library: read it, never free it.
    return ptr == nullptr ? 'unknown' : ptr.toDartString();
  }

  void _ensureOpen() {
    if (_closed) {
      throw const PhoenixException(
        PhoenixStatus.invalidArgument,
        'vector index is closed',
      );
    }
  }

  static PhoenixException _errorFor(
    PhoenixVectorBindings b,
    int status,
    String context,
  ) {
    final ptr = b.lastError();
    var detail = 'native call failed';
    if (ptr != nullptr) {
      try {
        detail = ptr.toDartString();
      } finally {
        // Allocated by the native library, so the native library frees it.
        b.stringFree(ptr);
      }
    }
    return PhoenixException(status, '$context: $detail');
  }

  Never _throw(int status, String context) =>
      throw _errorFor(_b, status, context);

  void _checkVector(Float32List vector, String parameter) {
    if (vector.length != _dimensions) {
      throw ArgumentError.value(
        vector.length,
        parameter,
        'index holds $_dimensions-dimensional vectors',
      );
    }
  }

  // -------------------------------------------------------------------------
  // Data plane
  // -------------------------------------------------------------------------

  /// Inserts or replaces [id] with [vector].
  ///
  /// Re-inserting an existing id replaces it; the old record is tombstoned and
  /// reclaimed by [compact].
  ///
  /// The vector is copied into native memory for the duration of the call and
  /// released immediately afterwards, so [vector] may be reused freely.
  void insert(String id, Float32List vector) {
    _ensureOpen();
    _checkVector(vector, 'vector');
    final idPtr = id.toNativeUtf8();
    final vecPtr = _copyVector(vector);
    try {
      final status = _b.insert(_owner.pointer, idPtr, vecPtr, vector.length);
      if (status != PhoenixStatus.ok) _throw(status, 'insert("$id")');
    } finally {
      calloc.free(idPtr);
      calloc.free(vecPtr);
    }
  }

  /// Inserts every entry of [vectors], keyed by id.
  ///
  /// Each vector is validated before any native call, so a wrong-width entry
  /// aborts the whole batch rather than leaving it half-applied.
  void insertAll(Map<String, Float32List> vectors) {
    _ensureOpen();
    for (final entry in vectors.entries) {
      _checkVector(entry.value, 'vectors["${entry.key}"]');
    }
    for (final entry in vectors.entries) {
      insert(entry.key, entry.value);
    }
  }

  /// Returns the [VectorQuery.k] nearest neighbours of [query], nearest first.
  ///
  /// Fewer than `k` results come back when the index holds fewer live vectors.
  /// Below 512 live vectors the engine scans exhaustively, so small indexes
  /// return exact answers rather than approximate ones.
  List<VectorMatch> search(VectorQuery query) {
    _ensureOpen();
    _checkVector(query.vector, 'query.vector');
    if (query.k <= 0) {
      throw ArgumentError.value(query.k, 'k', 'must be greater than zero');
    }
    final limit = maxK;
    if (query.k > limit) {
      throw ArgumentError.value(query.k, 'k', 'must not exceed $limit');
    }

    final queryPtr = _copyVector(query.vector);
    final idsPtr = calloc<Pointer<Utf8>>(query.k);
    final scoresPtr = calloc<Float>(query.k);
    final countPtr = calloc<Size>();
    try {
      final status = _b.search(
        _owner.pointer,
        queryPtr,
        query.vector.length,
        query.k,
        query.efSearch ?? 0,
        idsPtr,
        scoresPtr,
        countPtr,
      );
      if (status != PhoenixStatus.ok) _throw(status, 'search');

      final count = countPtr.value;
      final matches = <VectorMatch>[];
      try {
        for (var i = 0; i < count; i++) {
          final idPtr = idsPtr[i];
          if (idPtr == nullptr) continue;
          final distance = scoresPtr[i];
          matches.add(
            VectorMatch(
              id: idPtr.toDartString(),
              distance: distance,
              score: _scoreFor(distance),
            ),
          );
        }
      } finally {
        // Every id string is native-allocated; one call releases them all,
        // and it runs even if a conversion above throws.
        _b.freeStringArray(idsPtr, count);
      }
      return matches;
    } finally {
      calloc.free(queryPtr);
      calloc.free(idsPtr);
      calloc.free(scoresPtr);
      calloc.free(countPtr);
    }
  }

  /// Convenience wrapper: searches for the [k] nearest neighbours of [vector].
  List<VectorMatch> searchVector(Float32List vector, {int k = 10, int? ef}) =>
      search(VectorQuery(vector, k: k, efSearch: ef));

  /// Fetches the vector stored under [id], or `null` when it is absent.
  Float32List? get(String id) {
    _ensureOpen();
    final idPtr = id.toNativeUtf8();
    final outPtr = calloc<Float>(_dimensions);
    try {
      final status = _b.get(_owner.pointer, idPtr, outPtr, _dimensions);
      if (status == PhoenixStatus.notFound) return null;
      if (status != PhoenixStatus.ok) _throw(status, 'get("$id")');
      // Copy out of native memory before the buffer is freed: the returned
      // list must never alias it.
      return Float32List.fromList(outPtr.asTypedList(_dimensions));
    } finally {
      calloc.free(idPtr);
      calloc.free(outPtr);
    }
  }

  /// True when [id] is stored and has not been removed.
  bool contains(String id) {
    _ensureOpen();
    final idPtr = id.toNativeUtf8();
    final outPtr = calloc<Int32>();
    try {
      final status = _b.contains(_owner.pointer, idPtr, outPtr);
      if (status != PhoenixStatus.ok) _throw(status, 'contains("$id")');
      return outPtr.value != 0;
    } finally {
      calloc.free(idPtr);
      calloc.free(outPtr);
    }
  }

  /// Removes [id], returning `false` when it was not present.
  ///
  /// The record is tombstoned rather than erased, so graph ids stay stable.
  /// Call [compact] to reclaim the space.
  bool remove(String id) {
    _ensureOpen();
    final idPtr = id.toNativeUtf8();
    try {
      final status = _b.remove(_owner.pointer, idPtr);
      if (status == PhoenixStatus.notFound) return false;
      if (status != PhoenixStatus.ok) _throw(status, 'remove("$id")');
      return true;
    } finally {
      calloc.free(idPtr);
    }
  }

  /// Number of live vectors.
  int count() {
    _ensureOpen();
    final outPtr = calloc<Size>();
    try {
      final status = _b.count(_owner.pointer, outPtr);
      if (status != PhoenixStatus.ok) _throw(status, 'count');
      return outPtr.value;
    } finally {
      calloc.free(outPtr);
    }
  }

  /// Live, total and deleted record counts.
  VectorStats stats() {
    _ensureOpen();
    final live = calloc<Size>();
    final total = calloc<Size>();
    final deleted = calloc<Size>();
    try {
      final status = _b.stats(_owner.pointer, live, total, deleted);
      if (status != PhoenixStatus.ok) _throw(status, 'stats');
      return VectorStats(
        live: live.value,
        total: total.value,
        deleted: deleted.value,
      );
    } finally {
      calloc.free(live);
      calloc.free(total);
      calloc.free(deleted);
    }
  }

  /// Syncs the vector file and writes the HNSW graph snapshot.
  ///
  /// [path] overrides the default `<index>.hnsw` location. The write is atomic,
  /// so a crash mid-save leaves the previous snapshot intact.
  void save({String? path}) {
    _ensureOpen();
    final pathPtr = path?.toNativeUtf8() ?? nullptr;
    try {
      final status = _b.save(_owner.pointer, pathPtr.cast());
      if (status != PhoenixStatus.ok) _throw(status, 'save');
    } finally {
      if (pathPtr != nullptr) calloc.free(pathPtr);
    }
  }

  /// Syncs the vector file without writing a snapshot.
  ///
  /// Cheaper than [save] and enough to make inserts durable; the graph is
  /// rebuilt from the vectors on the next open if no snapshot exists.
  void flush() {
    _ensureOpen();
    final status = _b.flush(_owner.pointer);
    if (status != PhoenixStatus.ok) _throw(status, 'flush');
  }

  /// Rewrites the index without tombstoned records, returning how many were
  /// reclaimed.
  ///
  /// Costs `O(N log N)` because the graph is rebuilt. Worth running once
  /// [VectorStats.deletedRatio] passes roughly 0.3.
  int compact() {
    _ensureOpen();
    final outPtr = calloc<Size>();
    try {
      final status = _b.compact(_owner.pointer, outPtr);
      if (status != PhoenixStatus.ok) _throw(status, 'compact');
      return outPtr.value;
    } finally {
      calloc.free(outPtr);
    }
  }

  /// Saves and closes the index.
  ///
  /// Safe to call more than once. Detaches the [NativeFinalizer] first, so the
  /// handle cannot be freed twice.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer.detach(this);
    _b.free(_owner.pointer);
  }

  /// Copies [vector] into native memory as contiguous 32-bit floats.
  ///
  /// `asTypedList().setAll` is a bulk memory copy, not an element loop, so a
  /// 1536-dimensional embedding costs one 6 KiB `memcpy`.
  Pointer<Float> _copyVector(Float32List vector) {
    final ptr = calloc<Float>(vector.length);
    ptr.asTypedList(vector.length).setAll(0, vector);
    return ptr;
  }

  /// Converts a metric distance into a "higher is better" score.
  ///
  /// Mirrors `Metric::score` in the Rust engine exactly; the two must agree or
  /// a Dart-side ranking would disagree with the native one.
  double _scoreFor(double distance) => switch (_metric) {
    VectorMetric.cosine => 1.0 - distance,
    VectorMetric.euclidean => 1.0 / (1.0 + (distance < 0 ? 0 : distance)),
    VectorMetric.dotProduct => -distance,
  };
}
