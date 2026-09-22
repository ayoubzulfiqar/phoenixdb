/// Document collections: embeddings, metadata and full text, searched together.
///
/// ```dart
/// final kb = PhoenixCollection.open('kb', dimensions: 384);
/// kb.upsert([
///   Document('doc-1',
///       text: 'PhoenixDB is an embedded database',
///       metadata: {'lang': 'en', 'year': 2025},
///       vector: embedding),
/// ]);
/// final hits = kb.search(
///   vector: queryEmbedding,        // semantic similarity
///   text: 'embedded database',     // BM25 keyword relevance
///   filter: Filter.eq('lang', 'en') & Filter.gte('year', 2020),
///   k: 5,
///   mmr: 0.7,                      // diversify the results
/// );
/// kb.close();
/// ```
///
/// Every method blocks the calling isolate; use [AsyncPhoenixCollection] from
/// UI code.
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'native/collection_bindings.dart';
import 'native/vector_bindings.dart' show VectorMetric;
import 'phoenixdb_base.dart';

/// One document: an id plus any of text, metadata and an embedding.
class Document {
  /// Unique id, 1..=128 bytes of UTF-8 without control characters.
  final String id;

  /// Text, indexed for BM25 search and returned with hits.
  final String? text;

  /// JSON-compatible metadata (maps, lists, strings, numbers, booleans,
  /// null), filterable with [Filter]. Keys may not contain `.`.
  final Map<String, Object?>? metadata;

  /// Embedding; its length must equal the collection's dimensions.
  final Float32List? vector;

  /// Creates a document.
  const Document(this.id, {this.text, this.metadata, this.vector});

  Map<String, Object?> _wire() => {
    'id': id,
    if (text != null) 'text': text,
    if (metadata != null) 'metadata': metadata,
    if (vector != null) 'has_vector': true,
  };

  static Document _fromJson(Map<String, Object?> json) => Document(
    json['id']! as String,
    text: json['text'] as String?,
    metadata: (json['metadata'] as Map?)?.cast<String, Object?>(),
    vector: _floats(json['vector']),
  );

  @override
  String toString() =>
      'Document($id${text == null ? '' : ', text: ${_clip(text!)}'}'
      '${metadata == null ? '' : ', metadata: $metadata'}'
      '${vector == null ? '' : ', dim: ${vector!.length}'})';
}

String _clip(String s) => s.length <= 40 ? '"$s"' : '"${s.substring(0, 40)}…"';

Float32List? _floats(Object? raw) => raw == null
    ? null
    : Float32List.fromList([
        for (final v in raw as List) (v as num).toDouble(),
      ]);

/// A metadata filter, MongoDB style. Combine with `&`, `|` and `~`.
///
/// ```dart
/// Filter.eq('category', 'news') &
///     Filter.between('year', 2020, 2024) &
///     ~Filter.inList('tag', ['draft', 'spam'])
/// ```
///
/// Semantics: an array field matches when any element does; dot paths
/// address nested objects (`'author.name'`); `ne`/`notIn` also match
/// documents without the field; ranges compare numbers with numbers and
/// strings with strings only.
class Filter {
  /// The JSON form sent to the engine.
  final Map<String, Object?> json;

  /// Wraps a raw filter document, e.g. one parsed from user input.
  const Filter.raw(this.json);

  /// `field == value`.
  factory Filter.eq(String field, Object? value) => Filter.raw({field: value});

  /// `field != value` (or the field is absent).
  factory Filter.ne(String field, Object? value) => Filter.raw({
    field: {r'$ne': value},
  });

  /// `field > value`.
  factory Filter.gt(String field, Object value) => Filter.raw({
    field: {r'$gt': value},
  });

  /// `field >= value`.
  factory Filter.gte(String field, Object value) => Filter.raw({
    field: {r'$gte': value},
  });

  /// `field < value`.
  factory Filter.lt(String field, Object value) => Filter.raw({
    field: {r'$lt': value},
  });

  /// `field <= value`.
  factory Filter.lte(String field, Object value) => Filter.raw({
    field: {r'$lte': value},
  });

  /// `low <= field <= high`.
  factory Filter.between(String field, Object low, Object high) => Filter.raw({
    field: {r'$gte': low, r'$lte': high},
  });

  /// `field` equals one of [values].
  factory Filter.inList(String field, Iterable<Object?> values) => Filter.raw({
    field: {r'$in': values.toList(growable: false)},
  });

  /// `field` equals none of [values] (or is absent).
  factory Filter.notIn(String field, Iterable<Object?> values) => Filter.raw({
    field: {r'$nin': values.toList(growable: false)},
  });

  /// `field` is present (or absent, when [exists] is false).
  factory Filter.exists(String field, [bool exists = true]) => Filter.raw({
    field: {r'$exists': exists},
  });

  /// Every filter matches.
  factory Filter.and(Iterable<Filter> filters) =>
      Filter.raw({r'$and': filters.map((f) => f.json).toList(growable: false)});

  /// At least one filter matches.
  factory Filter.or(Iterable<Filter> filters) =>
      Filter.raw({r'$or': filters.map((f) => f.json).toList(growable: false)});

  /// The filter does not match.
  factory Filter.not(Filter filter) => Filter.raw({r'$not': filter.json});

  /// Both match.
  Filter operator &(Filter other) => Filter.and([this, other]);

  /// Either matches.
  Filter operator |(Filter other) => Filter.or([this, other]);

  /// Negation.
  Filter operator ~() => Filter.not(this);

  @override
  String toString() => jsonEncode(json);
}

/// How vector and keyword rankings combine in a hybrid search.
class Fusion {
  /// RRF damping constant, when this is RRF.
  final double? rrfK;

  /// Vector weight, when this is a weighted blend.
  final double? alpha;

  /// Reciprocal Rank Fusion, `Σ 1 / (k + rank)`: robust, needs no score
  /// calibration, and the default. [k] (at least 1) damps the advantage of
  /// the very top ranks.
  const Fusion.rrf([double k = 60])
    : assert(k >= 1, 'k must be >= 1'),
      rrfK = k,
      alpha = null;

  /// `alpha * vector + (1 - alpha) * text` over min-max-normalised scores;
  /// [alpha] is in `[0, 1]`.
  const Fusion.weighted(double this.alpha)
    : assert(alpha >= 0 && alpha <= 1, 'alpha must be in [0, 1]'),
      rrfK = null;

  Map<String, Object?> _wire() =>
      alpha != null ? {'alpha': alpha} : {'rrf': rrfK ?? 60.0};

  @override
  String toString() =>
      alpha != null ? 'Fusion.weighted($alpha)' : 'Fusion.rrf($rrfK)';
}

/// A collection search; see [PhoenixCollection.search].
class CollectionQuery {
  /// Query embedding for semantic search.
  final Float32List? vector;

  /// Query text for keyword (BM25) search.
  final String? text;

  /// Only documents matching this filter are returned.
  final Filter? filter;

  /// Number of results.
  final int k;

  /// HNSW beam width override.
  final int? ef;

  /// How vector and text rankings combine when both are given.
  final Fusion fusion;

  /// MMR diversity trade-off in `[0, 1]`: 1 is pure relevance, lower values
  /// favour results unlike those already picked.
  final double? mmr;

  /// Candidates fetched per retriever before fusion (default `max(4k, 20)`).
  final int? candidates;

  /// Results scoring below this are dropped.
  final double? minScore;

  /// Return each hit's text.
  final bool includeText;

  /// Return each hit's metadata.
  final bool includeMetadata;

  /// Return each hit's embedding.
  final bool includeVector;

  /// Creates a query.
  const CollectionQuery({
    this.vector,
    this.text,
    this.filter,
    this.k = 10,
    this.ef,
    this.fusion = const Fusion.rrf(),
    this.mmr,
    this.candidates,
    this.minScore,
    this.includeText = true,
    this.includeMetadata = true,
    this.includeVector = false,
  });

  Map<String, Object?> _wire() => {
    'k': k,
    if (text != null) 'text': text,
    if (filter != null) 'filter': filter!.json,
    if (ef != null) 'ef': ef,
    'fusion': fusion._wire(),
    if (mmr != null) 'mmr': mmr,
    if (candidates != null) 'candidates': candidates,
    if (minScore != null) 'min_score': minScore,
    'include': [
      if (includeText) 'text',
      if (includeMetadata) 'metadata',
      if (includeVector) 'vector',
    ],
  };
}

/// One search result.
class SearchHit {
  /// Document id.
  final String id;

  /// Final score; higher is better. For hybrid searches this is the fused
  /// score, whose scale depends on the [Fusion] used.
  final double score;

  /// Vector similarity (cosine similarity, `1/(1+d)` or dot product).
  final double? vectorScore;

  /// Raw metric distance to the query vector.
  final double? distance;

  /// BM25 relevance.
  final double? textScore;

  /// Document text, when requested.
  final String? text;

  /// Document metadata, when requested.
  final Map<String, Object?>? metadata;

  /// Document embedding, when requested.
  final Float32List? vector;

  /// Creates a hit. Normally produced by a search.
  const SearchHit({
    required this.id,
    required this.score,
    this.vectorScore,
    this.distance,
    this.textScore,
    this.text,
    this.metadata,
    this.vector,
  });

  factory SearchHit._fromJson(Map<String, Object?> j) => SearchHit(
    id: j['id']! as String,
    score: (j['score']! as num).toDouble(),
    vectorScore: (j['vector_score'] as num?)?.toDouble(),
    distance: (j['distance'] as num?)?.toDouble(),
    textScore: (j['text_score'] as num?)?.toDouble(),
    text: j['text'] as String?,
    metadata: (j['metadata'] as Map?)?.cast<String, Object?>(),
    vector: _floats(j['vector']),
  );

  @override
  String toString() =>
      'SearchHit($id, score: ${score.toStringAsFixed(4)}'
      '${text == null ? '' : ', text: ${_clip(text!)}'})';
}

/// Collection statistics.
class CollectionStats {
  /// Documents stored.
  final int documents;

  /// Documents with text.
  final int textDocuments;

  /// Documents with an embedding.
  final int vectors;

  /// Embedding dimensionality (0 = none).
  final int dimensions;

  /// Metric name (`cosine`, `euclidean` or `dot_product`).
  final String metric;

  /// Problems repaired when the collection was opened after a crash.
  final int repaired;

  /// Creates a statistics snapshot.
  const CollectionStats({
    required this.documents,
    required this.textDocuments,
    required this.vectors,
    required this.dimensions,
    required this.metric,
    required this.repaired,
  });

  factory CollectionStats._fromJson(Map<String, Object?> j) => CollectionStats(
    documents: j['documents']! as int,
    textDocuments: j['text_documents']! as int,
    vectors: j['vectors']! as int,
    dimensions: j['dim']! as int,
    metric: j['metric']! as String,
    repaired: j['repaired']! as int,
  );

  @override
  String toString() =>
      'CollectionStats(documents: $documents, vectors: $vectors, '
      'dim: $dimensions, metric: $metric)';
}

/// The collection operations the AI toolkit builds on, implemented by
/// [AsyncPhoenixCollection] directly and by [PhoenixCollection.asStore].
abstract interface class DocumentStore {
  /// Embedding dimensionality (0 when the store has no vectors).
  int get dimensions;

  /// Inserts or replaces [documents] atomically.
  Future<void> upsert(List<Document> documents);

  /// Deletes documents by id, returning how many existed.
  Future<int> delete(Iterable<String> ids);

  /// The document with [id], or `null`.
  Future<Document?> get(String id, {bool withVector = false});

  /// Runs a search.
  Future<List<SearchHit>> query(CollectionQuery query);

  /// Number of documents, or of those matching [filter].
  Future<int> count({Filter? filter});

  /// Documents matching [filter] in id order.
  Future<List<Document>> list({Filter? filter, int limit = 0, String? after});
}

class _SyncStore implements DocumentStore {
  final PhoenixCollection _c;
  _SyncStore(this._c);

  @override
  int get dimensions => _c.dimensions;

  @override
  Future<void> upsert(List<Document> documents) =>
      Future.sync(() => _c.upsert(documents));

  @override
  Future<int> delete(Iterable<String> ids) => Future.sync(() => _c.delete(ids));

  @override
  Future<Document?> get(String id, {bool withVector = false}) =>
      Future.sync(() => _c.get(id, withVector: withVector));

  @override
  Future<List<SearchHit>> query(CollectionQuery query) =>
      Future.sync(() => _c.query(query));

  @override
  Future<int> count({Filter? filter}) =>
      Future.sync(() => _c.count(filter: filter));

  @override
  Future<List<Document>> list({Filter? filter, int limit = 0, String? after}) =>
      Future.sync(() => _c.list(filter: filter, limit: limit, after: after));
}

class _CollectionOwner implements Finalizable {
  final Pointer<PhoenixCollectionNative> pointer;
  _CollectionOwner(this.pointer);
}

String _metricName(VectorMetric m) => switch (m) {
  VectorMetric.cosine => 'cosine',
  VectorMetric.euclidean => 'euclidean',
  VectorMetric.dotProduct => 'dot_product',
};

/// A synchronous handle to a document collection.
class PhoenixCollection implements Finalizable {
  final PhoenixCollectionBindings _b;
  final _CollectionOwner _owner;
  final NativeFinalizer _finalizer;
  final String path;
  int _dimensions;
  bool _closed = false;

  PhoenixCollection._(
    this._b,
    this._owner,
    this._finalizer,
    this.path,
    this._dimensions,
  ) {
    _finalizer.attach(
      this,
      _owner.pointer.cast(),
      detach: this,
      externalSize: 1 << 22,
    );
  }

  /// Opens (creating if needed) the collection in directory [path].
  ///
  /// * [dimensions] — embedding width; `0` for a text/metadata-only
  ///   collection, or to adopt an existing collection's layout.
  /// * [metric] — vector similarity, fixed at creation.
  /// * [textIndex] — maintain the BM25 index.
  /// * [sync] — `fsync` every write (default). Off trades power-loss
  ///   durability of the latest writes for much faster bulk ingest.
  /// * [m], [efConstruction], [efSearch] — HNSW tuning.
  static PhoenixCollection open(
    String path, {
    int dimensions = 0,
    VectorMetric metric = VectorMetric.cosine,
    bool textIndex = true,
    bool sync = true,
    int? m,
    int? efConstruction,
    int? efSearch,
    String? libraryPath,
  }) {
    if (dimensions < 0) {
      throw ArgumentError.value(dimensions, 'dimensions', 'must be >= 0');
    }
    final b = PhoenixCollectionBindings.load(path: libraryPath);
    final options = jsonEncode({
      'dim': dimensions,
      'metric': _metricName(metric),
      'text_index': textIndex,
      'sync': sync,
      'm': ?m,
      'ef_construction': ?efConstruction,
      'ef_search': ?efSearch,
    });
    final pathPtr = path.toNativeUtf8();
    final optionsPtr = options.toNativeUtf8();
    final out = calloc<Pointer<PhoenixCollectionNative>>();
    try {
      final status = b.open(pathPtr, optionsPtr, out);
      if (status != PhoenixStatus.ok) {
        throw _error(b, status, 'open("$path")');
      }
      final collection = PhoenixCollection._(
        b,
        _CollectionOwner(out.value),
        NativeFinalizer(b.closePtr.cast()),
        path,
        dimensions,
      );
      collection._dimensions = collection.stats().dimensions;
      return collection;
    } finally {
      calloc.free(pathPtr);
      calloc.free(optionsPtr);
      calloc.free(out);
    }
  }

  /// Embedding dimensionality (0 when the collection has no vectors).
  int get dimensions => _dimensions;

  /// Whether [close] has run.
  bool get isClosed => _closed;

  static PhoenixException _error(
    PhoenixCollectionBindings b,
    int status,
    String context,
  ) {
    final ptr = b.base.lastError();
    var detail = 'native call failed';
    if (ptr != nullptr) {
      try {
        detail = ptr.toDartString();
      } finally {
        b.base.stringFree(ptr);
      }
    }
    return PhoenixException(status, '$context: $detail');
  }

  void _check(int status, String context) {
    if (status != PhoenixStatus.ok) throw _error(_b, status, context);
  }

  Pointer<PhoenixCollectionNative> get _h {
    if (_closed) {
      throw const PhoenixException(
        PhoenixStatus.invalidArgument,
        'collection is closed',
      );
    }
    return _owner.pointer;
  }

  /// Runs [call] with an out-string slot and returns the decoded JSON.
  Object? _json(int Function(Pointer<Pointer<Utf8>> out) call, String context) {
    final out = calloc<Pointer<Utf8>>();
    try {
      _check(call(out), context);
      final ptr = out.value;
      try {
        return jsonDecode(ptr.toDartString());
      } finally {
        _b.base.stringFree(ptr);
      }
    } finally {
      calloc.free(out);
    }
  }

  static Pointer<Utf8> _encode(Object? value, String what) {
    try {
      return jsonEncode(value).toNativeUtf8();
    } on JsonUnsupportedObjectError catch (e) {
      throw ArgumentError(
        '$what is not JSON-encodable: ${e.unsupportedObject}',
      );
    }
  }

  void _checkVector(Float32List v, String what) {
    if (_dimensions == 0) {
      throw ArgumentError('$what: this collection has no vectors');
    }
    if (v.length != _dimensions) {
      throw ArgumentError.value(
        v.length,
        what,
        'collection holds $_dimensions-dimensional vectors',
      );
    }
  }

  /// Inserts or replaces [documents] atomically: all land or none do. A
  /// repeated id is applied in order (the last wins).
  void upsert(List<Document> documents) {
    if (documents.isEmpty) return;
    var floats = 0;
    for (final d in documents) {
      final v = d.vector;
      if (v != null) {
        _checkVector(v, 'documents["${d.id}"].vector');
        floats += v.length;
      }
    }
    final docs = _encode([for (final d in documents) d._wire()], 'documents');
    final vectors = floats == 0 ? nullptr : calloc<Float>(floats);
    try {
      if (floats > 0) {
        final view = vectors.asTypedList(floats);
        var at = 0;
        for (final d in documents) {
          final v = d.vector;
          if (v != null) {
            view.setAll(at, v);
            at += v.length;
          }
        }
      }
      _check(_b.upsert(_h, docs, vectors, floats), 'upsert');
    } finally {
      calloc.free(docs);
      if (floats > 0) calloc.free(vectors);
    }
  }

  /// Inserts or replaces one document.
  void add(Document document) => upsert([document]);

  /// Deletes documents by id, returning how many existed.
  int delete(Iterable<String> ids) {
    final list = ids.toList(growable: false);
    if (list.isEmpty) return 0;
    final json = _encode(list, 'ids');
    final out = calloc<Uint64>();
    try {
      _check(_b.delete(_h, json, out), 'delete');
      return out.value;
    } finally {
      calloc.free(json);
      calloc.free(out);
    }
  }

  /// The document with [id], or `null`.
  Document? get(String id, {bool withVector = false}) {
    final idPtr = id.toNativeUtf8();
    final out = calloc<Pointer<Utf8>>();
    try {
      final status = _b.get(_h, idPtr, withVector ? 1 : 0, out);
      if (status == PhoenixStatus.notFound) return null;
      _check(status, 'get("$id")');
      final ptr = out.value;
      try {
        return Document._fromJson(
          (jsonDecode(ptr.toDartString()) as Map).cast<String, Object?>(),
        );
      } finally {
        _b.base.stringFree(ptr);
      }
    } finally {
      calloc.free(idPtr);
      calloc.free(out);
    }
  }

  /// Searches the collection; see [CollectionQuery] for the parameters.
  ///
  /// * [vector] only — nearest neighbours;
  /// * [text] only — BM25 keyword search;
  /// * both — hybrid, fused by [fusion];
  /// * neither — the [filter]'s matches in id order.
  List<SearchHit> search({
    Float32List? vector,
    String? text,
    Filter? filter,
    int k = 10,
    int? ef,
    Fusion fusion = const Fusion.rrf(),
    double? mmr,
    int? candidates,
    double? minScore,
    bool includeText = true,
    bool includeMetadata = true,
    bool includeVector = false,
  }) => query(
    CollectionQuery(
      vector: vector,
      text: text,
      filter: filter,
      k: k,
      ef: ef,
      fusion: fusion,
      mmr: mmr,
      candidates: candidates,
      minScore: minScore,
      includeText: includeText,
      includeMetadata: includeMetadata,
      includeVector: includeVector,
    ),
  );

  /// Runs a prepared [CollectionQuery].
  List<SearchHit> query(CollectionQuery q) {
    if (q.k < 0) throw ArgumentError.value(q.k, 'k', 'must be >= 0');
    final v = q.vector;
    if (v != null) _checkVector(v, 'vector');
    final request = _encode(q._wire(), 'query');
    final qv = v == null ? nullptr : calloc<Float>(v.length);
    try {
      if (v != null) qv.asTypedList(v.length).setAll(0, v);
      final raw = _json(
        (out) => _b.search(_h, request, qv, v?.length ?? 0, out),
        'search',
      );
      return [
        for (final h in raw! as List)
          SearchHit._fromJson((h as Map).cast<String, Object?>()),
      ];
    } finally {
      calloc.free(request);
      if (v != null) calloc.free(qv);
    }
  }

  /// Number of documents, or of those matching [filter].
  int count({Filter? filter}) {
    final json = filter == null ? nullptr : _encode(filter.json, 'filter');
    final out = calloc<Uint64>();
    try {
      _check(_b.count(_h, json, out), 'count');
      return out.value;
    } finally {
      if (filter != null) calloc.free(json);
      calloc.free(out);
    }
  }

  /// Documents matching [filter] in id order, after [after] (exclusive), at
  /// most [limit] (0 = no limit). Vectors are not included.
  List<Document> list({Filter? filter, int limit = 0, String? after}) {
    if (limit < 0) throw ArgumentError.value(limit, 'limit', 'must be >= 0');
    final json = filter == null ? nullptr : _encode(filter.json, 'filter');
    final afterPtr = after == null ? nullptr : after.toNativeUtf8();
    try {
      final raw = _json(
        (out) => _b.list(_h, json, limit, afterPtr, out),
        'list',
      );
      return [
        for (final d in raw! as List)
          Document._fromJson((d as Map).cast<String, Object?>()),
      ];
    } finally {
      if (filter != null) calloc.free(json);
      if (after != null) calloc.free(afterPtr);
    }
  }

  /// Every document matching [filter], fetched lazily in pages of
  /// [pageSize].
  Iterable<Document> documents({Filter? filter, int pageSize = 256}) sync* {
    String? after;
    while (true) {
      final page = list(filter: filter, limit: pageSize, after: after);
      yield* page;
      if (page.length < pageSize) return;
      after = page.last.id;
    }
  }

  /// Statistics.
  CollectionStats stats() => CollectionStats._fromJson(
    (_json((out) => _b.stats(_h, out), 'stats')! as Map)
        .cast<String, Object?>(),
  );

  /// Syncs vectors and checkpoints documents.
  void flush() => _check(_b.flush(_h), 'flush');

  /// This collection as a [DocumentStore], for the AI toolkit. Calls still
  /// run on the calling isolate; use [AsyncPhoenixCollection] from UI code.
  DocumentStore asStore() => _SyncStore(this);

  /// Closes the collection. Idempotent.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer.detach(this);
    _b.close(_owner.pointer);
  }
}
