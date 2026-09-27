/// Isolate-backed async document collections, so ingest and search never
/// block a Flutter frame.
///
/// ```dart
/// final kb = await AsyncPhoenixCollection.open('kb', dimensions: 384);
/// await kb.upsert([Document('a', text: 'hello', vector: embedding)]);
/// final hits = await kb.search(text: 'hello', vector: query, k: 5);
/// await kb.close();
/// ```
library;

import 'dart:async';
import 'dart:convert' show JsonUnsupportedObjectError, jsonEncode;
import 'dart:isolate';
import 'dart:typed_data';

import 'collection.dart';
import 'native/vector_bindings.dart' show VectorMetric;
import 'watch.dart';
import 'watch_isolate.dart';
import 'worker.dart';

enum _Op {
  upsert,
  delete,
  get,
  query,
  count,
  list,
  stats,
  verify,
  compact,
  backup,
  flush,
  close,
}

class _Request {
  final _Op op;
  final Object? a;
  final Object? b;
  const _Request(this.op, [this.a, this.b]);
}

/// One page request, so `list` keeps its four parameters together.
class _Page {
  final Filter? filter;
  final int limit;
  final String? after;
  final bool newestFirst;
  const _Page(this.filter, this.limit, this.after, this.newestFirst);
}

class _Boot {
  final SendPort ready;
  final String path;
  final int dimensions;
  final int metric;
  final bool textIndex;
  final bool sync;
  final int? m;
  final int? efConstruction;
  final int? efSearch;
  final String? libraryPath;

  const _Boot(
    this.ready,
    this.path,
    this.dimensions,
    this.metric,
    this.textIndex,
    this.sync,
    this.m,
    this.efConstruction,
    this.efSearch,
    this.libraryPath,
  );
}

Object? _execute(PhoenixCollection c, _Request r) {
  switch (r.op) {
    case _Op.upsert:
      c.upsert(r.a! as List<Document>);
      return null;
    case _Op.delete:
      return c.delete(r.a! as List<String>);
    case _Op.get:
      return c.get(r.a! as String, withVector: r.b! as bool);
    case _Op.query:
      return c.query(r.a! as CollectionQuery);
    case _Op.count:
      return c.count(filter: r.a as Filter?);
    case _Op.list:
      final page = r.a! as _Page;
      return c.list(
        filter: page.filter,
        limit: page.limit,
        after: page.after,
        newestFirst: page.newestFirst,
      );
    case _Op.stats:
      return c.stats();
    case _Op.verify:
      return c.verify();
    case _Op.compact:
      return c.compact();
    case _Op.backup:
      c.backup(r.a! as String);
      return null;
    case _Op.flush:
      c.flush();
      return null;
    case _Op.close:
      c.close();
      return null;
  }
}

void _main(_Boot boot) => serveWorker<PhoenixCollection>(
  boot.ready,
  () => PhoenixCollection.open(
    boot.path,
    dimensions: boot.dimensions,
    metric: VectorMetric.values[boot.metric],
    textIndex: boot.textIndex,
    sync: boot.sync,
    m: boot.m,
    efConstruction: boot.efConstruction,
    efSearch: boot.efSearch,
    libraryPath: boot.libraryPath,
  ),
  (c, request) => _execute(c, request! as _Request),
  (request) => request is _Request && request.op == _Op.close,
);

/// Asynchronous [PhoenixCollection] backed by a supervised worker isolate.
///
/// Calls are served in order; if the worker dies, pending and later calls
/// fail instead of hanging.
class AsyncPhoenixCollection implements DocumentStore {
  final WorkerClient _worker;
  final int _dimensions;
  final Set<WatchSession> _watchers = {};

  /// Directory the collection lives in.
  final String path;

  /// Native library override this client was opened with, if any.
  final String? libraryPath;

  AsyncPhoenixCollection._(
    this._worker,
    this._dimensions,
    this.path,
    this.libraryPath,
  );

  /// Spawns the worker and opens the collection; arguments mirror
  /// [PhoenixCollection.open].
  static Future<AsyncPhoenixCollection> open(
    String path, {
    int dimensions = 0,
    VectorMetric metric = VectorMetric.cosine,
    bool textIndex = true,
    bool sync = true,
    int? m,
    int? efConstruction,
    int? efSearch,
    String? libraryPath,
  }) async {
    if (dimensions < 0) {
      throw ArgumentError.value(dimensions, 'dimensions', 'must be >= 0');
    }
    final worker = await WorkerClient.spawn<_Boot>(
      _main,
      (ready) => _Boot(
        ready,
        path,
        dimensions,
        metric.index,
        textIndex,
        sync,
        m,
        efConstruction,
        efSearch,
        libraryPath,
      ),
      debugName: 'phoenixdb-collection-worker',
      what: 'collection',
    );
    final client = AsyncPhoenixCollection._(
      worker,
      dimensions,
      path,
      libraryPath,
    );
    if (dimensions != 0) return client;
    try {
      // Adopt an existing collection's layout.
      final stats = await client.stats();
      return AsyncPhoenixCollection._(
        worker,
        stats.dimensions,
        path,
        libraryPath,
      );
    } on Object {
      // Otherwise the worker isolate — and the directory's file lock — would
      // outlive this failed open with no way to reach it.
      await client.close();
      rethrow;
    }
  }

  /// Embedding dimensionality (0 when the collection has no vectors).
  @override
  int get dimensions => _dimensions;

  /// Whether [close] has run (or the worker died).
  bool get isClosed => _worker.isClosed;

  Future<Object?> _send(_Request r) => _worker.call(r);

  void _checkVector(Float32List? v, String what) {
    if (v == null) return;
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

  /// Inserts or replaces [documents] atomically, in one round trip.
  @override
  Future<void> upsert(List<Document> documents) async {
    if (documents.isEmpty) return;
    for (final d in documents) {
      _checkVector(d.vector, 'documents["${d.id}"].vector');
      final metadata = d.metadata;
      if (metadata != null) {
        try {
          jsonEncode(metadata);
        } on JsonUnsupportedObjectError catch (e) {
          throw ArgumentError(
            'documents["${d.id}"].metadata is not JSON-encodable: '
            '${e.unsupportedObject}',
          );
        }
      }
    }
    await _send(_Request(_Op.upsert, List<Document>.of(documents)));
  }

  /// Inserts or replaces one document.
  Future<void> add(Document document) => upsert([document]);

  /// Deletes documents by id, returning how many existed.
  @override
  Future<int> delete(Iterable<String> ids) async =>
      await _send(_Request(_Op.delete, ids.toList(growable: false))) as int;

  /// The document with [id], or `null`.
  @override
  Future<Document?> get(String id, {bool withVector = false}) async =>
      await _send(_Request(_Op.get, id, withVector)) as Document?;

  /// Searches the collection; see [PhoenixCollection.search].
  Future<List<SearchHit>> search({
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
  @override
  Future<List<SearchHit>> query(CollectionQuery q) async {
    // Validated here as well as natively, so an obvious mistake fails at the
    // call site with the same error the synchronous client gives.
    if (q.k < 0) throw ArgumentError.value(q.k, 'k', 'must be >= 0');
    _checkVector(q.vector, 'vector');
    return ((await _send(_Request(_Op.query, q)))! as List).cast<SearchHit>();
  }

  /// Number of documents, or of those matching [filter].
  @override
  Future<int> count({Filter? filter}) async =>
      await _send(_Request(_Op.count, filter)) as int;

  /// Documents matching [filter] in id order; see [PhoenixCollection.list].
  @override
  Future<List<Document>> list({
    Filter? filter,
    int limit = 0,
    String? after,
    bool newestFirst = false,
  }) async {
    if (limit < 0) throw ArgumentError.value(limit, 'limit', 'must be >= 0');
    final raw = await _send(
      _Request(_Op.list, _Page(filter, limit, after, newestFirst)),
    );
    return (raw! as List).cast<Document>();
  }

  /// Every document matching [filter], fetched in pages of [pageSize],
  /// oldest first unless [newestFirst] is set.
  Stream<Document> documents({
    Filter? filter,
    int pageSize = 256,
    bool newestFirst = false,
  }) {
    // Validated before the stream exists, so a bad argument fails at the call
    // site rather than on the first `listen`.
    if (pageSize <= 0) {
      throw ArgumentError.value(pageSize, 'pageSize', 'must be positive');
    }
    return _documents(filter, pageSize, newestFirst);
  }

  Stream<Document> _documents(
    Filter? filter,
    int pageSize,
    bool newestFirst,
  ) async* {
    String? after;
    while (true) {
      final page = await list(
        filter: filter,
        limit: pageSize,
        after: after,
        newestFirst: newestFirst,
      );
      for (final d in page) {
        yield d;
      }
      if (page.length < pageSize) return;
      after = page.last.id;
    }
  }

  /// Statistics.
  Future<CollectionStats> stats() async =>
      await _send(const _Request(_Op.stats)) as CollectionStats;

  /// Checks the collection and reports what it found, without changing it.
  Future<CollectionReport> verify() async =>
      await _send(const _Request(_Op.verify)) as CollectionReport;

  /// Reclaims space; returns how many vector records were reclaimed.
  Future<int> compact() async =>
      await _send(const _Request(_Op.compact)) as int;

  /// Writes a consistent, compacted copy of the collection into [dir].
  Future<void> backup(String dir) => _send(_Request(_Op.backup, dir));

  /// Committed document changes as they happen: one [CollectionChange] per
  /// document written or removed, in commit order.
  ///
  /// The subscription runs on its own isolate — the blocking native poll
  /// cannot share the worker that serves the other calls — and ends when it is
  /// cancelled or the collection closes. A consumer that falls behind loses
  /// the oldest changes past [capacity]; rebuild from [list] if that matters.
  ///
  /// ```dart
  /// final sub = kb.changes().listen((c) => print('${c.kind} ${c.id}'));
  /// // ... later
  /// await sub.cancel();
  /// ```
  Stream<CollectionChange> changes({int capacity = 1024}) {
    if (capacity <= 0) {
      throw ArgumentError.value(capacity, 'capacity', 'must be positive');
    }
    final spec = WatchSpec(
      path: path,
      libraryPath: libraryPath,
      capacity: capacity,
      collection: true,
    );
    late WatchSession session;
    late StreamController<CollectionChange> controller;
    controller = StreamController<CollectionChange>(
      onListen: () {
        session = WatchSession.start(spec);
        _watchers.add(session);
        session.batches.listen(
          (batch) {
            for (final change in batch) {
              controller.add(change as CollectionChange);
            }
          },
          onError: controller.addError,
          onDone: () {
            _watchers.remove(session);
            if (!controller.isClosed) controller.close();
          },
        );
      },
      onCancel: () async {
        await session.stop();
        _watchers.remove(session);
      },
    );
    return controller.stream;
  }

  /// Syncs vectors and checkpoints documents.
  Future<void> flush() => _send(const _Request(_Op.flush));

  /// Closes the collection, ends any [changes] streams, and stops the
  /// worker. Idempotent.
  Future<void> close() async {
    for (final session in _watchers.toList()) {
      await session.stop();
    }
    _watchers.clear();
    await _worker.close(const _Request(_Op.close));
  }
}
