/// Isolate-based async wrapper that keeps blocking disk I/O off the UI thread.
///
/// A single long-lived worker isolate owns the native handle. Because the
/// handle never crosses an isolate boundary, there is no shared-memory hazard:
/// requests and responses are plain data sent over ports.
///
/// The worker is supervised: if it dies (a native abort, an uncaught error),
/// every in-flight and later call fails with a [PhoenixException] instead of
/// waiting forever.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'kv.dart';
import 'phoenixdb_base.dart';
import 'sql_result.dart';
import 'worker.dart';

/// Operations the worker understands.
enum _Op {
  insert,
  get,
  delete,
  count,
  checkpoint,
  flush,
  verify,
  begin,
  commit,
  rollback,
  query,
  close,
  scan,
  write,
  backup,
  restore,
  compact,
  stats,
  check,
  metrics,
  metricsPrometheus,
  setTracing,
  spans,
}

/// Arguments of a range scan.
class _ScanArgs {
  final Uint8List? start;
  final Uint8List? end;
  final bool startInclusive;
  final bool endInclusive;
  final int limit;

  const _ScanArgs(
    this.start,
    this.end,
    this.startInclusive,
    this.endInclusive,
    this.limit,
  );
}

/// A request sent to the worker isolate.
class _Request {
  final _Op op;
  final Uint8List? key;
  final Uint8List? value;
  final int? txnId;
  final bool flag;
  final String? text;
  final Object? args;

  const _Request(
    this.op, {
    this.key,
    this.value,
    this.txnId,
    this.flag = false,
    this.text,
    this.args,
  });
}

/// Startup payload for the worker isolate.
class _Boot {
  final SendPort ready;
  final String path;
  final int cachePages;
  final String? libraryPath;
  final PhoenixOptions? options;

  const _Boot(
    this.ready,
    this.path,
    this.cachePages,
    this.libraryPath,
    this.options,
  );
}

/// Executes one request against the worker's database.
Object? _execute(PhoenixDatabase db, _Request r) {
  switch (r.op) {
    case _Op.insert:
      db.insert(r.key!, r.value!, txnId: r.txnId);
      return null;
    case _Op.get:
      return db.get(r.key!, txnId: r.txnId);
    case _Op.delete:
      return db.delete(r.key!, txnId: r.txnId);
    case _Op.count:
      return db.count();
    case _Op.checkpoint:
      db.checkpoint();
      return null;
    case _Op.flush:
      db.flush();
      return null;
    case _Op.verify:
      db.verify();
      return null;
    case _Op.begin:
      return db.beginTransaction(readOnly: r.flag);
    case _Op.commit:
      db.commit(r.txnId!);
      return null;
    case _Op.rollback:
      db.rollback(r.txnId!);
      return null;
    // The result crosses the isolate boundary as JSON: SqlResult is not a
    // transferable type, and re-parsing on the far side is cheap next to the
    // query itself.
    case _Op.query:
      return db
          .query(r.text!, params: r.args as List<Object?>?, txnId: r.txnId)
          .toJsonString();
    case _Op.close:
      db.close();
      return null;
    case _Op.scan:
      final a = r.args! as _ScanArgs;
      return db.scan(
        start: a.start,
        end: a.end,
        startInclusive: a.startInclusive,
        endInclusive: a.endInclusive,
        limit: a.limit,
        txnId: r.txnId,
      );
    case _Op.write:
      db.write(r.args! as WriteBatch);
      return null;
    case _Op.backup:
      db.backup(r.text!);
      return null;
    case _Op.restore:
      db.restore(r.text!);
      return null;
    case _Op.compact:
      db.compact();
      return null;
    case _Op.stats:
      return db.stats();
    case _Op.check:
      return db.check();
    case _Op.metrics:
      return db.metricsReport();
    case _Op.metricsPrometheus:
      return db.metricsPrometheus();
    case _Op.setTracing:
      db.setTracing(r.flag);
      return null;
    case _Op.spans:
      return db.spans();
  }
}

/// Worker entry point: opens the database, then serves requests until closed.
void _workerMain(_Boot boot) => serveWorker<PhoenixDatabase>(
  boot.ready,
  () => PhoenixDatabase.open(
    boot.path,
    cachePages: boot.cachePages,
    libraryPath: boot.libraryPath,
    options: boot.options,
  ),
  (db, request) => _execute(db, request! as _Request),
  (request) => request is _Request && request.op == _Op.close,
);

/// Asynchronous PhoenixDB client backed by a dedicated worker isolate.
///
/// ```dart
/// final db = await AsyncPhoenixDB.open('data.pdb');
/// await db.insert(key, value);
/// final value = await db.get(key);
/// await db.close();
/// ```
class AsyncPhoenixDB {
  final WorkerClient _worker;

  AsyncPhoenixDB._(this._worker);

  /// Spawns the worker isolate and opens the database at [path].
  ///
  /// See [PhoenixDatabase.open] for [options] and sharing semantics.
  static Future<AsyncPhoenixDB> open(
    String path, {
    int cachePages = 0,
    String? libraryPath,
    PhoenixOptions? options,
  }) async => AsyncPhoenixDB._(
    await WorkerClient.spawn<_Boot>(
      _workerMain,
      (ready) => _Boot(ready, path, cachePages, libraryPath, options),
      debugName: 'phoenixdb-worker',
      what: 'database',
    ),
  );

  /// Whether [close] has already run (or the worker died).
  bool get isClosed => _worker.isClosed;

  Future<Object?> _send(_Request request) => _worker.call(request);

  /// Begins a transaction and returns its id.
  Future<int> beginTransaction({bool readOnly = false}) async =>
      await _send(_Request(_Op.begin, flag: readOnly)) as int;

  /// Commits [txnId].
  Future<void> commit(int txnId) => _send(_Request(_Op.commit, txnId: txnId));

  /// Rolls [txnId] back.
  Future<void> rollback(int txnId) =>
      _send(_Request(_Op.rollback, txnId: txnId));

  /// Inserts or replaces [key] with [value].
  Future<void> insert(Uint8List key, Uint8List value, {int? txnId}) =>
      _send(_Request(_Op.insert, key: key, value: value, txnId: txnId));

  /// Reads [key], returning `null` when it does not exist.
  Future<Uint8List?> get(Uint8List key, {int? txnId}) async =>
      await _send(_Request(_Op.get, key: key, txnId: txnId)) as Uint8List?;

  /// Deletes [key], returning `false` when it did not exist.
  Future<bool> delete(Uint8List key, {int? txnId}) async =>
      await _send(_Request(_Op.delete, key: key, txnId: txnId)) as bool;

  /// Number of visible keys.
  Future<int> count() async => await _send(_Request(_Op.count)) as int;

  /// Pairs with keys between [start] and [end]; see [PhoenixDatabase.scan].
  Future<List<PhoenixEntry>> scan({
    Uint8List? start,
    Uint8List? end,
    bool startInclusive = true,
    bool endInclusive = false,
    int limit = 0,
    int? txnId,
  }) async =>
      await _send(
            _Request(
              _Op.scan,
              txnId: txnId,
              args: _ScanArgs(start, end, startInclusive, endInclusive, limit),
            ),
          )
          as List<PhoenixEntry>;

  /// Pairs whose key starts with [prefix]; see [PhoenixDatabase.scanPrefix].
  Future<List<PhoenixEntry>> scanPrefix(
    Uint8List prefix, {
    int limit = 0,
    int? txnId,
  }) => scan(
    start: prefix.isEmpty ? null : prefix,
    end: prefixSuccessor(prefix),
    limit: limit,
    txnId: txnId,
  );

  /// Streams pairs page by page (each page is one worker round trip).
  ///
  /// Like [PhoenixDatabase.entries]: nothing is held open between pages;
  /// pass a read-only [txnId] for one consistent snapshot.
  Stream<PhoenixEntry> entries({
    Uint8List? start,
    Uint8List? end,
    Uint8List? prefix,
    int pageSize = 256,
    int? txnId,
  }) async* {
    if (pageSize <= 0) {
      throw ArgumentError.value(pageSize, 'pageSize', 'must be positive');
    }
    if (prefix != null) {
      if (start != null || end != null) {
        throw ArgumentError('pass either prefix or start/end, not both');
      }
      start = prefix.isEmpty ? null : prefix;
      end = prefixSuccessor(prefix);
    }
    var lo = start;
    var inclusive = true;
    while (true) {
      final page = await scan(
        start: lo,
        end: end,
        startInclusive: inclusive,
        limit: pageSize,
        txnId: txnId,
      );
      for (final entry in page) {
        yield entry;
      }
      if (page.length < pageSize) return;
      lo = page.last.key;
      inclusive = false;
    }
  }

  /// Applies every write in [batch] atomically.
  Future<void> write(WriteBatch batch) async {
    if (batch.isEmpty) return;
    await _send(_Request(_Op.write, args: batch));
  }

  /// Builds a [WriteBatch] with [build] and applies it atomically.
  Future<void> writeBatch(void Function(WriteBatch batch) build) {
    final batch = WriteBatch();
    build(batch);
    return write(batch);
  }

  /// Runs a SQL statement on the worker isolate; see
  /// [PhoenixDatabase.query] for [params] and [txnId].
  ///
  /// The parse and execution happen off the calling isolate, so a slow query
  /// never blocks a Flutter UI frame.
  ///
  /// ```dart
  /// await db.query('CREATE TABLE users (id INTEGER, name TEXT)');
  /// final r = await db.query('SELECT name FROM users WHERE id = 1');
  /// print(r.scalar); // alice
  /// ```
  Future<SqlResult> query(
    String sql, {
    List<Object?>? params,
    int? txnId,
  }) async => SqlResult.fromJson(
    await _send(
          _Request(
            _Op.query,
            text: sql,
            txnId: txnId,
            args: params == null ? null : List<Object?>.of(params),
          ),
        )
        as String,
  );

  /// Merges pending versions, flushes, and rewrites the WAL.
  Future<void> checkpoint() => _send(_Request(_Op.checkpoint));

  /// Syncs the WAL and flushes staged pages.
  Future<void> flush() => _send(_Request(_Op.flush));

  /// Verifies checksums and B+Tree invariants.
  Future<void> verify() => _send(_Request(_Op.verify));

  /// Full structural check; see [PhoenixDatabase.check].
  Future<PhoenixTreeReport> check() async =>
      await _send(_Request(_Op.check)) as PhoenixTreeReport;

  /// Writes a consistent, compacted backup to [path].
  Future<void> backup(String path) => _send(_Request(_Op.backup, text: path));

  /// Replaces the contents with the backup at [path].
  Future<void> restore(String path) => _send(_Request(_Op.restore, text: path));

  /// Rebuilds the file with live data only.
  Future<void> compact() => _send(_Request(_Op.compact));

  /// Runtime statistics.
  Future<PhoenixStats> stats() async =>
      await _send(_Request(_Op.stats)) as PhoenixStats;

  /// Human-readable metrics report.
  Future<String> metricsReport() async =>
      await _send(_Request(_Op.metrics)) as String;

  /// Metrics in the Prometheus text format.
  Future<String> metricsPrometheus() async =>
      await _send(_Request(_Op.metricsPrometheus)) as String;

  /// Turns engine span recording on or off.
  Future<void> setTracing(bool enabled) =>
      _send(_Request(_Op.setTracing, flag: enabled));

  /// Spans recorded while tracing was on.
  Future<List<TraceSpan>> spans() async =>
      await _send(_Request(_Op.spans)) as List<TraceSpan>;

  /// Runs [body] in a transaction, committing on success and rolling back on
  /// failure. Retries [retries] times on a write-write conflict.
  Future<T> transaction<T>(
    Future<T> Function(int txnId) body, {
    int retries = 3,
  }) async {
    var attempt = 0;
    while (true) {
      final txn = await beginTransaction();
      try {
        final result = await body(txn);
        await commit(txn);
        return result;
      } on PhoenixException catch (e) {
        try {
          await rollback(txn);
        } on PhoenixException {
          // Already finished; keep the original failure.
        }
        if (e.isConflict && attempt < retries) {
          attempt++;
          continue;
        }
        rethrow;
      } catch (_) {
        try {
          await rollback(txn);
        } on PhoenixException {
          // Ignore: the caller's error is the interesting one.
        }
        rethrow;
      }
    }
  }

  /// Closes the database and shuts the worker isolate down. Idempotent.
  Future<void> close() => _worker.close(const _Request(_Op.close));
}
