/// Isolate-based async wrapper that keeps blocking disk I/O off the UI thread.
///
/// A single long-lived worker isolate owns the native handle. Because the
/// handle never crosses an isolate boundary, there is no shared-memory hazard:
/// requests and responses are plain data sent over ports.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'phoenixdb_base.dart';
import 'sql_result.dart';

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
}

/// A request sent to the worker isolate.
class _Request {
  final int id;
  final _Op op;
  final Uint8List? key;
  final Uint8List? value;
  final int? txnId;
  final bool readOnly;
  final String? sql;

  const _Request(
    this.id,
    this.op, {
    this.key,
    this.value,
    this.txnId,
    this.readOnly = false,
    this.sql,
  });
}

/// A response returned by the worker isolate.
class _Response {
  final int id;
  final Object? result;
  final String? error;
  final int? status;

  const _Response(this.id, {this.result, this.error, this.status});
}

/// Startup payload for the worker isolate.
class _Boot {
  final SendPort ready;
  final String path;
  final int cachePages;
  final String? libraryPath;

  const _Boot(this.ready, this.path, this.cachePages, this.libraryPath);
}

/// Worker entry point: opens the database, then serves requests until closed.
void _workerMain(_Boot boot) {
  final commands = ReceivePort();
  PhoenixDatabase db;
  try {
    db = PhoenixDatabase.open(
      boot.path,
      cachePages: boot.cachePages,
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
    final request = message[0] as _Request;
    final reply = message[1] as SendPort;

    try {
      final Object? result = switch (request.op) {
        _Op.insert => () {
          db.insert(request.key!, request.value!, txnId: request.txnId);
          return null;
        }(),
        _Op.get => db.get(request.key!, txnId: request.txnId),
        _Op.delete => db.delete(request.key!, txnId: request.txnId),
        _Op.count => db.count(),
        _Op.checkpoint => () {
          db.checkpoint();
          return null;
        }(),
        _Op.flush => () {
          db.flush();
          return null;
        }(),
        _Op.verify => () {
          db.verify();
          return null;
        }(),
        _Op.begin => db.beginTransaction(readOnly: request.readOnly),
        _Op.commit => () {
          db.commit(request.txnId!);
          return null;
        }(),
        _Op.rollback => () {
          db.rollback(request.txnId!);
          return null;
        }(),
        // The result crosses the isolate boundary as JSON: SqlResult is not
        // a transferable type, and re-parsing on the far side is cheap next
        // to the query itself.
        _Op.query => db.query(request.sql!).toJsonString(),
        _Op.close => () {
          db.close();
          return null;
        }(),
      };
      reply.send(_Response(request.id, result: result));
      if (request.op == _Op.close) {
        commands.close();
      }
    } on PhoenixException catch (e) {
      reply.send(_Response(request.id, error: e.message, status: e.status));
    } catch (e) {
      reply.send(_Response(request.id, error: e.toString()));
    }
  });
}

/// Asynchronous PhoenixDB client backed by a dedicated worker isolate.
///
/// ```dart
/// final db = await AsyncPhoenixDB.open('data.pdb');
/// await db.insert(key, value);
/// final value = await db.get(key);
/// await db.close();
/// ```
class AsyncPhoenixDB {
  final SendPort _commands;
  final ReceivePort _responses;
  final Map<int, Completer<Object?>> _pending = {};
  int _nextId = 1;
  bool _closed = false;

  AsyncPhoenixDB._(this._commands, this._responses) {
    _responses.listen((message) {
      if (message is! _Response) return;
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

  /// Spawns the worker isolate and opens the database at [path].
  static Future<AsyncPhoenixDB> open(
    String path, {
    int cachePages = 0,
    String? libraryPath,
  }) async {
    final ready = ReceivePort();
    await Isolate.spawn(
      _workerMain,
      _Boot(ready.sendPort, path, cachePages, libraryPath),
      debugName: 'phoenixdb-worker',
    );
    final first = await ready.first;
    ready.close();
    if (first is String) {
      throw PhoenixException(-1, 'failed to open database: $first');
    }
    return AsyncPhoenixDB._(first as SendPort, ReceivePort());
  }

  /// Whether [close] has already run.
  bool get isClosed => _closed;

  Future<Object?> _send(_Request request) {
    if (_closed) {
      return Future.error(const PhoenixException(-2, 'database is closed'));
    }
    final completer = Completer<Object?>();
    _pending[request.id] = completer;
    _commands.send([request, _responses.sendPort]);
    return completer.future;
  }

  int get _id => _nextId++;

  /// Begins a transaction and returns its id.
  Future<int> beginTransaction({bool readOnly = false}) async =>
      await _send(_Request(_id, _Op.begin, readOnly: readOnly)) as int;

  /// Commits [txnId].
  Future<void> commit(int txnId) =>
      _send(_Request(_id, _Op.commit, txnId: txnId));

  /// Rolls [txnId] back.
  Future<void> rollback(int txnId) =>
      _send(_Request(_id, _Op.rollback, txnId: txnId));

  /// Inserts or replaces [key] with [value].
  Future<void> insert(Uint8List key, Uint8List value, {int? txnId}) =>
      _send(_Request(_id, _Op.insert, key: key, value: value, txnId: txnId));

  /// Reads [key], returning `null` when it does not exist.
  Future<Uint8List?> get(Uint8List key, {int? txnId}) async =>
      await _send(_Request(_id, _Op.get, key: key, txnId: txnId)) as Uint8List?;

  /// Deletes [key], returning `false` when it did not exist.
  Future<bool> delete(Uint8List key, {int? txnId}) async =>
      await _send(_Request(_id, _Op.delete, key: key, txnId: txnId)) as bool;

  /// Number of visible keys.
  Future<int> count() async => await _send(_Request(_id, _Op.count)) as int;

  /// Runs a SQL statement on the worker isolate.
  ///
  /// The parse and execution happen off the calling isolate, so a slow query
  /// never blocks a Flutter UI frame.
  ///
  /// ```dart
  /// await db.query('CREATE TABLE users (id INTEGER, name TEXT)');
  /// final r = await db.query('SELECT name FROM users WHERE id = 1');
  /// print(r.scalar); // alice
  /// ```
  Future<SqlResult> query(String sql) async => SqlResult.fromJson(
    await _send(_Request(_id, _Op.query, sql: sql)) as String,
  );

  /// Merges pending versions, flushes, and truncates the WAL.
  Future<void> checkpoint() => _send(_Request(_id, _Op.checkpoint));

  /// Flushes dirty pages without truncating the WAL.
  Future<void> flush() => _send(_Request(_id, _Op.flush));

  /// Verifies checksums and B+Tree invariants.
  Future<void> verify() => _send(_Request(_id, _Op.verify));

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

  /// Closes the database and shuts the worker isolate down.
  Future<void> close() async {
    if (_closed) return;
    try {
      await _send(_Request(_id, _Op.close));
    } finally {
      _closed = true;
      _responses.close();
      for (final completer in _pending.values) {
        if (!completer.isCompleted) {
          completer.completeError(
            const PhoenixException(
              -1,
              'database closed while a call was in flight',
            ),
          );
        }
      }
      _pending.clear();
    }
  }
}
