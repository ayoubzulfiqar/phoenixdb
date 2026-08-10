/// Synchronous PhoenixDB API built directly on the C ABI.
///
/// Every method here blocks the calling isolate for the duration of the native
/// call. Use `AsyncPhoenixDB` from `package:phoenixdb/phoenixdb.dart` to keep
/// disk I/O off the UI thread.
library;

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'sql_result.dart';

/// Thrown when a native call fails.
///
/// [status] is one of the [PhoenixStatus] constants; [message] carries the
/// engine's own description when one is available.
class PhoenixException implements Exception {
  /// Native status code (always negative).
  final int status;

  /// Human-readable description from the engine.
  final String message;

  /// Creates an exception for [status] with [message].
  const PhoenixException(this.status, this.message);

  /// True when the failure was a write-write conflict worth retrying.
  bool get isConflict => status == PhoenixStatus.conflict;

  /// True when the key was absent.
  bool get isNotFound => status == PhoenixStatus.notFound;

  /// True when the engine reported on-disk corruption.
  bool get isCorruption => status == PhoenixStatus.corruption;

  @override
  String toString() => 'PhoenixException($status): $message';
}

/// Thrown specifically when a key is missing, so callers can catch it narrowly.
class KeyNotFoundException extends PhoenixException {
  /// Creates a not-found error for [key].
  KeyNotFoundException(Uint8List key)
    : super(PhoenixStatus.notFound, 'key not found: ${_preview(key)}');

  static String _preview(Uint8List key) {
    final shown = key.length <= 32 ? key : key.sublist(0, 32);
    final hex = shown.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
    return key.length <= 32 ? '0x$hex' : '0x$hex... (${key.length} bytes)';
  }
}

/// Owns the native handle and releases it if the Dart object is collected.
///
/// [NativeFinalizer] guarantees `phoenix_close` runs even when a caller forgets
/// [PhoenixDatabase.close]; the explicit path detaches the finalizer first so
/// the handle is never closed twice.
class _HandleOwner implements Finalizable {
  final Pointer<PhoenixDB> pointer;

  _HandleOwner(this.pointer);
}

/// A synchronous handle to an open PhoenixDB database.
class PhoenixDatabase implements Finalizable {
  final PhoenixBindings _b;
  final _HandleOwner _owner;
  final NativeFinalizer _finalizer;
  bool _closed = false;

  PhoenixDatabase._(this._b, this._owner, this._finalizer) {
    // Attach with `externalSize` so the GC accounts for the native footprint.
    _finalizer.attach(
      this,
      _owner.pointer.cast(),
      detach: this,
      externalSize: 1 << 20,
    );
  }

  /// Opens (or creates) the database at [path].
  ///
  /// [cachePages] sets the clean-page cache size; 0 selects the engine default.
  /// [libraryPath] overrides native library discovery.
  static PhoenixDatabase open(
    String path, {
    int cachePages = 0,
    String? libraryPath,
  }) {
    final bindings = PhoenixBindings.load(path: libraryPath);
    final pathPtr = path.toNativeUtf8();
    final outHandle = calloc<Pointer<PhoenixDB>>();
    try {
      final status = bindings.open(pathPtr, cachePages, outHandle);
      if (status != PhoenixStatus.ok) {
        throw _errorFor(bindings, status, 'open("$path")');
      }
      final handle = outHandle.value;
      if (handle == nullptr) {
        throw const PhoenixException(
          PhoenixStatus.error,
          'open returned a null handle',
        );
      }
      final finalizer = NativeFinalizer(bindings.closePtr.cast());
      return PhoenixDatabase._(bindings, _HandleOwner(handle), finalizer);
    } finally {
      calloc.free(pathPtr);
      calloc.free(outHandle);
    }
  }

  /// Whether [close] has already run.
  bool get isClosed => _closed;

  /// Maximum key length the native layer accepts, in bytes.
  int get maxKeyLength => _b.maxKeyLen();

  /// Maximum value length the native layer accepts, in bytes.
  int get maxValueLength => _b.maxValueLen();

  /// Native ABI version.
  int get abiVersion => _b.abiVersion();

  void _ensureOpen() {
    if (_closed) {
      throw const PhoenixException(
        PhoenixStatus.invalidArgument,
        'database is closed',
      );
    }
  }

  static PhoenixException _errorFor(
    PhoenixBindings b,
    int status,
    String context,
  ) {
    final ptr = b.lastError();
    var detail = 'native call failed';
    if (ptr != nullptr) {
      try {
        detail = ptr.toDartString();
      } finally {
        b.stringFree(ptr);
      }
    }
    return PhoenixException(status, '$context: $detail');
  }

  Never _throw(int status, String context) =>
      throw _errorFor(_b, status, context);

  /// Begins a transaction and returns its id.
  ///
  /// Pass `readOnly: true` for a snapshot that cannot write but never blocks a
  /// writer at commit time.
  int beginTransaction({bool readOnly = false}) {
    _ensureOpen();
    final out = calloc<Uint64>();
    try {
      final status = _b.beginTxn(_owner.pointer, readOnly ? 1 : 0, out);
      if (status != PhoenixStatus.ok) _throw(status, 'beginTransaction');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// Commits [txnId], making its writes durable before returning.
  void commit(int txnId) {
    _ensureOpen();
    final status = _b.commitTxn(_owner.pointer, txnId);
    if (status != PhoenixStatus.ok) _throw(status, 'commit($txnId)');
  }

  /// Rolls [txnId] back, discarding its writes.
  void rollback(int txnId) {
    _ensureOpen();
    final status = _b.rollbackTxn(_owner.pointer, txnId);
    if (status != PhoenixStatus.ok) _throw(status, 'rollback($txnId)');
  }

  /// Inserts or replaces [key] with [value].
  ///
  /// When [txnId] is omitted the write runs in its own implicit transaction and
  /// is durable when this method returns.
  void insert(Uint8List key, Uint8List value, {int? txnId}) {
    _ensureOpen();
    final keyPtr = _copyToNative(key);
    final valuePtr = _copyToNative(value);
    try {
      final status = txnId == null
          ? _b.putAuto(
              _owner.pointer,
              keyPtr,
              key.length,
              valuePtr,
              value.length,
            )
          : _b.insert(
              _owner.pointer,
              txnId,
              keyPtr,
              key.length,
              valuePtr,
              value.length,
            );
      if (status != PhoenixStatus.ok) _throw(status, 'insert');
    } finally {
      calloc.free(keyPtr);
      calloc.free(valuePtr);
    }
  }

  /// Reads [key], returning `null` when it does not exist.
  Uint8List? get(Uint8List key, {int? txnId}) {
    _ensureOpen();
    final keyPtr = _copyToNative(key);
    final out = calloc<PhoenixBuffer>();
    try {
      final status = _b.get(
        _owner.pointer,
        txnId ?? 0,
        keyPtr,
        key.length,
        out,
      );
      if (status == PhoenixStatus.notFound) return null;
      if (status != PhoenixStatus.ok) _throw(status, 'get');
      return _takeBuffer(out);
    } finally {
      _b.bufferFree(out); // idempotent; the buffer is already drained
      calloc.free(keyPtr);
      calloc.free(out);
    }
  }

  /// Reads [key] or throws [KeyNotFoundException] when it is absent.
  Uint8List getOrThrow(Uint8List key, {int? txnId}) {
    final value = get(key, txnId: txnId);
    if (value == null) throw KeyNotFoundException(key);
    return value;
  }

  /// True when [key] exists.
  bool contains(Uint8List key, {int? txnId}) => get(key, txnId: txnId) != null;

  /// Deletes [key], returning `false` when it did not exist.
  bool delete(Uint8List key, {int? txnId}) {
    _ensureOpen();
    final keyPtr = _copyToNative(key);
    try {
      final status = _b.delete(_owner.pointer, txnId ?? 0, keyPtr, key.length);
      if (status == PhoenixStatus.notFound) return false;
      if (status != PhoenixStatus.ok) _throw(status, 'delete');
      return true;
    } finally {
      calloc.free(keyPtr);
    }
  }

  /// Number of visible keys.
  int count() {
    _ensureOpen();
    final out = calloc<Uint64>();
    try {
      final status = _b.count(_owner.pointer, out);
      if (status != PhoenixStatus.ok) _throw(status, 'count');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// Merges pending versions into the tree, flushes and truncates the WAL.
  void checkpoint() {
    _ensureOpen();
    final status = _b.checkpoint(_owner.pointer);
    if (status != PhoenixStatus.ok) _throw(status, 'checkpoint');
  }

  /// Flushes dirty pages and syncs the WAL without truncating it.
  void flush() {
    _ensureOpen();
    final status = _b.flush(_owner.pointer);
    if (status != PhoenixStatus.ok) _throw(status, 'flush');
  }

  /// Verifies every page checksum and the B+Tree ordering invariants.
  void verify() {
    _ensureOpen();
    final status = _b.verify(_owner.pointer);
    if (status != PhoenixStatus.ok) _throw(status, 'verify');
  }

  /// Runs [body] in a transaction, committing on success and rolling back on
  /// any error.
  ///
  /// Retries up to [retries] times when the engine reports a write-write
  /// conflict, since snapshot isolation makes conflicts a normal outcome.
  T transaction<T>(T Function(int txnId) body, {int retries = 3}) {
    _ensureOpen();
    var attempt = 0;
    while (true) {
      final txn = beginTransaction();
      try {
        final result = body(txn);
        commit(txn);
        return result;
      } on PhoenixException catch (e) {
        try {
          rollback(txn);
        } on PhoenixException {
          // The transaction is already finished; nothing further to undo.
        }
        if (e.isConflict && attempt < retries) {
          attempt++;
          continue;
        }
        rethrow;
      } catch (_) {
        try {
          rollback(txn);
        } on PhoenixException {
          // Ignore: propagate the caller's original error instead.
        }
        rethrow;
      }
    }
  }

  /// Runs a SQL statement and returns its result.
  ///
  /// Requires a native library built with the `sql` feature; check
  /// [supportsSql] first when targeting a lean embedded build.
  ///
  /// ```dart
  /// db.query('CREATE TABLE users (id INTEGER, name TEXT)');
  /// db.query("INSERT INTO users VALUES (1, 'alice')");
  /// final r = db.query('SELECT name FROM users WHERE id = 1');
  /// print(r.rows.first.first); // alice
  /// ```
  ///
  /// The heavy work happens in Rust; this call is synchronous, so prefer
  /// [PhoenixDatabaseAsync.query] on a UI isolate.
  SqlResult query(String sql) {
    _ensureOpen();
    final sqlPtr = sql.toNativeUtf8();
    final outPtr = calloc<Pointer<Utf8>>();
    try {
      final status = _b.sqlQuery(_owner.pointer, sqlPtr, outPtr);
      if (status != PhoenixStatus.ok) _throw(status, 'query');
      final json = outPtr.value;
      if (json == nullptr) {
        throw const PhoenixException(
          PhoenixStatus.error,
          'query returned no result document',
        );
      }
      try {
        return SqlResult.fromJson(json.toDartString());
      } finally {
        // The native side owns the buffer until we hand it back.
        _b.stringFree(json);
      }
    } finally {
      calloc.free(sqlPtr);
      calloc.free(outPtr);
    }
  }

  /// Whether the loaded native library was built with the SQL layer.
  bool get supportsSql => _b.hasSql() != 0;

  /// Checkpoints and closes the database.
  ///
  /// Safe to call more than once. Detaches the [NativeFinalizer] first so the
  /// handle cannot be closed twice.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer.detach(this);
    final status = _b.close(_owner.pointer);
    if (status != PhoenixStatus.ok) _throw(status, 'close');
  }

  /// Copies [data] into freshly allocated native memory.
  ///
  /// A one-byte allocation stands in for an empty list so the pointer is never
  /// null, keeping the native validation path unambiguous.
  static Pointer<Uint8> _copyToNative(Uint8List data) {
    final ptr = calloc<Uint8>(data.isEmpty ? 1 : data.length);
    if (data.isNotEmpty) {
      ptr.asTypedList(data.length).setAll(0, data);
    }
    return ptr;
  }

  /// Copies a native buffer into Dart memory and releases the native side.
  Uint8List _takeBuffer(Pointer<PhoenixBuffer> out) {
    final buffer = out.ref;
    if (buffer.ptr == nullptr || buffer.len == 0) return Uint8List(0);
    // Copy before freeing: the returned list must not alias native memory.
    final copy = Uint8List.fromList(buffer.ptr.asTypedList(buffer.len));
    _b.bufferFree(out);
    return copy;
  }
}
