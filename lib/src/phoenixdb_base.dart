/// Synchronous PhoenixDB API built directly on the C ABI.
///
/// Every method here blocks the calling isolate for the duration of the native
/// call. Use `AsyncPhoenixDB` from `package:phoenixdb/phoenixdb.dart` to keep
/// disk I/O off the UI thread.
library;

import 'dart:collection';
import 'dart:convert' show jsonDecode, jsonEncode;
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'kv.dart';
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

  /// True when the file is locked by another process.
  bool get isBusy => status == PhoenixStatus.busy;

  /// True when the transaction id is unknown or already finished.
  bool get isTxnNotFound => status == PhoenixStatus.txnNotFound;

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
  /// [options] tunes the engine (see [PhoenixOptions]); [cachePages], when
  /// non-zero, overrides `options.cachePages`. [libraryPath] overrides native
  /// library discovery.
  ///
  /// Opening a path this process already has open — from this isolate or any
  /// other — returns a new handle to the same engine, so several isolates can
  /// share one database safely; close every handle. A file held open by
  /// another *process* fails with a [PhoenixException] whose `isBusy` is set.
  static PhoenixDatabase open(
    String path, {
    int cachePages = 0,
    String? libraryPath,
    PhoenixOptions? options,
  }) {
    final bindings = PhoenixBindings.load(path: libraryPath);
    final opts = options ?? const PhoenixOptions();
    final pathPtr = path.toNativeUtf8();
    final outHandle = calloc<Pointer<PhoenixDB>>();
    final nativeOptions = calloc<PhoenixOptionsStruct>();
    try {
      nativeOptions.ref
        ..structSize = sizeOf<PhoenixOptionsStruct>()
        ..syncOnCommit = opts.syncOnCommit ? 1 : 0
        ..cachePages = cachePages != 0 ? cachePages : opts.cachePages
        ..checkpointBytes = opts.checkpointBytes
        ..tracing = opts.tracing ? 1 : 0
        ..fillFactorMax = opts.fillFactor ?? 0;
      final status = bindings.openEx(pathPtr, nativeOptions, outHandle);
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
      final db = PhoenixDatabase._(bindings, _HandleOwner(handle), finalizer);
      db._recordTrace('open(path=$path)');
      return db;
    } finally {
      calloc.free(pathPtr);
      calloc.free(outHandle);
      calloc.free(nativeOptions);
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

  static String _preview(Uint8List key) {
    final shown = key.length <= 32 ? key : key.sublist(0, 32);
    final hex = shown.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
    return key.length <= 32 ? '0x$hex' : '0x$hex... (${key.length} bytes)';
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

  /// Collects tracing events emitted by this database instance.
  ///
  /// Attach a listener before operations you want to observe, and detach it
  /// afterwards. Only one listener can be active at a time.
  void setTraceListener(void Function(String event) listener) {
    _ensureOpen();
    _traceListener = listener;
  }

  /// Removes the active trace listener, if any.
  void clearTraceListener() {
    _traceListener = null;
  }

  /// The most recent trace events (at most [maxTraceEvents]), oldest first.
  List<String> get traceEvents {
    return List<String>.unmodifiable(_traceEvents);
  }

  /// How many recent events [traceEvents] retains.
  ///
  /// Events are kept in a bounded ring: the list used to grow by one string
  /// per call for the lifetime of the handle, which leaked memory in any
  /// long-running app.
  static const int maxTraceEvents = 256;

  void _recordTrace(String event) {
    if (_traceEvents.length == maxTraceEvents) _traceEvents.removeFirst();
    _traceEvents.add(event);
    _traceListener?.call(event);
  }

  final ListQueue<String> _traceEvents = ListQueue<String>(maxTraceEvents);
  void Function(String event)? _traceListener;

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
      final txn = out.value;
      _recordTrace('begin(txn=$txn, readOnly=$readOnly)');
      return txn;
    } finally {
      calloc.free(out);
    }
  }

  /// Commits [txnId], making its writes durable before returning.
  void commit(int txnId) {
    _ensureOpen();
    final status = _b.commitTxn(_owner.pointer, txnId);
    if (status != PhoenixStatus.ok) _throw(status, 'commit($txnId)');
    _recordTrace('commit(txn=$txnId)');
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
      _recordTrace('insert(key=${_preview(key)}, txn=${txnId ?? 0})');
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
      final value = _takeBuffer(out);
      _recordTrace('get(key=${_preview(key)}, txn=${txnId ?? 0})');
      return value;
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

  /// Calls [callback] for every visible key/value pair in ascending order.
  ///
  /// Pairs are streamed one at a time; nothing is accumulated. The native
  /// side holds the engine's shared lock for the duration of the scan, so the
  /// callback must return quickly and must not call back into PhoenixDB. An
  /// exception thrown by [callback] stops the scan and is rethrown here.
  void scanIter(void Function(Uint8List key, Uint8List value) callback) {
    scanWhile((key, value) {
      callback(key, value);
      return true;
    });
  }

  /// Like [scanIter], but stops as soon as [callback] returns `false`.
  void scanWhile(bool Function(Uint8List key, Uint8List value) callback) {
    _ensureOpen();
    final frame = _ScanFrame(callback);
    _scanFrames.add(frame);
    final int rc;
    try {
      rc = _b.scanIter(_owner.pointer, _scanCallbackPointer);
    } finally {
      _scanFrames.removeLast();
    }
    final error = frame.error;
    if (error != null) Error.throwWithStackTrace(error, frame.stackTrace!);
    if (rc == PhoenixStatus.aborted) return; // the callback asked to stop
    if (rc != PhoenixStatus.ok) _throw(rc, 'scanIter');
  }

  /// Returns the pairs with keys between [start] and [end], ascending.
  ///
  /// [start] is inclusive and [end] exclusive by default; `null` leaves that
  /// side open. [limit] caps the number of pairs (`0` = no cap). With [txnId]
  /// the scan sees that transaction's snapshot and its own uncommitted
  /// writes; otherwise it reads the latest committed state.
  List<PhoenixEntry> scan({
    Uint8List? start,
    Uint8List? end,
    bool startInclusive = true,
    bool endInclusive = false,
    int limit = 0,
    int? txnId,
  }) {
    _ensureOpen();
    return _scanPage(
      start,
      start == null ? 0 : (startInclusive ? 1 : 2),
      end,
      end == null ? 0 : (endInclusive ? 1 : 2),
      limit,
      0,
      txnId,
    );
  }

  /// Returns the pairs whose key starts with [prefix], ascending.
  List<PhoenixEntry> scanPrefix(Uint8List prefix, {int limit = 0, int? txnId}) {
    final end = prefixSuccessor(prefix);
    return scan(
      start: prefix.isEmpty ? null : prefix,
      end: end,
      limit: limit,
      txnId: txnId,
    );
  }

  /// Lazily iterates pairs between [start] and [end] (or with [prefix]),
  /// fetching [pageSize] pairs per native call.
  ///
  /// Nothing is held open between pages, so this is safe to abandon at any
  /// point. Each page reads the latest committed state; pass a read-only
  /// [txnId] for one consistent snapshot across all pages.
  Iterable<PhoenixEntry> entries({
    Uint8List? start,
    Uint8List? end,
    Uint8List? prefix,
    int pageSize = 256,
    int? txnId,
  }) sync* {
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
    var loMode = start == null ? 0 : 1;
    while (true) {
      _ensureOpen();
      final page = _scanPage(
        lo,
        loMode,
        end,
        end == null ? 0 : 2,
        pageSize,
        4 << 20,
        txnId,
      );
      yield* page;
      if (page.length < pageSize) return;
      lo = page.last.key;
      loMode = 2; // continue strictly after the last key seen
    }
  }

  List<PhoenixEntry> _scanPage(
    Uint8List? lo,
    int loMode,
    Uint8List? hi,
    int hiMode,
    int limit,
    int maxBytes,
    int? txnId,
  ) {
    if (limit < 0) throw ArgumentError.value(limit, 'limit', 'must be >= 0');
    final loPtr = lo == null ? nullptr : _copyToNative(lo);
    final hiPtr = hi == null ? nullptr : _copyToNative(hi);
    final out = calloc<PhoenixBuffer>();
    try {
      final status = _b.scanRange(
        _owner.pointer,
        txnId ?? 0,
        loPtr.cast(),
        lo?.length ?? 0,
        loMode,
        hiPtr.cast(),
        hi?.length ?? 0,
        hiMode,
        limit,
        maxBytes,
        out,
      );
      if (status != PhoenixStatus.ok) _throw(status, 'scan');
      return decodeEntries(_takeBuffer(out));
    } finally {
      _b.bufferFree(out);
      if (loPtr != nullptr) calloc.free(loPtr);
      if (hiPtr != nullptr) calloc.free(hiPtr);
      calloc.free(out);
    }
  }

  /// Applies every write in [batch] atomically.
  void write(WriteBatch batch) {
    _ensureOpen();
    if (batch.isEmpty) return;
    final bytes = batch.toBytes();
    final ptr = _copyToNative(bytes);
    try {
      final status = _b.writeBatch(_owner.pointer, ptr, bytes.length);
      if (status != PhoenixStatus.ok) _throw(status, 'write');
      _recordTrace('write(batch=${batch.length})');
    } finally {
      calloc.free(ptr);
    }
  }

  /// Builds a [WriteBatch] with [build] and applies it atomically.
  void writeBatch(void Function(WriteBatch batch) build) {
    final batch = WriteBatch();
    build(batch);
    write(batch);
  }

  /// Writes a consistent, compacted, self-contained copy of the database to
  /// [path] while other callers keep reading and writing.
  void backup(String path) => _pathOp(_b.backup, path, 'backup');

  /// Replaces the database contents with the backup at [path], atomically.
  /// Fails while any transaction is open on this database.
  void restore(String path) => _pathOp(_b.restore, path, 'restore');

  void _pathOp(PathOpDart op, String path, String what) {
    _ensureOpen();
    final ptr = path.toNativeUtf8();
    try {
      final status = op(_owner.pointer, ptr);
      if (status != PhoenixStatus.ok) _throw(status, '$what("$path")');
    } finally {
      calloc.free(ptr);
    }
  }

  /// Rebuilds the file with live data only, returning free space to the
  /// filesystem. Blocks other callers while it runs.
  void compact() {
    _ensureOpen();
    final status = _b.compact(_owner.pointer);
    if (status != PhoenixStatus.ok) _throw(status, 'compact');
  }

  /// Runtime statistics.
  PhoenixStats stats() {
    _ensureOpen();
    final out = calloc<PhoenixStatsStruct>();
    try {
      final status = _b.stats(_owner.pointer, out);
      if (status != PhoenixStatus.ok) _throw(status, 'stats');
      final s = out.ref;
      return PhoenixStats(
        pageCount: s.pageCount,
        activeTransactions: s.activeTxns,
        pendingKeys: s.pendingKeys,
        walBytes: s.walBytes,
        commitTimestamp: s.commitTs,
        treeTimestamp: s.treeTs,
        cacheHits: s.cacheHits,
        cacheMisses: s.cacheMisses,
      );
    } finally {
      calloc.free(out);
    }
  }

  /// Full structural check of the B+Tree; throws a [PhoenixException] with
  /// `isCorruption` set when any invariant is violated.
  PhoenixTreeReport check() {
    _ensureOpen();
    final out = calloc<PhoenixTreeReportStruct>();
    try {
      final status = _b.check(_owner.pointer, out);
      if (status != PhoenixStatus.ok) _throw(status, 'check');
      final r = out.ref;
      return PhoenixTreeReport(
        depth: r.depth,
        keys: r.keys,
        leafPages: r.leafPages,
        internalPages: r.internalPages,
        overflowPages: r.overflowPages,
        freePages: r.freePages,
        unreachablePages: r.unreachablePages,
        underfullLeaves: r.underfullLeaves,
      );
    } finally {
      calloc.free(out);
    }
  }

  /// Turns engine span recording on or off; see [spans].
  void setTracing(bool enabled) {
    _ensureOpen();
    final status = _b.setTracing(_owner.pointer, enabled ? 1 : 0);
    if (status != PhoenixStatus.ok) _throw(status, 'setTracing');
  }

  /// Spans recorded while tracing was on (the most recent 1024).
  List<TraceSpan> spans() {
    final json = _takeString(
      (out) => _b.spansJson(_owner.pointer, out),
      'spans',
    );
    return (jsonDecode(json) as List<Object?>)
        .map((e) => TraceSpan.fromJson(e as Map<String, Object?>))
        .toList(growable: false);
  }

  /// Metrics in the Prometheus text exposition format.
  String metricsPrometheus() => _takeString(
    (out) => _b.metricsText(_owner.pointer, 1, out),
    'metricsPrometheus',
  );

  /// Calls a native function that yields an owned string, and copies it.
  String _takeString(
    int Function(Pointer<Pointer<Utf8>> out) call,
    String what,
  ) {
    _ensureOpen();
    final out = calloc<Pointer<Utf8>>();
    try {
      final status = call(out);
      if (status != PhoenixStatus.ok) _throw(status, what);
      final ptr = out.value;
      if (ptr == nullptr) return '';
      try {
        return ptr.toDartString();
      } finally {
        _b.stringFree(ptr);
      }
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
  /// Bind user data with [params] rather than splicing it into [sql]: each
  /// `?` (or `?N`, 1-based) takes the next value (or the Nth), and a value
  /// can never change the statement's meaning. Values may be `null`, `bool`
  /// (stored as 0/1), `int`, `double` or `String`.
  ///
  /// ```dart
  /// db.query('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)');
  /// db.query('INSERT INTO users VALUES (?, ?)', params: [1, userInput]);
  /// final r = db.query('SELECT name FROM users WHERE id = ?', params: [1]);
  /// print(r.scalar);
  /// ```
  ///
  /// Without [txnId] the statement runs in its own transaction, retried
  /// transparently on a write conflict. With [txnId] it joins that
  /// transaction (atomically: a failing statement stages nothing) and the
  /// caller commits — so SQL and key/value writes can commit together.
  SqlResult query(String sql, {List<Object?>? params, int? txnId}) {
    _ensureOpen();
    final sqlPtr = sql.toNativeUtf8();
    final paramsPtr = params == null || params.isEmpty
        ? nullptr
        : _encodeParams(params).toNativeUtf8();
    final outPtr = calloc<Pointer<Utf8>>();
    try {
      final status = _b.sqlQueryParams(
        _owner.pointer,
        txnId ?? 0,
        sqlPtr,
        paramsPtr,
        outPtr,
      );
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
      if (paramsPtr != nullptr) calloc.free(paramsPtr);
      calloc.free(outPtr);
    }
  }

  static String _encodeParams(List<Object?> params) {
    for (var i = 0; i < params.length; i++) {
      final p = params[i];
      if (p != null && p is! bool && p is! num && p is! String) {
        throw ArgumentError.value(
          p,
          'params[$i]',
          'must be null, bool, int, double or String',
        );
      }
      if (p is double && !p.isFinite) {
        throw ArgumentError.value(p, 'params[$i]', 'must be finite');
      }
    }
    return jsonEncode(params);
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

  /// Returns the engine's metrics report as a human-readable string: WAL
  /// fsync latency percentiles, cache hit ratio, commit/read/write/scan and
  /// checkpoint statistics.
  String metricsReport() => _takeString(
    (out) => _b.metricsText(_owner.pointer, 0, out),
    'metricsReport',
  );
}

/// State of one in-flight [PhoenixDatabase.scanWhile] call.
final class _ScanFrame {
  final bool Function(Uint8List key, Uint8List value) callback;
  Object? error;
  StackTrace? stackTrace;

  _ScanFrame(this.callback);
}

/// Active scans on this isolate, innermost last.
///
/// Native callbacks cannot capture a closure, so the callback reaches its
/// Dart handler through this stack; a stack rather than a single slot keeps a
/// scan started from inside another scan's callback from clobbering it.
final List<_ScanFrame> _scanFrames = <_ScanFrame>[];

/// Returning `1` (also the value used if this throws) stops the scan.
final Pointer<NativeFunction<NativeScanIterCallback>> _scanCallbackPointer =
    Pointer.fromFunction<NativeScanIterCallback>(_scanCallback, 1);

int _scanCallback(
  Pointer<Uint8> key,
  int keyLen,
  Pointer<Uint8> value,
  int valueLen,
) {
  if (_scanFrames.isEmpty) return 1;
  final frame = _scanFrames.last;
  try {
    // Copy out: the native memory is only valid during this call.
    final k = Uint8List.fromList(key.asTypedList(keyLen));
    final v = Uint8List.fromList(value.asTypedList(valueLen));
    return frame.callback(k, v) ? 0 : 1;
  } catch (e, st) {
    frame.error = e;
    frame.stackTrace = st;
    return 1;
  }
}
