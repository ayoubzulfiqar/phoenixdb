/// Reactive queries: watch committed changes instead of polling the database.
///
/// ```dart
/// final db = PhoenixDatabase.open('app.pdb');
/// final watcher = db.watch(prefix: utf8Key('user:'));
/// db.insert(utf8Key('user:1'), utf8Value('ada'));
/// for (final change in watcher.poll()) {
///   print('${change.kind} ${change.keyString}');
/// }
/// watcher.close();
/// ```
///
/// [ChangeWatcher.poll] blocks the calling isolate, which is what makes it
/// usable from a plain Dart program. In Flutter use the `Stream` APIs —
/// `AsyncPhoenixDB.changes()` and `AsyncPhoenixCollection.changes()` — which
/// run the blocking poll on their own isolate.
library;

import 'dart:convert' show utf8;
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'native/watch_bindings.dart';
import 'phoenixdb_base.dart';

/// What happened to a key or document.
enum ChangeKind {
  /// Written: inserted or overwritten.
  put(1),

  /// Deleted.
  delete(2),

  /// Everything changed at once — the database was restored from a backup.
  /// Re-read whatever you are displaying; no per-key changes follow for it.
  reset(3);

  const ChangeKind(this.code);

  /// Wire value used by the native layer.
  final int code;

  static ChangeKind _fromCode(int code) => switch (code) {
    1 => ChangeKind.put,
    2 => ChangeKind.delete,
    _ => ChangeKind.reset,
  };
}

/// One committed change to a key.
class Change {
  /// What happened.
  final ChangeKind kind;

  /// The key, empty for [ChangeKind.reset].
  final Uint8List key;

  /// The value written, when the subscription asked for values and this is a
  /// [ChangeKind.put].
  final Uint8List? value;

  /// Commit timestamp; changes arrive in this order.
  final int commitTs;

  /// Creates a change.
  const Change({
    required this.kind,
    required this.key,
    required this.commitTs,
    this.value,
  });

  /// The key decoded as UTF-8.
  String get keyString => utf8.decode(key, allowMalformed: true);

  /// The value decoded as UTF-8, when one was delivered.
  String? get valueString {
    final v = value;
    return v == null ? null : utf8.decode(v, allowMalformed: true);
  }

  @override
  String toString() =>
      'Change(${kind.name}, ${key.length} byte key'
      '${value == null ? '' : ', ${value!.length} byte value'})';
}

/// One committed change to a document in a collection.
class CollectionChange {
  /// The document's id, empty for [ChangeKind.reset].
  final String id;

  /// Whether the document was written, removed, or the whole collection was
  /// replaced.
  final ChangeKind kind;

  /// Commit timestamp; changes arrive in this order.
  final int commitTs;

  /// Creates a change.
  const CollectionChange({
    required this.id,
    required this.kind,
    required this.commitTs,
  });

  @override
  String toString() => 'CollectionChange(${kind.name} $id)';
}

/// Decodes the native batch format: for each change
/// `[u8 kind][u64 commitTs][u32 keyLen][key][u32 valueLen][value]`, where a
/// `valueLen` of `0xFFFFFFFF` means "no value" (distinct from an empty one).
List<Change> decodeChanges(Uint8List bytes) {
  final out = <Change>[];
  final data = ByteData.sublistView(bytes);
  var at = 0;
  while (at < bytes.length) {
    final kind = ChangeKind._fromCode(data.getUint8(at));
    at += 1;
    final commitTs = data.getUint64(at, Endian.little);
    at += 8;
    final keyLen = data.getUint32(at, Endian.little);
    at += 4;
    final key = Uint8List.fromList(bytes.sublist(at, at + keyLen));
    at += keyLen;
    final valueLen = data.getUint32(at, Endian.little);
    at += 4;
    Uint8List? value;
    if (valueLen != 0xFFFFFFFF) {
      value = Uint8List.fromList(bytes.sublist(at, at + valueLen));
      at += valueLen;
    }
    out.add(Change(kind: kind, key: key, commitTs: commitTs, value: value));
  }
  return out;
}

class _WatcherOwner implements Finalizable {
  final Pointer<PhoenixWatcherNative> pointer;
  _WatcherOwner(this.pointer);
}

/// A subscription to committed changes. Close it when done.
///
/// [poll] blocks until a change arrives or the timeout expires. A watcher may
/// be polled from one isolate while [wake] and [close] are called from
/// another — that is how the `Stream` APIs shut a subscription down without
/// waiting out the timeout.
class ChangeWatcher implements Finalizable {
  final PhoenixWatchBindings _b;
  final _WatcherOwner _owner;
  final NativeFinalizer _finalizer;
  bool _closed = false;

  /// Changes dropped because the queue was full since the last [poll].
  ///
  /// Non-zero means this consumer fell behind and lost the *oldest* changes:
  /// re-read what you are displaying rather than trusting the stream alone.
  int dropped = 0;

  ChangeWatcher._(this._b, this._owner, this._finalizer) {
    _finalizer.attach(this, _owner.pointer.cast(), detach: this);
  }

  /// Subscribes on `handle`; called by [PhoenixDatabase.watch].
  static ChangeWatcher openOnDatabase(
    PhoenixWatchBindings b,
    Pointer<PhoenixDB> handle, {
    Uint8List? prefix,
    int capacity = 1024,
    bool values = false,
  }) {
    final length = prefix?.length ?? 0;
    final prefixPtr = length == 0 ? nullptr : calloc<Uint8>(length);
    final out = calloc<Pointer<PhoenixWatcherNative>>();
    try {
      if (length > 0) prefixPtr.asTypedList(length).setAll(0, prefix!);
      final status = b.open(
        handle,
        prefixPtr,
        length,
        capacity,
        values ? 1 : 0,
        out,
      );
      _check(b, status, 'watch');
      return ChangeWatcher._(
        b,
        _WatcherOwner(out.value),
        NativeFinalizer(b.closePtr.cast()),
      );
    } finally {
      if (length > 0) calloc.free(prefixPtr);
      calloc.free(out);
    }
  }

  /// Subscribes to a collection's document changes.
  static ChangeWatcher openOnCollection(
    PhoenixWatchBindings b,
    Pointer<Never> handle, {
    int capacity = 1024,
  }) {
    final out = calloc<Pointer<PhoenixWatcherNative>>();
    try {
      _check(b, b.openCollection(handle.cast(), capacity, out), 'watch');
      return ChangeWatcher._(
        b,
        _WatcherOwner(out.value),
        NativeFinalizer(b.closePtr.cast()),
      );
    } finally {
      calloc.free(out);
    }
  }

  static void _check(PhoenixWatchBindings b, int status, String what) {
    if (status == PhoenixStatus.ok) return;
    final ptr = b.base.lastError();
    var detail = 'native call failed';
    if (ptr != nullptr) {
      try {
        detail = ptr.toDartString();
      } finally {
        b.base.stringFree(ptr);
      }
    }
    throw PhoenixException(status, '$what: $detail');
  }

  /// Whether [close] has run.
  bool get isClosed => _closed;

  /// Whether the database being watched has closed, so no more changes can
  /// arrive.
  bool get isFinished {
    if (_closed) return true;
    final out = calloc<Int32>();
    try {
      _check(_b, _b.isClosed(_owner.pointer, out), 'watch.isFinished');
      return out.value != 0;
    } finally {
      calloc.free(out);
    }
  }

  /// Waits up to [timeout] for changes and returns everything buffered.
  ///
  /// Empty means nothing arrived in time, [wake] was called, or the database
  /// closed. Blocks the calling isolate.
  List<Change> poll({Duration timeout = const Duration(seconds: 1)}) {
    if (_closed) {
      throw const PhoenixException(
        PhoenixStatus.invalidArgument,
        'watcher is closed',
      );
    }
    final buffer = calloc<PhoenixBuffer>();
    final droppedOut = calloc<Uint64>();
    try {
      final status = _b.poll(
        _owner.pointer,
        timeout.inMilliseconds < 0 ? 0 : timeout.inMilliseconds,
        buffer,
        droppedOut,
      );
      _check(_b, status, 'watch.poll');
      dropped += droppedOut.value;
      final ptr = buffer.ref.ptr;
      if (ptr == nullptr || buffer.ref.len == 0) return const [];
      try {
        return decodeChanges(
          Uint8List.fromList(ptr.asTypedList(buffer.ref.len)),
        );
      } finally {
        _b.base.bufferFree(buffer);
      }
    } finally {
      calloc.free(buffer);
      calloc.free(droppedOut);
    }
  }

  /// Returns a [poll] blocked on another isolate immediately.
  void wake() {
    if (_closed) return;
    _check(_b, _b.wake(_owner.pointer), 'watch.wake');
  }

  /// Unsubscribes. Idempotent.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer.detach(this);
    _b.close(_owner.pointer);
  }
}
