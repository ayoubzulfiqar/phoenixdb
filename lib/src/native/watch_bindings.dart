/// Low-level `dart:ffi` bindings to the `phoenix_watch_*` C ABI.
///
/// Mirrors `rust/src/ffi/watch_ffi.rs`. Use [PhoenixDatabase.watch],
/// [AsyncPhoenixDB.changes] or the collection equivalents instead of calling
/// these directly.
///
/// `phoenix_watch_poll` **blocks** for up to its timeout, so it must never run
/// on an isolate that has other work to do: the async clients give each
/// subscription its own isolate.
library;

import 'dart:ffi';

import '../bindings.dart';
import 'collection_bindings.dart';

/// Opaque watcher handle.
final class PhoenixWatcherNative extends Opaque {}

/// Native signature for `phoenix_watch_open`.
typedef WatchOpenNative =
    Int32 Function(
      Pointer<PhoenixDB> handle,
      Pointer<Uint8> prefix,
      Size prefixLen,
      Uint64 capacity,
      Int32 withValues,
      Pointer<Pointer<PhoenixWatcherNative>> outWatcher,
    );

/// Dart signature for `phoenix_watch_open`.
typedef WatchOpenDart =
    int Function(
      Pointer<PhoenixDB> handle,
      Pointer<Uint8> prefix,
      int prefixLen,
      int capacity,
      int withValues,
      Pointer<Pointer<PhoenixWatcherNative>> outWatcher,
    );

/// Native signature for `phoenix_collection_watch_open`.
typedef CollectionWatchOpenNative =
    Int32 Function(
      Pointer<PhoenixCollectionNative> handle,
      Uint64 capacity,
      Pointer<Pointer<PhoenixWatcherNative>> outWatcher,
    );

/// Dart signature for `phoenix_collection_watch_open`.
typedef CollectionWatchOpenDart =
    int Function(
      Pointer<PhoenixCollectionNative> handle,
      int capacity,
      Pointer<Pointer<PhoenixWatcherNative>> outWatcher,
    );

/// Native signature for `phoenix_watch_poll`.
typedef WatchPollNative =
    Int32 Function(
      Pointer<PhoenixWatcherNative> handle,
      Uint64 timeoutMs,
      Pointer<PhoenixBuffer> out,
      Pointer<Uint64> outDropped,
    );

/// Dart signature for `phoenix_watch_poll`.
typedef WatchPollDart =
    int Function(
      Pointer<PhoenixWatcherNative> handle,
      int timeoutMs,
      Pointer<PhoenixBuffer> out,
      Pointer<Uint64> outDropped,
    );

/// Native signature for `phoenix_watch_wake`.
typedef WatchWakeNative = Int32 Function(Pointer<PhoenixWatcherNative> handle);

/// Dart signature for `phoenix_watch_wake`.
typedef WatchWakeDart = int Function(Pointer<PhoenixWatcherNative> handle);

/// Native signature for `phoenix_watch_is_closed`.
typedef WatchIsClosedNative =
    Int32 Function(
      Pointer<PhoenixWatcherNative> handle,
      Pointer<Int32> outClosed,
    );

/// Dart signature for `phoenix_watch_is_closed`.
typedef WatchIsClosedDart =
    int Function(
      Pointer<PhoenixWatcherNative> handle,
      Pointer<Int32> outClosed,
    );

/// Native signature for `phoenix_watch_close`.
typedef WatchCloseNative = Void Function(Pointer<PhoenixWatcherNative> handle);

/// Dart signature for `phoenix_watch_close`.
typedef WatchCloseDart = void Function(Pointer<PhoenixWatcherNative> handle);

/// Resolved `phoenix_watch_*` functions.
class PhoenixWatchBindings {
  /// The key/value bindings sharing this library (error channel, buffer free).
  final PhoenixBindings base;

  /// `phoenix_watch_open`.
  final WatchOpenDart open;

  /// `phoenix_collection_watch_open`.
  final CollectionWatchOpenDart openCollection;

  /// `phoenix_watch_poll`.
  final WatchPollDart poll;

  /// `phoenix_watch_wake`.
  final WatchWakeDart wake;

  /// `phoenix_watch_is_closed`.
  final WatchIsClosedDart isClosed;

  /// `phoenix_watch_close`.
  final WatchCloseDart close;

  /// Pointer to `phoenix_watch_close`, for a [NativeFinalizer].
  final Pointer<NativeFunction<WatchCloseNative>> closePtr;

  PhoenixWatchBindings._(this.base)
    : open = base.library.lookupFunction<WatchOpenNative, WatchOpenDart>(
        'phoenix_watch_open',
      ),
      openCollection = base.library
          .lookupFunction<CollectionWatchOpenNative, CollectionWatchOpenDart>(
            'phoenix_collection_watch_open',
          ),
      poll = base.library.lookupFunction<WatchPollNative, WatchPollDart>(
        'phoenix_watch_poll',
      ),
      wake = base.library.lookupFunction<WatchWakeNative, WatchWakeDart>(
        'phoenix_watch_wake',
      ),
      isClosed = base.library
          .lookupFunction<WatchIsClosedNative, WatchIsClosedDart>(
            'phoenix_watch_is_closed',
          ),
      close = base.library.lookupFunction<WatchCloseNative, WatchCloseDart>(
        'phoenix_watch_close',
      ),
      closePtr = base.library.lookup<NativeFunction<WatchCloseNative>>(
        'phoenix_watch_close',
      );

  /// Wraps already-loaded key/value bindings.
  factory PhoenixWatchBindings.from(PhoenixBindings base) =>
      PhoenixWatchBindings._(base);

  /// Loads the bindings through [PhoenixBindings.load]'s library search and
  /// ABI check.
  factory PhoenixWatchBindings.load({String? path}) =>
      PhoenixWatchBindings._(PhoenixBindings.load(path: path));
}
