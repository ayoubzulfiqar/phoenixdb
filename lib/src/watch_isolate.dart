/// Streams change notifications from a dedicated isolate.
///
/// `phoenix_watch_poll` blocks, so it cannot share an isolate with anything
/// else — including the request/response worker the async clients use for
/// their normal calls. Each subscription therefore gets its own isolate, which
/// opens its own handle on the same path. Handles opened in one process share
/// the engine, so that watcher sees the very writes the other isolate makes.
///
/// The poll timeout is short and the loop yields between polls, so a cancel
/// message is processed promptly instead of waiting out a long block.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'collection.dart';
import 'native/vector_bindings.dart' show VectorMetric;
import 'phoenixdb_base.dart';
import 'watch.dart';

/// How long one native poll blocks before the loop checks for a cancel.
const _pollTimeout = Duration(milliseconds: 200);

/// What to subscribe to.
class WatchSpec {
  /// Database file, or collection directory when [collection] is set.
  final String path;

  /// Overrides native library discovery.
  final String? libraryPath;

  /// Key prefix; ignored for a collection (which always reports ids).
  final Uint8List? prefix;

  /// Changes buffered before the oldest are dropped.
  final int capacity;

  /// Deliver written values as well as keys.
  final bool values;

  /// Watch a collection's documents rather than raw keys.
  final bool collection;

  /// Creates a specification.
  const WatchSpec({
    required this.path,
    this.libraryPath,
    this.prefix,
    this.capacity = 1024,
    this.values = false,
    this.collection = false,
  });
}

/// Sent by the watcher isolate once it is subscribed.
class _Ready {
  final SendPort control;
  const _Ready(this.control);
}

/// Sent when the isolate could not subscribe, or died trying.
class _Failed {
  final String message;
  final int status;
  const _Failed(this.message, this.status);
}

/// Sent when the watched database closed, so no more changes can arrive.
class _Finished {
  const _Finished();
}

class _Boot {
  final SendPort events;
  final WatchSpec spec;
  const _Boot(this.events, this.spec);
}

void _watchMain(_Boot boot) async {
  final control = ReceivePort();
  final Object resource;
  final ChangeWatcher watcher;
  try {
    if (boot.spec.collection) {
      final c = PhoenixCollection.open(
        boot.spec.path,
        libraryPath: boot.spec.libraryPath,
      );
      resource = c;
      watcher = c.rawWatcher(capacity: boot.spec.capacity);
    } else {
      final db = PhoenixDatabase.open(
        boot.spec.path,
        libraryPath: boot.spec.libraryPath,
      );
      resource = db;
      watcher = db.watch(
        prefix: boot.spec.prefix,
        capacity: boot.spec.capacity,
        values: boot.spec.values,
      );
    }
  } on PhoenixException catch (e) {
    boot.events.send(_Failed(e.message, e.status));
    control.close();
    return;
  } catch (e) {
    boot.events.send(_Failed('$e', -1));
    control.close();
    return;
  }

  var stop = false;
  control.listen((_) => stop = true);
  boot.events.send(_Ready(control.sendPort));

  while (!stop) {
    List<Change> batch;
    try {
      batch = watcher.poll(timeout: _pollTimeout);
    } on PhoenixException catch (e) {
      boot.events.send(_Failed(e.message, e.status));
      break;
    }
    if (batch.isNotEmpty) {
      boot.events.send(
        boot.spec.collection
            ? [
                for (final c in batch)
                  CollectionChange(
                    id: c.keyString,
                    kind: c.kind,
                    commitTs: c.commitTs,
                  ),
              ]
            : batch,
      );
    }
    if (watcher.isFinished) {
      boot.events.send(const _Finished());
      break;
    }
    // Let the control port be serviced between blocking polls.
    await Future<void>.delayed(Duration.zero);
  }

  watcher.close();
  switch (resource) {
    case PhoenixCollection c:
      c.close();
    case PhoenixDatabase db:
      db.close();
  }
  control.close();
}

/// A running subscription: a stream of batches plus the way to stop it.
class WatchSession {
  final StreamController<List<Object>> _controller = StreamController(
    sync: true,
  );
  final ReceivePort _events = ReceivePort();
  SendPort? _control;
  bool _stopped = false;

  WatchSession._();

  /// Batches of changes: `List<Change>`, or `List<CollectionChange>` when the
  /// spec asked for a collection.
  Stream<List<Object>> get batches => _controller.stream;

  /// Spawns the watcher isolate for [spec].
  static WatchSession start(WatchSpec spec) {
    final session = WatchSession._();
    session._spawn(spec);
    return session;
  }

  Future<void> _spawn(WatchSpec spec) async {
    _events.listen((message) {
      switch (message) {
        case _Ready(control: final port):
          _control = port;
          // A stop that arrived before the isolate was ready.
          if (_stopped) {
            port.send(null);
          }
        case List<Object> batch:
          if (!_controller.isClosed) _controller.add(batch);
        case _Failed(message: final m, status: final s):
          if (!_controller.isClosed) {
            _controller.addError(PhoenixException(s, m));
          }
          _finish();
        case _Finished():
          _finish();
        case null:
          // The isolate exited (onExit sends null).
          _finish();
      }
    });
    try {
      await Isolate.spawn(
        _watchMain,
        _Boot(_events.sendPort, spec),
        debugName: 'phoenixdb-watch',
        onExit: _events.sendPort,
        onError: _events.sendPort,
      );
      if (_stopped) await stop();
    } catch (e) {
      if (!_controller.isClosed) {
        _controller.addError(PhoenixException(-1, 'watch isolate failed: $e'));
      }
      _finish();
    }
  }

  void _finish() {
    _stopped = true;
    _events.close();
    if (!_controller.isClosed) _controller.close();
  }

  /// Stops the subscription and releases its isolate. Idempotent.
  Future<void> stop() async {
    if (_stopped) {
      _finish();
      return;
    }
    _stopped = true;
    _control?.send(null);
    // The isolate exits on its own once the poll returns; do not kill it, so
    // it can close its handle and release the engine reference cleanly.
    await Future<void>.delayed(_pollTimeout + const Duration(milliseconds: 50));
    _finish();
  }
}

/// A stream of key/value changes for [spec], ending when the subscription is
/// cancelled or the database closes.
Stream<Change> watchChanges(WatchSpec spec) {
  late WatchSession session;
  late StreamController<Change> controller;
  StreamSubscription<List<Object>>? sub;
  controller = StreamController<Change>(
    onListen: () {
      session = WatchSession.start(spec);
      sub = session.batches.listen(
        (batch) {
          for (final change in batch) {
            controller.add(change as Change);
          }
        },
        onError: controller.addError,
        onDone: controller.close,
      );
    },
    onCancel: () async {
      await session.stop();
      await sub?.cancel();
    },
  );
  return controller.stream;
}

/// A stream of document changes for a collection [spec].
Stream<CollectionChange> watchCollectionChanges(WatchSpec spec) {
  late WatchSession session;
  late StreamController<CollectionChange> controller;
  StreamSubscription<List<Object>>? sub;
  controller = StreamController<CollectionChange>(
    onListen: () {
      session = WatchSession.start(spec);
      sub = session.batches.listen(
        (batch) {
          for (final change in batch) {
            controller.add(change as CollectionChange);
          }
        },
        onError: controller.addError,
        onDone: controller.close,
      );
    },
    onCancel: () async {
      await session.stop();
      await sub?.cancel();
    },
  );
  return controller.stream;
}

/// Kept so the collection watcher can adopt an existing layout without the
/// caller restating it.
const defaultWatchMetric = VectorMetric.cosine;
