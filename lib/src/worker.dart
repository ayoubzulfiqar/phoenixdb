/// Supervised request/response worker isolates, shared by the async clients.
///
/// One long-lived isolate owns a native handle and serves requests in order.
/// The supervision is what the async APIs rely on:
///
/// * a failed open is reported, never a hang (the isolate's exit reaches the
///   boot port too);
/// * if the worker dies later — a native abort, an uncaught error — every
///   in-flight and later call fails with a [PhoenixException] instead of
///   waiting forever;
/// * the final request (close) stops the worker even when it throws, so the
///   isolate can never outlive its client.
library;

import 'dart:async';
import 'dart:isolate';

import 'phoenixdb_base.dart';

/// Sent by a worker that could not open its resource.
class WorkerBootFailure {
  /// What went wrong.
  final String message;

  /// Native status code, or -1.
  final int status;

  /// Creates a boot failure.
  const WorkerBootFailure(this.message, this.status);
}

/// One request on the wire.
class _Call {
  final int id;
  final Object? request;
  final SendPort reply;

  const _Call(this.id, this.request, this.reply);
}

/// One response on the wire.
class _Reply {
  final int id;
  final Object? result;
  final String? error;
  final int? status;

  const _Reply(this.id, {this.result, this.error, this.status});
}

/// Worker side: opens the resource with [open], then answers every request
/// with [handle] until a request for which [isFinal] is true has been served.
void serveWorker<T>(
  SendPort ready,
  T Function() open,
  Object? Function(T resource, Object? request) handle,
  bool Function(Object? request) isFinal,
) {
  final commands = ReceivePort();
  final T resource;
  try {
    resource = open();
  } on PhoenixException catch (e) {
    ready.send(WorkerBootFailure(e.message, e.status));
    commands.close();
    return;
  } catch (e) {
    ready.send(WorkerBootFailure(e.toString(), -1));
    commands.close();
    return;
  }
  ready.send(commands.sendPort);

  commands.listen((message) {
    if (message is! _Call) return;
    try {
      message.reply.send(
        _Reply(message.id, result: handle(resource, message.request)),
      );
    } on PhoenixException catch (e) {
      message.reply.send(
        _Reply(message.id, error: e.message, status: e.status),
      );
    } catch (e) {
      message.reply.send(_Reply(message.id, error: e.toString()));
    } finally {
      // Even when the final request fails the resource is gone, so stop
      // listening: otherwise the isolate outlives the client forever.
      if (isFinal(message.request)) commands.close();
    }
  });
}

/// Client side of a supervised worker.
class WorkerClient {
  final SendPort _commands;
  final ReceivePort _responses = ReceivePort();
  final ReceivePort _exit;
  final ReceivePort _errors;
  final Map<int, Completer<Object?>> _pending = {};
  final String _what;
  int _nextId = 1;
  bool _closed = false;
  PhoenixException? _dead;

  WorkerClient._(this._commands, this._exit, this._errors, this._what) {
    _responses.listen((message) {
      if (message is! _Reply) return;
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
    _errors.listen((message) {
      final detail = message is List && message.isNotEmpty
          ? message.first
          : message;
      _fail(PhoenixException(-1, '$_what worker crashed: $detail'));
    });
    _exit.listen((_) {
      _fail(PhoenixException(-1, '$_what worker isolate exited'));
    });
  }

  /// Spawns [entry] with the boot message built by [boot] (which receives
  /// the port the worker must report readiness on) and waits for it to open.
  static Future<WorkerClient> spawn<B>(
    void Function(B boot) entry,
    B Function(SendPort ready) boot, {
    required String debugName,
    required String what,
  }) async {
    final ready = ReceivePort();
    final exit = ReceivePort();
    final errors = ReceivePort();
    final Isolate isolate;
    try {
      // `onExit` also targets `ready`: a worker that dies before reporting
      // delivers null there instead of leaving this await hanging.
      isolate = await Isolate.spawn(
        entry,
        boot(ready.sendPort),
        debugName: debugName,
        onExit: ready.sendPort,
        onError: errors.sendPort,
      );
    } catch (_) {
      ready.close();
      exit.close();
      errors.close();
      rethrow;
    }
    // Registered before the worker can finish opening, so any later death is
    // reported to the client.
    isolate.addOnExitListener(exit.sendPort);

    final first = await ready.first;
    ready.close();
    if (first is! SendPort) {
      exit.close();
      errors.close();
      final failure = first is WorkerBootFailure
          ? first
          : const WorkerBootFailure('worker exited during open', -1);
      throw PhoenixException(
        failure.status,
        'failed to open $what: ${failure.message}',
      );
    }
    return WorkerClient._(first, exit, errors, what);
  }

  /// Whether the client was closed or the worker died.
  bool get isClosed => _closed || _dead != null;

  /// Sends [request] and completes with the worker's result.
  Future<Object?> call(Object? request) {
    final dead = _dead;
    if (dead != null) return Future.error(dead);
    if (_closed) {
      return Future.error(PhoenixException(-2, '$_what is closed'));
    }
    final id = _nextId++;
    final completer = Completer<Object?>();
    _pending[id] = completer;
    _commands.send(_Call(id, request, _responses.sendPort));
    return completer.future;
  }

  /// Sends the final [request] and releases the client. Idempotent.
  Future<void> close(Object? request) async {
    if (_closed) return;
    if (_dead != null) {
      _closed = true;
      return;
    }
    try {
      await call(request);
    } finally {
      _closed = true;
      final error = PhoenixException(
        -1,
        '$_what closed while a call was in flight',
      );
      for (final completer in _pending.values) {
        if (!completer.isCompleted) completer.completeError(error);
      }
      _pending.clear();
      _shutdownPorts();
    }
  }

  void _fail(PhoenixException error) {
    _dead ??= error;
    for (final completer in _pending.values) {
      if (!completer.isCompleted) completer.completeError(error);
    }
    _pending.clear();
    _shutdownPorts();
  }

  void _shutdownPorts() {
    _responses.close();
    _exit.close();
    _errors.close();
  }
}
