/// HTTP plumbing shared by the model clients: JSON requests with retries and
/// Server-Sent Event streams, on `dart:io` alone.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

/// Base class of every model-client failure.
class LlmException implements Exception {
  /// What went wrong.
  final String message;

  /// Creates an exception.
  const LlmException(this.message);

  @override
  String toString() => 'LlmException: $message';
}

/// The provider answered with an HTTP error.
class LlmHttpException extends LlmException {
  /// HTTP status code.
  final int statusCode;

  /// Provider error type, e.g. `rate_limit_error` or `overloaded_error`.
  final String? errorType;

  /// Creates an exception.
  const LlmHttpException(this.statusCode, String message, {this.errorType})
    : super(message);

  /// Whether retrying later can succeed (timeouts, conflicts, rate limits,
  /// overload and server errors).
  bool get isRetryable =>
      statusCode == 408 ||
      statusCode == 409 ||
      statusCode == 429 ||
      statusCode >= 500;

  @override
  String toString() =>
      'LlmHttpException($statusCode${errorType == null ? '' : ' $errorType'})'
      ': $message';
}

/// The model (or a safety classifier) declined the request.
///
/// When streaming, text already delivered before the refusal is partial and
/// should be discarded.
class LlmRefusalException extends LlmException {
  /// Refusal category reported by the provider, if any.
  final String? category;

  /// Creates an exception.
  const LlmRefusalException(super.message, {this.category});

  @override
  String toString() =>
      'LlmRefusalException${category == null ? '' : '($category)'}: $message';
}

/// Appends [path] to [base] without dropping [base]'s last path segment
/// (which `Uri.resolve` would).
Uri joinUrl(Uri base, String path) {
  final root = base.path.endsWith('/')
      ? base
      : base.replace(path: '${base.path}/');
  return root.resolve(path);
}

/// One Server-Sent Event.
class SseEvent {
  /// The `event:` field, or `message` when absent.
  final String event;

  /// The joined `data:` lines.
  final String data;

  /// Creates an event.
  const SseEvent(this.event, this.data);

  @override
  String toString() => 'SseEvent($event, $data)';
}

/// Parses a byte stream of `text/event-stream` into events.
Stream<SseEvent> parseSse(Stream<List<int>> bytes) async* {
  String? event;
  final data = StringBuffer();
  var hasData = false;
  // `bind` rather than `transform`: `transform` checks the transformer
  // against the stream's runtime element type, which rejects a
  // `Stream<Uint8List>`.
  await for (final line in const LineSplitter().bind(
    utf8.decoder.bind(bytes),
  )) {
    if (line.isEmpty) {
      if (hasData) yield SseEvent(event ?? 'message', data.toString());
      event = null;
      data.clear();
      hasData = false;
      continue;
    }
    if (line.startsWith(':')) continue; // comment / keep-alive
    final colon = line.indexOf(':');
    final field = colon < 0 ? line : line.substring(0, colon);
    var value = colon < 0 ? '' : line.substring(colon + 1);
    if (value.startsWith(' ')) value = value.substring(1);
    switch (field) {
      case 'event':
        event = value;
      case 'data':
        if (hasData) data.write('\n');
        data.write(value);
        hasData = true;
    }
  }
  if (hasData) yield SseEvent(event ?? 'message', data.toString());
}

/// JSON-over-HTTP transport with bounded, jittered retries.
class HttpTransport {
  final HttpClient _client;
  final bool _ownsClient;

  /// Retries after the first attempt for retryable failures.
  final int maxRetries;

  /// Deadline for a whole non-streaming response, and the longest silence
  /// tolerated between streamed events.
  final Duration timeout;

  /// First retry delay; later retries double it (plus jitter).
  final Duration retryDelay;

  final Random _random = Random();

  /// Creates a transport over [client] (or a private one).
  HttpTransport({
    HttpClient? client,
    this.maxRetries = 2,
    this.timeout = const Duration(minutes: 10),
    this.retryDelay = const Duration(milliseconds: 500),
  }) : _client = client ?? (HttpClient()..connectionTimeout = _connect),
       _ownsClient = client == null;

  static const _connect = Duration(seconds: 30);

  Duration _backoff(int attempt, String? retryAfter) {
    final seconds = int.tryParse(retryAfter ?? '');
    if (seconds != null && seconds >= 0) {
      return Duration(seconds: min(seconds, 60));
    }
    final base = retryDelay.inMicroseconds * (1 << min(attempt, 6));
    final jitter = _random.nextDouble() * 0.25 * base;
    return Duration(microseconds: min(base + jitter.round(), 8000000));
  }

  Future<HttpClientResponse> _open(
    Uri uri,
    Map<String, String> headers,
    Object body,
  ) async {
    final request = await _client.postUrl(uri);
    headers.forEach(request.headers.set);
    request.headers.contentType = ContentType.json;
    request.add(utf8.encode(jsonEncode(body)));
    return request.close();
  }

  static LlmHttpException _error(int status, String body) {
    try {
      final json = jsonDecode(body);
      if (json is Map) {
        final error = json['error'];
        if (error is Map) {
          return LlmHttpException(
            status,
            '${error['message'] ?? body}',
            errorType: error['type'] as String? ?? error['code']?.toString(),
          );
        }
      }
    } on FormatException {
      // Not JSON: report the raw body.
    }
    return LlmHttpException(status, body.isEmpty ? 'HTTP $status' : body);
  }

  /// Opens a 2xx response, retrying connection failures and retryable
  /// statuses.
  Future<HttpClientResponse> _withRetries(
    Uri uri,
    Map<String, String> headers,
    Object body,
  ) async {
    for (var attempt = 0; ; attempt++) {
      try {
        final response = await _open(uri, headers, body).timeout(timeout);
        if (response.statusCode >= 200 && response.statusCode < 300) {
          return response;
        }
        final text = await response
            .transform(utf8.decoder)
            .join()
            .timeout(timeout);
        final error = _error(response.statusCode, text);
        if (!error.isRetryable || attempt >= maxRetries) throw error;
        await Future<void>.delayed(
          _backoff(attempt, response.headers.value('retry-after')),
        );
      } on LlmException {
        rethrow;
      } on Object catch (e) {
        // SocketException, HttpException, TimeoutException, ...
        if (e is! IOException && e is! TimeoutException) rethrow;
        if (attempt >= maxRetries) {
          throw LlmException('request to ${uri.host} failed: $e');
        }
        await Future<void>.delayed(_backoff(attempt, null));
      }
    }
  }

  /// POSTs [body] as JSON and decodes a JSON object response.
  Future<Map<String, Object?>> postJson(
    Uri uri,
    Map<String, String> headers,
    Object body,
  ) async {
    final response = await _withRetries(uri, headers, body);
    final text = await response.transform(utf8.decoder).join().timeout(timeout);
    final json = jsonDecode(text);
    if (json is! Map) {
      throw LlmException('expected a JSON object from ${uri.host}');
    }
    return json.cast<String, Object?>();
  }

  /// POSTs [body] as JSON and streams the Server-Sent Events of the answer.
  /// Retries happen only before the first byte arrives.
  Stream<SseEvent> postSse(
    Uri uri,
    Map<String, String> headers,
    Object body,
  ) async* {
    final response = await _withRetries(uri, {
      ...headers,
      'accept': 'text/event-stream',
    }, body);
    yield* parseSse(response).timeout(
      timeout,
      onTimeout: (sink) {
        sink.addError(
          LlmException('no data from ${uri.host} for ${timeout.inSeconds}s'),
        );
        sink.close();
      },
    );
  }

  /// Releases the HTTP client (if this transport created it).
  void close() {
    if (_ownsClient) _client.close(force: true);
  }
}
