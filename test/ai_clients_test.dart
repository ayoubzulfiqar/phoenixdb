/// Model clients against a local fake server: request shapes, response and
/// SSE parsing, refusals and retries. No network access or API keys needed.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:phoenixdb/ai.dart';
import 'package:phoenixdb/src/ai/http.dart' show parseSse;
import 'package:test/test.dart';

/// A recorded request.
class Seen {
  final String path;
  final HttpHeaders headers;
  final Map<String, Object?> body;
  Seen(this.path, this.headers, this.body);
}

/// A loopback server that answers each request with the next scripted reply.
class FakeServer {
  final HttpServer _server;
  final List<Future<void> Function(HttpResponse)> _replies = [];
  final List<Seen> seen = [];

  FakeServer._(this._server) {
    _server.listen((request) async {
      final text = await utf8.decoder.bind(request).join();
      seen.add(
        Seen(
          request.uri.path,
          request.headers,
          (jsonDecode(text) as Map).cast<String, Object?>(),
        ),
      );
      final reply = _replies.isEmpty
          ? (HttpResponse r) async => r.statusCode = 500
          : _replies.removeAt(0);
      await reply(request.response);
      await request.response.close();
    });
  }

  static Future<FakeServer> start() async =>
      FakeServer._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));

  Uri get url => Uri.parse('http://127.0.0.1:${_server.port}');

  void json(Object body, {int status = 200, Map<String, String>? headers}) {
    _replies.add((r) async {
      r.statusCode = status;
      headers?.forEach(r.headers.set);
      r.headers.contentType = ContentType.json;
      r.write(jsonEncode(body));
    });
  }

  void sse(List<String> frames) {
    _replies.add((r) async {
      r.headers.contentType = ContentType('text', 'event-stream');
      for (final f in frames) {
        r.write(f);
        await r.flush();
      }
    });
  }

  Future<void> close() => _server.close(force: true);
}

String event(String name, Object data) =>
    'event: $name\ndata: ${jsonEncode(data)}\n\n';

void main() {
  late FakeServer server;

  setUp(() async => server = await FakeServer.start());
  tearDown(() => server.close());

  test('SSE parsing handles comments, multi-line data and CRLF', () async {
    final bytes = Stream.value(
      utf8.encode(
        ': keep-alive\r\n'
        'event: a\r\ndata: one\r\ndata: two\r\n\r\n'
        'data: {"x":1}\n\n'
        'data:tail',
      ),
    );
    final events = await parseSse(bytes).toList();
    expect(events.map((e) => (e.event, e.data)), [
      ('a', 'one\ntwo'),
      ('message', '{"x":1}'),
      ('message', 'tail'),
    ]);
  });

  group('AnthropicChatModel', () {
    AnthropicChatModel claude({String model = 'claude-opus-5'}) =>
        AnthropicChatModel(
          apiKey: 'sk-test',
          model: model,
          baseUrl: server.url,
          maxRetries: 1,
          adaptiveThinking: model != 'claude-haiku-4-5',
        );

    test('sends a current Messages API request and reads the text', () async {
      server.json({
        'model': 'claude-opus-5',
        'stop_reason': 'end_turn',
        'content': [
          {'type': 'thinking', 'thinking': ''},
          {'type': 'text', 'text': 'Hello'},
          {'type': 'text', 'text': ', world'},
        ],
        'usage': {
          'input_tokens': 12,
          'output_tokens': 3,
          'cache_read_input_tokens': 8,
        },
      });
      final c = claude();
      final r = await c.complete(const [
        ChatMessage.system('Be brief.'),
        ChatMessage.user('Hi'),
        ChatMessage.assistant('Hello!'),
        ChatMessage.user('Again'),
      ], system: 'You are helpful.');
      c.close();
      expect(r.text, 'Hello, world');
      expect(r.stopReason, 'end_turn');
      expect(r.usage!.cachedInputTokens, 8);

      final req = server.seen.single;
      expect(req.path, '/v1/messages');
      expect(req.headers.value('x-api-key'), 'sk-test');
      expect(req.headers.value('anthropic-version'), '2023-06-01');
      expect(
        req.headers.value('anthropic-beta'),
        'server-side-fallback-2026-07-01',
      );
      expect(req.body['model'], 'claude-opus-5');
      expect(req.body['max_tokens'], 16000);
      expect(req.body['system'], 'You are helpful.\n\nBe brief.');
      expect(req.body['thinking'], {'type': 'adaptive'});
      expect(req.body['fallbacks'], 'default');
      expect(req.body['cache_control'], {'type': 'ephemeral'});
      expect(req.body['messages'], [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello!'},
        {'role': 'user', 'content': 'Again'},
      ]);
      expect(req.body.containsKey('stream'), isFalse);
    });

    test('other models get no fallbacks or adaptive thinking', () async {
      server.json({
        'stop_reason': 'end_turn',
        'content': [
          {'type': 'text', 'text': 'ok'},
        ],
      });
      final c = claude(model: 'claude-haiku-4-5');
      await c.complete(const [ChatMessage.user('Hi')]);
      c.close();
      final req = server.seen.single;
      expect(req.headers.value('anthropic-beta'), isNull);
      expect(req.body.containsKey('fallbacks'), isFalse);
      expect(req.body.containsKey('thinking'), isFalse);
    });

    test('a refusal is an exception, checked before content', () async {
      server.json({
        'stop_reason': 'refusal',
        'stop_details': {'type': 'refusal', 'category': 'cyber'},
        'content': [],
      });
      final c = claude();
      await expectLater(
        c.complete(const [ChatMessage.user('...')]),
        throwsA(
          isA<LlmRefusalException>().having(
            (e) => e.category,
            'category',
            'cyber',
          ),
        ),
      );
      c.close();
    });

    test('retries overload, but not a bad request', () async {
      server.json(
        {
          'type': 'error',
          'error': {'type': 'overloaded_error', 'message': 'busy'},
        },
        status: 529,
        headers: {'retry-after': '0'},
      );
      server.json({
        'stop_reason': 'end_turn',
        'content': [
          {'type': 'text', 'text': 'recovered'},
        ],
      });
      server.json({
        'type': 'error',
        'error': {'type': 'invalid_request_error', 'message': 'bad'},
      }, status: 400);
      final c = claude();
      expect(
        (await c.complete(const [ChatMessage.user('x')])).text,
        'recovered',
      );
      await expectLater(
        c.complete(const [ChatMessage.user('x')]),
        throwsA(
          isA<LlmHttpException>()
              .having((e) => e.statusCode, 'status', 400)
              .having((e) => e.errorType, 'type', 'invalid_request_error')
              .having((e) => e.isRetryable, 'retryable', isFalse),
        ),
      );
      c.close();
      expect(server.seen, hasLength(3));
    });

    test('streams text deltas and ignores thinking and pings', () async {
      server.sse([
        event('message_start', {
          'type': 'message_start',
          'message': {'model': 'claude-opus-5'},
        }),
        event('ping', {'type': 'ping'}),
        event('content_block_delta', {
          'type': 'content_block_delta',
          'index': 0,
          'delta': {'type': 'thinking_delta', 'thinking': 'hmm'},
        }),
        event('content_block_delta', {
          'type': 'content_block_delta',
          'index': 1,
          'delta': {'type': 'text_delta', 'text': 'Hel'},
        }),
        event('content_block_delta', {
          'type': 'content_block_delta',
          'index': 1,
          'delta': {'type': 'text_delta', 'text': 'lo'},
        }),
        event('message_delta', {
          'type': 'message_delta',
          'delta': {'stop_reason': 'end_turn'},
        }),
        event('message_stop', {'type': 'message_stop'}),
      ]);
      final c = claude();
      final parts = await c.stream(const [ChatMessage.user('Hi')]).toList();
      c.close();
      expect(parts, ['Hel', 'lo']);
      expect(server.seen.single.body['stream'], isTrue);
      expect(server.seen.single.body['max_tokens'], 64000);
    });

    test('a streamed refusal throws after the partial text', () async {
      server.sse([
        event('content_block_delta', {
          'type': 'content_block_delta',
          'index': 0,
          'delta': {'type': 'text_delta', 'text': 'Par'},
        }),
        event('message_delta', {
          'type': 'message_delta',
          'delta': {'stop_reason': 'refusal'},
        }),
      ]);
      final c = claude();
      final seen = <String>[];
      await expectLater(
        c.stream(const [ChatMessage.user('Hi')]).forEach(seen.add),
        throwsA(isA<LlmRefusalException>()),
      );
      c.close();
      expect(seen, ['Par']);
    });

    test('a stream error event surfaces as an HTTP-style error', () async {
      server.sse([
        event('error', {
          'type': 'error',
          'error': {'type': 'overloaded_error', 'message': 'Overloaded'},
        }),
      ]);
      final c = claude();
      await expectLater(
        c.stream(const [ChatMessage.user('Hi')]).toList(),
        throwsA(
          isA<LlmHttpException>().having((e) => e.statusCode, 'status', 529),
        ),
      );
      c.close();
    });
  });

  group('OpenAI-compatible', () {
    test('chat completes and streams', () async {
      server.json({
        'model': 'llama3',
        'choices': [
          {
            'message': {'role': 'assistant', 'content': 'Hi there'},
            'finish_reason': 'stop',
          },
        ],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 2},
      });
      server.sse([
        'data: ${jsonEncode({
          'choices': [
            {
              'delta': {'content': 'A'},
            },
          ],
        })}\n\n',
        'data: ${jsonEncode({
          'choices': [
            {'delta': <String, Object?>{}},
          ],
        })}\n\n',
        'data: ${jsonEncode({
          'choices': [
            {
              'delta': {'content': 'B'},
            },
          ],
        })}\n\n',
        'data: [DONE]\n\n',
      ]);
      final m = OpenAICompatibleChatModel(
        baseUrl: server.url.replace(path: '/v1'),
        model: 'llama3',
        apiKey: 'key',
        maxTokens: 100,
        extraBody: {'temperature': 0.2},
      );
      final r = await m.complete(const [ChatMessage.user('Hi')], system: 'sys');
      expect(r.text, 'Hi there');
      expect(r.usage!.inputTokens, 5);
      expect(await m.stream(const [ChatMessage.user('Hi')]).join(), 'AB');
      m.close();

      final req = server.seen.first;
      expect(req.path, '/v1/chat/completions');
      expect(req.headers.value('authorization'), 'Bearer key');
      expect(req.body['max_tokens'], 100);
      expect(req.body['temperature'], 0.2);
      expect(req.body['messages'], [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'Hi'},
      ]);
    });

    test('a refusal field is an exception', () async {
      server.json({
        'choices': [
          {
            'message': {'content': null, 'refusal': 'I cannot help'},
            'finish_reason': 'stop',
          },
        ],
      });
      final m = OpenAICompatibleChatModel(baseUrl: server.url, model: 'x');
      await expectLater(
        m.complete(const [ChatMessage.user('?')]),
        throwsA(isA<LlmRefusalException>()),
      );
      m.close();
    });

    test('embeddings batch, reorder by index and check dimensions', () async {
      List<Map<String, Object?>> rows(int n, int dim, {int offset = 0}) => [
        for (var i = n - 1; i >= 0; i--)
          {'index': i, 'embedding': List.filled(dim, (offset + i).toDouble())},
      ];
      server.json({'data': rows(2, 3)});
      server.json({'data': rows(1, 3, offset: 2)});
      server.json({'data': rows(1, 4)});
      final e = OpenAICompatibleEmbedder(
        baseUrl: server.url.replace(path: '/v1'),
        model: 'embed',
        dimensions: 3,
        batchSize: 2,
        purposeField: 'input_type',
      );
      final vs = await e.embed(['a', 'b', 'c'], purpose: EmbedPurpose.query);
      expect([for (final v in vs) v.first], [0, 1, 2]);
      expect(server.seen[0].body['input'], ['a', 'b']);
      expect(server.seen[0].body['input_type'], 'query');
      expect(server.seen[1].body['input'], ['c']);
      await expectLater(e.embed(['d']), throwsA(isA<LlmException>()));
      e.close();
    });
  });
}
