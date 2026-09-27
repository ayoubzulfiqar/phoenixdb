/// Tool calling and the agent loop: the wire shapes both providers expect,
/// and what the loop does when a tool fails, is unknown, or never stops.
library;

import 'dart:convert';
import 'dart:io';

import 'package:phoenixdb/ai.dart';
import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

/// A scripted tool-calling model: answers from a queue of responses and
/// records what it was sent.
class ScriptedModel implements ToolCallingModel {
  final List<ChatResponse> replies;
  final requests = <List<ChatMessage>>[];
  final toolsSeen = <List<String>>[];

  ScriptedModel(this.replies);

  @override
  String get model => 'scripted';

  @override
  Future<ChatResponse> completeWithTools(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
    List<Tool> tools = const [],
  }) async {
    requests.add([...messages]);
    toolsSeen.add([for (final t in tools) t.name]);
    return replies.removeAt(0);
  }

  @override
  Future<ChatResponse> complete(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) => completeWithTools(messages, system: system, maxTokens: maxTokens);

  @override
  Stream<String> stream(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async* {
    yield (await complete(messages)).text;
  }

  @override
  void close() {}
}

Tool echoTool({String name = 'echo'}) => Tool(
  name: name,
  description: 'Echoes its input back.',
  schema: {
    'type': 'object',
    'properties': {
      'text': {'type': 'string'},
    },
    'required': ['text'],
  },
  run: (input) async => 'echo: ${input['text']}',
);

ChatResponse callOf(
  String name,
  Map<String, Object?> input, {
  String id = 't1',
}) => ChatResponse(
  '',
  toolCalls: [ToolCall(id: id, name: name, input: input)],
  usage: const ChatUsage(inputTokens: 10, outputTokens: 5),
);

void main() {
  group('Agent', () {
    test('runs a tool and feeds the result back', () async {
      final model = ScriptedModel([
        callOf('echo', {'text': 'hello'}),
        const ChatResponse(
          'The tool said hello.',
          usage: ChatUsage(inputTokens: 20, outputTokens: 7),
        ),
      ]);
      final steps = <AgentStep>[];
      final agent = Agent(
        model: model,
        tools: [echoTool()],
        system: 'Be brief.',
        onStep: steps.add,
      );

      final run = await agent.run('say hello');
      expect(run.answer, 'The tool said hello.');
      expect(run.exhausted, isFalse);
      expect(run.steps.single.result, 'echo: hello');
      expect(steps.single.call.name, 'echo');
      expect(run.usage.inputTokens, 30, reason: 'summed over round trips');
      expect(run.usage.outputTokens, 12);

      // The second request replays the call and its result, so the model can
      // see what happened.
      final second = model.requests[1];
      expect(second.first.content, 'say hello');
      expect(second[1].toolCalls.single.name, 'echo');
      expect(second[2].toolResults.single.content, 'echo: hello');
      expect(second[2].toolResults.single.isError, isFalse);
      expect(model.toolsSeen.first, ['echo']);

      // The transcript can be continued.
      expect(run.transcript.last.content, 'The tool said hello.');
    });

    test('a failing tool is reported to the model, not thrown', () async {
      final model = ScriptedModel([
        callOf('boom', {}),
        const ChatResponse('I could not do that.'),
      ]);
      final agent = Agent(
        model: model,
        tools: [
          Tool(
            name: 'boom',
            description: 'Always fails.',
            schema: const {'type': 'object', 'properties': {}},
            run: (_) async => throw StateError('no network'),
          ),
        ],
      );
      final run = await agent.run('try it');
      expect(run.answer, 'I could not do that.');
      expect(run.steps.single.isError, isTrue);
      expect(run.steps.single.result, contains('no network'));
      expect(
        model.requests[1].last.toolResults.single.isError,
        isTrue,
        reason: 'the model is told it failed',
      );
    });

    test('an unknown tool name is answered with the real names', () async {
      final model = ScriptedModel([
        callOf('nope', {}),
        const ChatResponse('Using echo instead.'),
      ]);
      final agent = Agent(model: model, tools: [echoTool()]);
      final run = await agent.run('go');
      expect(run.steps.single.isError, isTrue);
      expect(run.steps.single.result, contains('echo'));
      expect(run.answer, 'Using echo instead.');
    });

    test('several calls in one round all run, in order', () async {
      final model = ScriptedModel([
        ChatResponse(
          '',
          toolCalls: const [
            ToolCall(id: 'a', name: 'echo', input: {'text': 'one'}),
            ToolCall(id: 'b', name: 'echo', input: {'text': 'two'}),
          ],
        ),
        const ChatResponse('done'),
      ]);
      final agent = Agent(model: model, tools: [echoTool()]);
      final run = await agent.run('both');
      expect(run.steps.map((s) => s.result), ['echo: one', 'echo: two']);
      expect(model.requests[1].last.toolResults.map((r) => r.id), [
        'a',
        'b',
      ], reason: 'results are returned together, keyed by call id');
    });

    test('a model that never stops is bounded by maxSteps', () async {
      final model = ScriptedModel([
        for (var i = 0; i < 5; i++)
          callOf('echo', {'text': 'again'}, id: 't$i'),
      ]);
      final agent = Agent(model: model, tools: [echoTool()], maxSteps: 3);
      final run = await agent.run('loop');
      expect(run.exhausted, isTrue);
      expect(run.answer, isEmpty);
      expect(run.steps, hasLength(3));
    });

    test('configuration is validated', () {
      final model = ScriptedModel([]);
      expect(
        () => Agent(model: model, tools: [echoTool()], maxSteps: 0),
        throwsArgumentError,
      );
      expect(
        () => Agent(model: model, tools: [echoTool(), echoTool()]),
        throwsArgumentError,
        reason: 'two tools with one name would be ambiguous',
      );
    });
  });

  group('tool wire shapes', () {
    late HttpServer server;
    late List<Map<String, Object?>> bodies;
    late List<Object> replies;

    setUp(() async {
      bodies = [];
      replies = [];
      server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      server.listen((request) async {
        final text = await utf8.decoder.bind(request).join();
        bodies.add((jsonDecode(text) as Map).cast<String, Object?>());
        request.response.headers.contentType = ContentType.json;
        request.response.write(jsonEncode(replies.removeAt(0)));
        await request.response.close();
      });
    });
    tearDown(() => server.close(force: true));

    Uri url() => Uri.parse('http://127.0.0.1:${server.port}');

    test('Claude sends input_schema and parses tool_use blocks', () async {
      replies.add({
        'model': 'claude-opus-5',
        'stop_reason': 'tool_use',
        'content': [
          {'type': 'text', 'text': 'Looking that up.'},
          {
            'type': 'tool_use',
            'id': 'toolu_1',
            'name': 'echo',
            'input': {'text': 'hi'},
          },
        ],
        'usage': {'input_tokens': 5, 'output_tokens': 2},
      });
      final c = AnthropicChatModel(
        apiKey: 'sk-test',
        baseUrl: url(),
        maxRetries: 0,
      );
      final response = await c.completeWithTools(
        const [ChatMessage.user('echo hi')],
        tools: [echoTool()],
      );
      expect(response.text, 'Looking that up.');
      expect(response.toolCalls.single.name, 'echo');
      expect(response.toolCalls.single.input, {'text': 'hi'});

      final tools = bodies.single['tools']! as List;
      expect((tools.single as Map)['input_schema'], isA<Map>());

      // Replaying the call and its result becomes content blocks.
      replies.add({
        'stop_reason': 'end_turn',
        'content': [
          {'type': 'text', 'text': 'It said hi.'},
        ],
      });
      await c.completeWithTools(
        [
          const ChatMessage.user('echo hi'),
          ChatMessage(ChatRole.assistant, '', toolCalls: response.toolCalls),
          const ChatMessage.toolResults([
            ToolResult(id: 'toolu_1', content: 'echo: hi'),
          ]),
        ],
        tools: [echoTool()],
      );
      final messages = bodies[1]['messages']! as List;
      final assistant = (messages[1] as Map)['content']! as List;
      expect((assistant.single as Map)['type'], 'tool_use');
      final user = (messages[2] as Map)['content']! as List;
      expect((user.single as Map)['type'], 'tool_result');
      expect((user.single as Map)['tool_use_id'], 'toolu_1');
      c.close();
    });

    test('OpenAI sends functions and parses string arguments', () async {
      replies.add({
        'model': 'gpt-test',
        'choices': [
          {
            'message': {
              'role': 'assistant',
              'content': null,
              'tool_calls': [
                {
                  'id': 'call_1',
                  'type': 'function',
                  'function': {
                    'name': 'echo',
                    // Arguments are a JSON *string* in this API.
                    'arguments': '{"text": "hi"}',
                  },
                },
              ],
            },
            'finish_reason': 'tool_calls',
          },
        ],
      });
      final m = OpenAICompatibleChatModel(baseUrl: url(), model: 'gpt-test');
      final response = await m.completeWithTools(
        const [ChatMessage.user('echo hi')],
        tools: [echoTool()],
      );
      expect(response.toolCalls.single.input, {'text': 'hi'});
      expect(
        ((bodies.single['tools']! as List).single as Map)['type'],
        'function',
      );

      // Results go back as their own `role: tool` messages.
      replies.add({
        'choices': [
          {
            'message': {'content': 'It said hi.'},
            'finish_reason': 'stop',
          },
        ],
      });
      await m.completeWithTools(
        [
          const ChatMessage.user('echo hi'),
          ChatMessage(ChatRole.assistant, '', toolCalls: response.toolCalls),
          const ChatMessage.toolResults([
            ToolResult(id: 'call_1', content: 'echo: hi'),
          ]),
        ],
        tools: [echoTool()],
      );
      final messages = bodies[1]['messages']! as List;
      expect((messages[1] as Map)['tool_calls'], isA<List>());
      expect((messages[2] as Map)['role'], 'tool');
      expect((messages[2] as Map)['tool_call_id'], 'call_1');
      m.close();
    });

    test('malformed tool arguments decode to an empty input', () {
      expect(decodeToolInput('{"a": 1}'), {'a': 1});
      expect(decodeToolInput({'a': 1}), {'a': 1});
      expect(decodeToolInput('{"a": trunc'), isEmpty, reason: 'truncated');
      expect(decodeToolInput(''), isEmpty);
      expect(decodeToolInput(null), isEmpty);
      expect(decodeToolInput(42), isEmpty);
    });
  });

  group('knowledgeBaseTool', () {
    test('lets a model search the local collection', () async {
      final dir = Directory.systemTemp.createTempSync('phoenix_tools_');
      final kb = PhoenixCollection.open(
        '${dir.path}/kb',
        dimensions: 256,
        sync: false,
      );
      try {
        final rag = RagPipeline(
          store: kb.asStore(),
          embedder: HashingEmbedder(),
          chat: ScriptedModel([]),
          k: 2,
        );
        await rag.ingest(
          const RagDocument(
            'handbook',
            'Employees get 25 vacation days per year.',
          ),
        );

        final tool = knowledgeBaseTool(rag);
        expect(tool.schema['required'], ['query']);
        final found = await tool.run({'query': 'vacation days'});
        expect(found, contains('25 vacation days'));
        expect(found, contains('handbook'), reason: 'cites its source');
        expect(
          await tool.run({'query': 'nothing like this at all'}),
          isNotEmpty,
        );
        await expectLater(tool.run({}), throwsA(isA<LlmException>()));

        // End to end: the model asks the tool, then answers from it.
        final model = ScriptedModel([
          callOf('search_knowledge_base', {'query': 'vacation'}),
          const ChatResponse('25 days [1].'),
        ]);
        final agent = Agent(model: model, tools: [knowledgeBaseTool(rag)]);
        final run = await agent.run('How many vacation days?');
        expect(run.answer, '25 days [1].');
        expect(run.steps.single.result, contains('vacation days'));
      } finally {
        kb.close();
        dir.deleteSync(recursive: true);
      }
    });
  });
}
