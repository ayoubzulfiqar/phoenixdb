/// Tool calling and a small agent loop, so a model can use your code and your
/// on-device data instead of only answering from the prompt.
///
/// ```dart
/// final agent = Agent(
///   model: AnthropicChatModel(apiKey: key),
///   tools: [
///     knowledgeBaseTool(rag),                  // search the local collection
///     Tool(
///       name: 'set_reminder',
///       description: 'Save a reminder for the user.',
///       schema: {
///         'type': 'object',
///         'properties': {
///           'text': {'type': 'string'},
///           'at': {'type': 'string', 'description': 'ISO-8601 time'},
///         },
///         'required': ['text', 'at'],
///       },
///       run: (input) async => saveReminder(input),
///     ),
///   ],
/// );
///
/// final run = await agent.run('Remind me about the vacation policy tomorrow');
/// print(run.answer);
/// for (final step in run.steps) {
///   print('${step.call.name}(${step.call.input}) -> ${step.result}');
/// }
/// ```
///
/// The loop is deliberately small and inspectable: every step is recorded, a
/// tool that throws is reported to the model rather than aborting the run, and
/// [Agent.maxSteps] bounds the work so a confused model cannot spin.
library;

import 'dart:convert';

import 'chat.dart';
import 'http.dart' show LlmException;
import 'rag.dart';

/// Something the model can ask to have run.
///
/// [schema] is JSON Schema for the input object — the same shape both the
/// Anthropic and OpenAI-compatible APIs expect.
class Tool {
  /// Name the model calls. Keep it short and specific.
  final String name;

  /// What the tool does, when to use it, and what it returns. This is the
  /// prompt the model reads to decide; vague descriptions are the usual reason
  /// a tool is never called.
  final String description;

  /// JSON Schema of the input object.
  final Map<String, Object?> schema;

  /// Runs the tool. Whatever it returns is given back to the model verbatim,
  /// so return text the model can read. Throwing is fine: the failure is
  /// reported to the model, which can try something else.
  final Future<String> Function(Map<String, Object?> input) run;

  /// Creates a tool.
  const Tool({
    required this.name,
    required this.description,
    required this.schema,
    required this.run,
  });

  /// The Anthropic Messages API form.
  Map<String, Object?> toAnthropicJson() => {
    'name': name,
    'description': description,
    'input_schema': schema,
  };

  /// The OpenAI-compatible form.
  Map<String, Object?> toOpenAIJson() => {
    'type': 'function',
    'function': {
      'name': name,
      'description': description,
      'parameters': schema,
    },
  };
}

/// A model that can be given [Tool]s.
///
/// Separate from [ChatModel] so adding tool support broke no existing
/// implementation: a plain [ChatModel] still only has to answer with text.
abstract interface class ToolCallingModel implements ChatModel {
  /// Like [ChatModel.complete], but the model may answer with
  /// [ChatResponse.toolCalls] instead of (or as well as) text.
  Future<ChatResponse> completeWithTools(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
    List<Tool> tools = const [],
  });
}

/// One tool the agent ran.
class AgentStep {
  /// What the model asked for.
  final ToolCall call;

  /// What the tool returned, or the failure text.
  final String result;

  /// Whether the tool threw.
  final bool isError;

  /// Creates a step.
  const AgentStep({
    required this.call,
    required this.result,
    this.isError = false,
  });

  @override
  String toString() =>
      'AgentStep(${call.name}${isError ? ' failed' : ''}: '
      '${result.length > 60 ? '${result.substring(0, 60)}…' : result})';
}

/// The outcome of [Agent.run].
class AgentRun {
  /// The model's final answer.
  final String answer;

  /// Every tool call, in order.
  final List<AgentStep> steps;

  /// The whole conversation, ready to continue with.
  final List<ChatMessage> transcript;

  /// True when the loop hit [Agent.maxSteps] with the model still calling
  /// tools, so [answer] may be incomplete.
  final bool exhausted;

  /// Token usage, summed over every round trip.
  final ChatUsage usage;

  /// Creates a run.
  const AgentRun({
    required this.answer,
    required this.steps,
    required this.transcript,
    required this.usage,
    this.exhausted = false,
  });

  @override
  String toString() =>
      'AgentRun(${steps.length} tool call(s)'
      '${exhausted ? ', exhausted' : ''}, ${answer.length} chars)';
}

/// Runs a model in a loop, executing the tools it asks for.
class Agent {
  /// The model. Must support tool calling.
  final ToolCallingModel model;

  /// Tools the model may call.
  final List<Tool> tools;

  /// Instructions prepended to every request.
  final String? system;

  /// Most model round trips in one [run]. Each round trip may call several
  /// tools; the bound is what stops a confused model from spinning.
  final int maxSteps;

  /// Output limit per round trip.
  final int? maxTokens;

  /// Called after each tool runs, for logging or a progress indicator.
  final void Function(AgentStep step)? onStep;

  /// Creates an agent.
  Agent({
    required this.model,
    required this.tools,
    this.system,
    this.maxSteps = 8,
    this.maxTokens,
    this.onStep,
  }) {
    if (maxSteps <= 0) {
      throw ArgumentError.value(maxSteps, 'maxSteps', 'must be positive');
    }
    final names = <String>{};
    for (final tool in tools) {
      if (!names.add(tool.name)) {
        throw ArgumentError('two tools are both called "${tool.name}"');
      }
    }
  }

  /// Answers [task], calling tools as the model asks.
  Future<AgentRun> run(String task, {List<ChatMessage> history = const []}) =>
      continueRun([...history, ChatMessage.user(task)]);

  /// Continues an existing [transcript] (e.g. from a previous [AgentRun]).
  Future<AgentRun> continueRun(List<ChatMessage> transcript) async {
    final messages = [...transcript];
    final steps = <AgentStep>[];
    var input = 0;
    var output = 0;
    var cached = 0;
    final byName = {for (final t in tools) t.name: t};

    for (var round = 0; round < maxSteps; round++) {
      final response = await model.completeWithTools(
        messages,
        system: system,
        maxTokens: maxTokens,
        tools: tools,
      );
      input += response.usage?.inputTokens ?? 0;
      output += response.usage?.outputTokens ?? 0;
      cached += response.usage?.cachedInputTokens ?? 0;

      if (response.toolCalls.isEmpty) {
        messages.add(ChatMessage.assistant(response.text));
        return AgentRun(
          answer: response.text,
          steps: steps,
          transcript: messages,
          usage: ChatUsage(
            inputTokens: input,
            outputTokens: output,
            cachedInputTokens: cached,
          ),
        );
      }

      messages.add(
        ChatMessage(
          ChatRole.assistant,
          response.text,
          toolCalls: response.toolCalls,
        ),
      );
      final results = <ToolResult>[];
      for (final call in response.toolCalls) {
        final tool = byName[call.name];
        AgentStep step;
        if (tool == null) {
          // Tell the model rather than failing the run: it can pick a real
          // tool on the next round.
          step = AgentStep(
            call: call,
            result:
                'No tool named "${call.name}". Available: '
                '${tools.map((t) => t.name).join(', ')}.',
            isError: true,
          );
        } else {
          try {
            step = AgentStep(call: call, result: await tool.run(call.input));
          } catch (e) {
            step = AgentStep(call: call, result: '$e', isError: true);
          }
        }
        steps.add(step);
        onStep?.call(step);
        results.add(
          ToolResult(id: call.id, content: step.result, isError: step.isError),
        );
      }
      messages.add(ChatMessage.toolResults(results));
    }

    return AgentRun(
      answer: '',
      steps: steps,
      transcript: messages,
      usage: ChatUsage(
        inputTokens: input,
        outputTokens: output,
        cachedInputTokens: cached,
      ),
      exhausted: true,
    );
  }
}

/// A tool that searches a [RagPipeline]'s collection, so a model can look
/// things up in on-device data instead of guessing.
///
/// The result is the retrieved chunks with their source ids, which is what a
/// model needs to answer and cite.
Tool knowledgeBaseTool(
  RagPipeline rag, {
  String name = 'search_knowledge_base',
  String description =
      'Search the local knowledge base for passages relevant to a query. '
      'Use it whenever the answer might be in the stored documents. '
      'Returns numbered passages with their document ids.',
  int maxResults = 5,
}) => Tool(
  name: name,
  description: description,
  schema: {
    'type': 'object',
    'properties': {
      'query': {
        'type': 'string',
        'description': 'What to look for, in the user\'s own words.',
      },
      'limit': {
        'type': 'integer',
        'description': 'How many passages to return (default 5).',
      },
    },
    'required': ['query'],
  },
  run: (input) async {
    final query = input['query'];
    if (query is! String || query.trim().isEmpty) {
      throw const LlmException('`query` must be a non-empty string');
    }
    final requested = input['limit'];
    final limit = requested is int && requested > 0
        ? (requested < maxResults ? requested : maxResults)
        : maxResults;
    final sources = await rag.retrieve(query, k: limit);
    if (sources.isEmpty) return 'No matching passages.';
    return sources
        .map(
          (s) =>
              '[${s.number}] (${s.documentId ?? s.id}) '
              '${s.text.replaceAll('\n', ' ')}',
        )
        .join('\n');
  },
);

/// Decodes a provider's tool-call arguments, which arrive as a JSON string in
/// the OpenAI shape and as an object in the Anthropic one.
Map<String, Object?> decodeToolInput(Object? raw) {
  if (raw == null) return const {};
  if (raw is Map) return raw.cast<String, Object?>();
  if (raw is String) {
    if (raw.trim().isEmpty) return const {};
    try {
      final decoded = jsonDecode(raw);
      if (decoded is Map) return decoded.cast<String, Object?>();
    } on FormatException {
      // A model can emit invalid JSON, especially when truncated; report it
      // as an empty input so the tool's own validation explains the problem.
      return const {};
    }
  }
  return const {};
}
