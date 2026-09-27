/// Provider-neutral chat model interface.
library;

/// Who wrote a message.
enum ChatRole {
  /// Operator instructions. Providers with a dedicated system field receive
  /// every system message there, joined in order.
  system,

  /// The end user.
  user,

  /// The model.
  assistant,
}

/// One conversation turn.
class ChatMessage {
  /// Author.
  final ChatRole role;

  /// Text content.
  final String content;

  /// Tools the model asked to run, on an assistant turn.
  final List<ToolCall> toolCalls;

  /// Results being handed back, on a user turn.
  final List<ToolResult> toolResults;

  /// Creates a message.
  const ChatMessage(
    this.role,
    this.content, {
    this.toolCalls = const [],
    this.toolResults = const [],
  });

  /// A user turn carrying tool results; see [Agent].
  const ChatMessage.toolResults(this.toolResults)
    : role = ChatRole.user,
      content = '',
      toolCalls = const [];

  /// A system message.
  const ChatMessage.system(this.content)
    : role = ChatRole.system,
      toolCalls = const [],
      toolResults = const [];

  /// A user message.
  const ChatMessage.user(this.content)
    : role = ChatRole.user,
      toolCalls = const [],
      toolResults = const [];

  /// An assistant message.
  const ChatMessage.assistant(this.content)
    : role = ChatRole.assistant,
      toolCalls = const [],
      toolResults = const [];

  @override
  bool operator ==(Object other) =>
      other is ChatMessage && other.role == role && other.content == content;

  @override
  int get hashCode => Object.hash(role, content);

  @override
  String toString() => '${role.name}: $content';
}

/// A model's request to run a tool.
class ToolCall {
  /// Provider-assigned id, echoed back with the result.
  final String id;

  /// Which tool.
  final String name;

  /// Arguments, decoded from the provider's JSON.
  final Map<String, Object?> input;

  /// Creates a call.
  const ToolCall({required this.id, required this.name, required this.input});

  @override
  String toString() => 'ToolCall($name, $input)';
}

/// What a tool returned, on its way back to the model.
class ToolResult {
  /// The [ToolCall.id] this answers.
  final String id;

  /// Text the model will read.
  final String content;

  /// Whether the tool failed. The model is told, so it can adapt.
  final bool isError;

  /// Creates a result.
  const ToolResult({
    required this.id,
    required this.content,
    this.isError = false,
  });
}

/// Token accounting for one response.
class ChatUsage {
  /// Prompt tokens billed.
  final int inputTokens;

  /// Completion tokens billed.
  final int outputTokens;

  /// Prompt tokens served from the provider's cache, when reported.
  final int cachedInputTokens;

  /// Creates a usage record.
  const ChatUsage({
    required this.inputTokens,
    required this.outputTokens,
    this.cachedInputTokens = 0,
  });

  @override
  String toString() =>
      'ChatUsage(in: $inputTokens, out: $outputTokens, '
      'cached: $cachedInputTokens)';
}

/// A complete model response.
class ChatResponse {
  /// The generated text.
  final String text;

  /// Why generation stopped (`end_turn`, `max_tokens`, `stop`, `length`, …),
  /// in the provider's vocabulary.
  final String? stopReason;

  /// The model that produced the response (it can differ from the one
  /// requested, e.g. after a server-side fallback).
  final String? model;

  /// Token usage, when reported.
  final ChatUsage? usage;

  /// Tools the model asked to run. Empty for a plain answer.
  final List<ToolCall> toolCalls;

  /// Creates a response.
  const ChatResponse(
    this.text, {
    this.stopReason,
    this.model,
    this.usage,
    this.toolCalls = const [],
  });

  /// Whether the output hit the token limit and is truncated.
  bool get truncated => stopReason == 'max_tokens' || stopReason == 'length';

  @override
  String toString() => 'ChatResponse($stopReason, ${text.length} chars)';
}

/// A chat-completion model.
abstract interface class ChatModel {
  /// Model identifier sent to the provider.
  String get model;

  /// Generates a complete response to [messages].
  ///
  /// [system] is prepended to any system messages in [messages].
  /// [maxTokens] overrides the client's default output limit.
  Future<ChatResponse> complete(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  });

  /// Streams the response to [messages] as text deltas.
  Stream<String> stream(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  });

  /// Releases network resources.
  void close();
}

/// Splits system text out of [messages]: `(system, rest)`.
(String?, List<ChatMessage>) splitSystem(
  List<ChatMessage> messages,
  String? system,
) {
  final parts = <String>[?system];
  final rest = <ChatMessage>[];
  for (final m in messages) {
    if (m.role == ChatRole.system) {
      parts.add(m.content);
    } else {
      rest.add(m);
    }
  }
  final joined = parts.where((p) => p.trim().isNotEmpty).join('\n\n');
  return (joined.isEmpty ? null : joined, rest);
}
