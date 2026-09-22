/// Claude through the Anthropic Messages API.
///
/// Dart has no official Anthropic SDK, so this client speaks the Messages
/// API (`POST /v1/messages`) over HTTP directly, including its
/// Server-Sent-Event streaming format.
library;

import 'dart:convert';
import 'dart:io';

import 'chat.dart';
import 'http.dart';

/// Thinking effort for Claude (`output_config.effort`).
enum ClaudeEffort {
  /// Fastest and cheapest; good for simple or high-volume work.
  low,

  /// A cost-saving step down where quality holds.
  medium,

  /// The API default.
  high,

  /// Deeper reasoning for hard coding and agentic work.
  xhigh,

  /// The deepest reasoning.
  max,
}

/// A [ChatModel] backed by Claude.
///
/// ```dart
/// final claude = AnthropicChatModel(apiKey: Platform.environment['ANTHROPIC_API_KEY']!);
/// final reply = await claude.complete([ChatMessage.user('Hello, Claude')]);
/// print(reply.text);
/// ```
///
/// Defaults follow Anthropic's current guidance: `claude-opus-5`, adaptive
/// thinking, automatic prompt caching and — on models that support it —
/// server-side refusal fallbacks (`fallbacks: "default"`), which re-run a
/// request declined by a safety classifier on Anthropic's recommended
/// fallback model instead of failing it. A request that is still declined
/// throws [LlmRefusalException].
class AnthropicChatModel implements ChatModel {
  @override
  final String model;

  /// API key (`x-api-key`).
  final String apiKey;

  /// Output limit for [complete].
  final int maxTokens;

  /// Output limit for [stream] (streaming is not bound by HTTP timeouts, so
  /// it gets more room).
  final int streamMaxTokens;

  /// Send `thinking: {type: "adaptive"}`. Turn off for models without
  /// adaptive thinking (e.g. Claude Haiku 4.5).
  final bool adaptiveThinking;

  /// Thinking effort; `null` uses the API default (`high`).
  final ClaudeEffort? effort;

  /// Server-side refusal fallbacks; `null` enables them for the models that
  /// support the `"default"` mode (Claude Opus 5 and Claude Fable 5.1).
  final bool? serverSideFallbacks;

  /// Ask the API to cache the longest reusable prompt prefix
  /// (top-level `cache_control`).
  final bool promptCaching;

  /// API root, for proxies and gateways.
  final Uri baseUrl;

  /// Extra request headers (e.g. additional `anthropic-beta` flags).
  final Map<String, String> headers;

  final HttpTransport _http;

  /// Creates a client.
  AnthropicChatModel({
    required this.apiKey,
    this.model = 'claude-opus-5',
    this.maxTokens = 16000,
    this.streamMaxTokens = 64000,
    this.adaptiveThinking = true,
    this.effort,
    this.serverSideFallbacks,
    this.promptCaching = true,
    Uri? baseUrl,
    this.headers = const {},
    Duration timeout = const Duration(minutes: 10),
    int maxRetries = 2,
    HttpClient? httpClient,
  }) : baseUrl = baseUrl ?? Uri.parse('https://api.anthropic.com'),
       _http = HttpTransport(
         client: httpClient,
         timeout: timeout,
         maxRetries: maxRetries,
       );

  /// Creates a client from the `ANTHROPIC_API_KEY` environment variable.
  factory AnthropicChatModel.fromEnvironment({String model = 'claude-opus-5'}) {
    final key = Platform.environment['ANTHROPIC_API_KEY'];
    if (key == null || key.isEmpty) {
      throw const LlmException('ANTHROPIC_API_KEY is not set');
    }
    return AnthropicChatModel(apiKey: key, model: model);
  }

  bool get _fallbacks =>
      serverSideFallbacks ??
      (model == 'claude-opus-5' || model == 'claude-fable-5-1');

  Uri get _endpoint => joinUrl(baseUrl, 'v1/messages');

  Map<String, String> get _headers {
    final betas = [
      if (_fallbacks) 'server-side-fallback-2026-07-01',
      ?headers['anthropic-beta'],
    ];
    return {
      ...headers,
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      if (betas.isNotEmpty) 'anthropic-beta': betas.join(','),
    };
  }

  Map<String, Object?> _body(
    List<ChatMessage> messages,
    String? system,
    int limit, {
    required bool stream,
  }) {
    final (sys, rest) = splitSystem(messages, system);
    if (rest.isEmpty) {
      throw ArgumentError.value(messages, 'messages', 'needs a user message');
    }
    return {
      'model': model,
      'max_tokens': limit,
      'system': ?sys,
      'messages': [
        for (final m in rest) {'role': m.role.name, 'content': m.content},
      ],
      if (adaptiveThinking) 'thinking': {'type': 'adaptive'},
      if (effort != null) 'output_config': {'effort': effort!.name},
      if (_fallbacks) 'fallbacks': 'default',
      if (promptCaching) 'cache_control': {'type': 'ephemeral'},
      if (stream) 'stream': true,
    };
  }

  static LlmRefusalException _refusal(Object? details) {
    final d = details is Map ? details : const {};
    return LlmRefusalException(
      '${d['explanation'] ?? 'Claude declined the request'}',
      category: d['category'] as String?,
    );
  }

  static ChatUsage? _usage(Object? raw) {
    if (raw is! Map) return null;
    int n(String k) => (raw[k] as num?)?.toInt() ?? 0;
    return ChatUsage(
      inputTokens: n('input_tokens'),
      outputTokens: n('output_tokens'),
      cachedInputTokens: n('cache_read_input_tokens'),
    );
  }

  @override
  Future<ChatResponse> complete(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async {
    final json = await _http.postJson(
      _endpoint,
      _headers,
      _body(messages, system, maxTokens ?? this.maxTokens, stream: false),
    );
    final stop = json['stop_reason'] as String?;
    // Check the stop reason before reading content: a refusal can come with
    // empty or partial content.
    if (stop == 'refusal') throw _refusal(json['stop_details']);
    final text = StringBuffer();
    for (final block in (json['content'] as List? ?? const [])) {
      // Only text: thinking and fallback-marker blocks are not the answer.
      if (block is Map && block['type'] == 'text') text.write(block['text']);
    }
    return ChatResponse(
      text.toString(),
      stopReason: stop,
      model: json['model'] as String?,
      usage: _usage(json['usage']),
    );
  }

  @override
  Stream<String> stream(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async* {
    String? stop;
    Object? details;
    final events = _http.postSse(
      _endpoint,
      _headers,
      _body(messages, system, maxTokens ?? streamMaxTokens, stream: true),
    );
    await for (final e in events) {
      if (e.event == 'ping') continue;
      final Object? json;
      try {
        json = jsonDecode(e.data);
      } on FormatException {
        continue;
      }
      if (json is! Map) continue;
      switch (json['type']) {
        case 'content_block_delta':
          final delta = json['delta'];
          if (delta is Map && delta['type'] == 'text_delta') {
            final text = delta['text'] as String? ?? '';
            if (text.isNotEmpty) yield text;
          }
        case 'message_delta':
          final delta = json['delta'];
          if (delta is Map) {
            stop = delta['stop_reason'] as String? ?? stop;
            details = delta['stop_details'] ?? details;
          }
        case 'error':
          final error = json['error'];
          final map = error is Map ? error : const {};
          final type = map['type'] as String?;
          throw LlmHttpException(
            type == 'overloaded_error' ? 529 : 500,
            '${map['message'] ?? 'stream error'}',
            errorType: type,
          );
        case 'message_stop':
          break;
      }
    }
    if (stop == 'refusal') throw _refusal(details);
  }

  @override
  void close() => _http.close();
}
