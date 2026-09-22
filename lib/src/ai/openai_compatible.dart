/// Models behind the widely implemented OpenAI-style HTTP API
/// (`/chat/completions`, `/embeddings`): OpenAI, Ollama, LM Studio, vLLM,
/// llama.cpp server, Mistral, Groq, OpenRouter, Together, Voyage AI
/// embeddings, Gemini's compatibility endpoint and many more.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'chat.dart';
import 'embedder.dart';
import 'http.dart';

Map<String, String> _auth(String? apiKey, Map<String, String> headers) => {
  ...headers,
  if (apiKey != null && apiKey.isNotEmpty) 'authorization': 'Bearer $apiKey',
};

/// A [ChatModel] for any OpenAI-compatible chat-completions server.
class OpenAICompatibleChatModel implements ChatModel {
  @override
  final String model;

  /// API root, e.g. `https://api.openai.com/v1` or `http://localhost:11434/v1`.
  final Uri baseUrl;

  /// Bearer token; `null` for local servers that need none.
  final String? apiKey;

  /// Output limit, or `null` for the server default.
  final int? maxTokens;

  /// Name of the output-limit field (`max_tokens`, or
  /// `max_completion_tokens` for OpenAI reasoning models).
  final String maxTokensField;

  /// Extra body fields (e.g. `temperature`, `reasoning_effort`).
  final Map<String, Object?> extraBody;

  /// Extra headers.
  final Map<String, String> headers;

  final HttpTransport _http;

  /// Creates a client.
  OpenAICompatibleChatModel({
    required this.baseUrl,
    required this.model,
    this.apiKey,
    this.maxTokens,
    this.maxTokensField = 'max_tokens',
    this.extraBody = const {},
    this.headers = const {},
    Duration timeout = const Duration(minutes: 10),
    int maxRetries = 2,
    HttpClient? httpClient,
  }) : _http = HttpTransport(
         client: httpClient,
         timeout: timeout,
         maxRetries: maxRetries,
       );

  /// OpenAI's API.
  factory OpenAICompatibleChatModel.openAI({
    required String apiKey,
    required String model,
    int? maxTokens,
  }) => OpenAICompatibleChatModel(
    baseUrl: Uri.parse('https://api.openai.com/v1'),
    model: model,
    apiKey: apiKey,
    maxTokens: maxTokens,
    maxTokensField: 'max_completion_tokens',
  );

  /// A local Ollama server.
  factory OpenAICompatibleChatModel.ollama({
    required String model,
    Uri? host,
    int? maxTokens,
  }) => OpenAICompatibleChatModel(
    baseUrl: joinUrl(host ?? Uri.parse('http://localhost:11434'), 'v1'),
    model: model,
    maxTokens: maxTokens,
  );

  /// Google Gemini through its OpenAI-compatible endpoint.
  factory OpenAICompatibleChatModel.gemini({
    required String apiKey,
    required String model,
    int? maxTokens,
  }) => OpenAICompatibleChatModel(
    baseUrl: Uri.parse(
      'https://generativelanguage.googleapis.com/v1beta/openai',
    ),
    model: model,
    apiKey: apiKey,
    maxTokens: maxTokens,
  );

  Map<String, Object?> _body(
    List<ChatMessage> messages,
    String? system,
    int? limit, {
    required bool stream,
  }) {
    final (sys, rest) = splitSystem(messages, system);
    if (rest.isEmpty) {
      throw ArgumentError.value(messages, 'messages', 'needs a user message');
    }
    return {
      ...extraBody,
      'model': model,
      'messages': [
        if (sys != null) {'role': 'system', 'content': sys},
        for (final m in rest) {'role': m.role.name, 'content': m.content},
      ],
      if ((limit ?? maxTokens) != null) maxTokensField: limit ?? maxTokens,
      if (stream) 'stream': true,
    };
  }

  static ChatUsage? _usage(Object? raw) {
    if (raw is! Map) return null;
    final details = raw['prompt_tokens_details'];
    return ChatUsage(
      inputTokens: (raw['prompt_tokens'] as num?)?.toInt() ?? 0,
      outputTokens: (raw['completion_tokens'] as num?)?.toInt() ?? 0,
      cachedInputTokens: details is Map
          ? (details['cached_tokens'] as num?)?.toInt() ?? 0
          : 0,
    );
  }

  @override
  Future<ChatResponse> complete(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async {
    final json = await _http.postJson(
      joinUrl(baseUrl, 'chat/completions'),
      _auth(apiKey, headers),
      _body(messages, system, maxTokens, stream: false),
    );
    final choices = json['choices'];
    if (choices is! List || choices.isEmpty) {
      throw const LlmException('the response has no choices');
    }
    final choice = choices.first as Map;
    final message = choice['message'] as Map? ?? const {};
    final refusal = message['refusal'];
    if (refusal is String && refusal.isNotEmpty) {
      throw LlmRefusalException(refusal);
    }
    return ChatResponse(
      message['content'] as String? ?? '',
      stopReason: choice['finish_reason'] as String?,
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
    final events = _http.postSse(
      joinUrl(baseUrl, 'chat/completions'),
      _auth(apiKey, headers),
      _body(messages, system, maxTokens, stream: true),
    );
    await for (final e in events) {
      if (e.data == '[DONE]') break;
      final Object? json;
      try {
        json = jsonDecode(e.data);
      } on FormatException {
        continue;
      }
      if (json is! Map) continue;
      final error = json['error'];
      if (error is Map) {
        throw LlmHttpException(
          500,
          '${error['message'] ?? 'stream error'}',
          errorType: error['type'] as String?,
        );
      }
      final choices = json['choices'];
      if (choices is! List || choices.isEmpty) continue;
      final delta = (choices.first as Map)['delta'];
      if (delta is! Map) continue;
      final refusal = delta['refusal'];
      if (refusal is String && refusal.isNotEmpty) {
        throw LlmRefusalException(refusal);
      }
      final text = delta['content'];
      if (text is String && text.isNotEmpty) yield text;
    }
  }

  @override
  void close() => _http.close();
}

/// An [Embedder] for any OpenAI-compatible `/embeddings` endpoint.
class OpenAICompatibleEmbedder implements Embedder {
  @override
  final int dimensions;

  /// Embedding model name.
  final String model;

  /// API root.
  final Uri baseUrl;

  /// Bearer token; `null` for local servers.
  final String? apiKey;

  /// Texts per request.
  final int batchSize;

  /// Send `dimensions` in the request (models with adjustable output size).
  final bool requestDimensions;

  /// Body field that tells the server whether a text is a query or a
  /// document (`input_type` for Voyage AI), or `null` when unsupported.
  final String? purposeField;

  /// Extra body fields.
  final Map<String, Object?> extraBody;

  /// Extra headers.
  final Map<String, String> headers;

  final HttpTransport _http;

  /// Creates an embedder.
  OpenAICompatibleEmbedder({
    required this.baseUrl,
    required this.model,
    required this.dimensions,
    this.apiKey,
    this.batchSize = 128,
    this.requestDimensions = false,
    this.purposeField,
    this.extraBody = const {},
    this.headers = const {},
    Duration timeout = const Duration(minutes: 2),
    int maxRetries = 2,
    HttpClient? httpClient,
  }) : _http = HttpTransport(
         client: httpClient,
         timeout: timeout,
         maxRetries: maxRetries,
       ) {
    if (dimensions <= 0) {
      throw ArgumentError.value(dimensions, 'dimensions', 'must be positive');
    }
    if (batchSize <= 0) {
      throw ArgumentError.value(batchSize, 'batchSize', 'must be positive');
    }
  }

  /// OpenAI embeddings (e.g. `text-embedding-3-small`, 1536 dimensions).
  factory OpenAICompatibleEmbedder.openAI({
    required String apiKey,
    required String model,
    required int dimensions,
  }) => OpenAICompatibleEmbedder(
    baseUrl: Uri.parse('https://api.openai.com/v1'),
    model: model,
    dimensions: dimensions,
    apiKey: apiKey,
    requestDimensions: true,
  );

  /// Voyage AI embeddings (Anthropic's recommended embedding provider),
  /// with query/document input types.
  factory OpenAICompatibleEmbedder.voyage({
    required String apiKey,
    required String model,
    required int dimensions,
  }) => OpenAICompatibleEmbedder(
    baseUrl: Uri.parse('https://api.voyageai.com/v1'),
    model: model,
    dimensions: dimensions,
    apiKey: apiKey,
    purposeField: 'input_type',
  );

  /// A local Ollama server (e.g. `nomic-embed-text`, 768 dimensions).
  factory OpenAICompatibleEmbedder.ollama({
    required String model,
    required int dimensions,
    Uri? host,
  }) => OpenAICompatibleEmbedder(
    baseUrl: joinUrl(host ?? Uri.parse('http://localhost:11434'), 'v1'),
    model: model,
    dimensions: dimensions,
  );

  @override
  Future<List<Float32List>> embed(
    List<String> texts, {
    EmbedPurpose purpose = EmbedPurpose.document,
  }) async {
    final out = <Float32List>[];
    for (var i = 0; i < texts.length; i += batchSize) {
      final batch = texts.sublist(i, (i + batchSize).clamp(0, texts.length));
      final json = await _http
          .postJson(joinUrl(baseUrl, 'embeddings'), _auth(apiKey, headers), {
            ...extraBody,
            'model': model,
            'input': batch,
            if (requestDimensions) 'dimensions': dimensions,
            ?purposeField: purpose.name,
          });
      final data = json['data'];
      if (data is! List || data.length != batch.length) {
        throw LlmException(
          'expected ${batch.length} embeddings, got '
          '${data is List ? data.length : 'none'}',
        );
      }
      final ordered = [...data.cast<Map>()]
        ..sort(
          (a, b) =>
              ((a['index'] as num?) ?? 0).compareTo((b['index'] as num?) ?? 0),
        );
      for (final item in ordered) {
        final raw = item['embedding'];
        if (raw is! List || raw.length != dimensions) {
          throw LlmException(
            'expected $dimensions-dimensional embeddings from $model, got '
            '${raw is List ? raw.length : 'none'}',
          );
        }
        out.add(
          Float32List.fromList([for (final v in raw) (v as num).toDouble()]),
        );
      }
    }
    return out;
  }

  @override
  void close() => _http.close();
}
