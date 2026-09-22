/// Semantic response caching: reuse an answer when a new prompt means the
/// same thing as one already answered.
library;

import 'dart:convert';

import '../collection.dart';
import 'chat.dart';
import 'embedder.dart';

/// Caches model responses by prompt similarity.
///
/// Prompts are embedded and stored with their responses in a
/// [DocumentStore]; a lookup returns the response of the most similar prior
/// prompt when its similarity reaches [threshold]. Use a cosine-metric
/// collection (the default), so similarity is a cosine in `[-1, 1]`.
///
/// Entries are scoped by [namespace] — use one per model and system prompt,
/// since a cached answer is only valid for the setup that produced it — and
/// expire after [ttl].
class SemanticCache {
  /// Where entries live (may be shared with other data; entries are tagged).
  final DocumentStore store;

  /// Embeds prompts.
  final Embedder embedder;

  /// Minimum cosine similarity for a hit.
  final double threshold;

  /// Entry lifetime, or `null` for no expiry.
  final Duration? ttl;

  /// Scope of the entries.
  final String namespace;

  /// Lookups answered from the cache.
  int hits = 0;

  /// Lookups that missed.
  int misses = 0;

  /// Creates a cache.
  SemanticCache({
    required this.store,
    required this.embedder,
    this.namespace = 'default',
    this.threshold = 0.95,
    this.ttl,
  }) {
    if (store.dimensions != embedder.dimensions) {
      throw ArgumentError(
        'the store holds ${store.dimensions}-dimensional vectors but the '
        'embedder produces ${embedder.dimensions}',
      );
    }
  }

  Filter _scope() => Filter.and([
    Filter.eq('kind', 'semantic-cache'),
    Filter.eq('namespace', namespace),
    if (ttl != null)
      Filter.gte(
        'created_at',
        DateTime.now().millisecondsSinceEpoch - ttl!.inMilliseconds,
      ),
  ]);

  static String _id(String namespace, String prompt) {
    // Two independent 32-bit FNV-1a hashes: ids must stay under 128 bytes.
    var a = 0x811c9dc5, b = 0x050c5d1f;
    for (final x in utf8.encode('$namespace\u0000$prompt')) {
      a = ((a ^ x) * 0x01000193) & 0xffffffff;
      b = ((b ^ x) * 0x01000193 + 0x9e37) & 0xffffffff;
    }
    return 'cache:${a.toRadixString(16).padLeft(8, '0')}'
        '${b.toRadixString(16).padLeft(8, '0')}';
  }

  /// The cached response for a prompt similar to [prompt], or `null`.
  Future<String?> lookup(String prompt) async {
    final vector = await embedder.embedOne(prompt, purpose: EmbedPurpose.query);
    final found = await store.query(
      CollectionQuery(
        vector: vector,
        filter: _scope(),
        k: 1,
        includeText: false,
      ),
    );
    final best = found.isEmpty ? null : found.first;
    final similarity = best?.vectorScore;
    if (best == null || similarity == null || similarity < threshold) {
      misses++;
      return null;
    }
    hits++;
    return best.metadata?['response'] as String?;
  }

  /// Stores [response] as the answer to [prompt].
  Future<void> put(String prompt, String response) async {
    final vector = await embedder.embedOne(prompt);
    await store.upsert([
      Document(
        _id(namespace, prompt),
        metadata: {
          'kind': 'semantic-cache',
          'namespace': namespace,
          'prompt': prompt,
          'response': response,
          'created_at': DateTime.now().millisecondsSinceEpoch,
        },
        vector: vector,
      ),
    ]);
  }

  /// Removes every entry in this namespace; returns how many.
  Future<int> clear() async {
    final entries = await store.list(
      filter:
          Filter.eq('kind', 'semantic-cache') &
          Filter.eq('namespace', namespace),
    );
    if (entries.isEmpty) return 0;
    return store.delete([for (final e in entries) e.id]);
  }

  /// [model] answering from this cache when it can.
  ChatModel wrap(ChatModel model) => CachedChatModel(model, this);
}

/// A [ChatModel] that consults a [SemanticCache] before calling the model.
///
/// The cache key is the whole rendered conversation, so only a conversation
/// that says the same thing hits — typically a first question.
class CachedChatModel implements ChatModel {
  /// The model called on a miss.
  final ChatModel inner;

  /// The cache.
  final SemanticCache cache;

  /// Wraps [inner].
  CachedChatModel(this.inner, this.cache);

  @override
  String get model => inner.model;

  static String _render(List<ChatMessage> messages, String? system) {
    final (sys, rest) = splitSystem(messages, system);
    return [
      if (sys != null) 'system: $sys',
      for (final m in rest) '${m.role.name}: ${m.content}',
    ].join('\n');
  }

  @override
  Future<ChatResponse> complete(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async {
    final key = _render(messages, system);
    final cached = await cache.lookup(key);
    if (cached != null) {
      return ChatResponse(cached, stopReason: 'cache_hit', model: inner.model);
    }
    final response = await inner.complete(
      messages,
      system: system,
      maxTokens: maxTokens,
    );
    // Truncated answers are not worth replaying.
    if (!response.truncated) await cache.put(key, response.text);
    return response;
  }

  @override
  Stream<String> stream(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async* {
    final key = _render(messages, system);
    final cached = await cache.lookup(key);
    if (cached != null) {
      yield cached;
      return;
    }
    final text = StringBuffer();
    await for (final delta in inner.stream(
      messages,
      system: system,
      maxTokens: maxTokens,
    )) {
      text.write(delta);
      yield delta;
    }
    await cache.put(key, text.toString());
  }

  @override
  void close() => inner.close();
}
