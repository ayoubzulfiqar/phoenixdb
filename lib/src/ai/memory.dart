/// Long-term conversation memory: recent turns plus semantically recalled
/// older ones.
library;

import 'dart:convert';

import '../collection.dart';
import 'chat.dart';
import 'embedder.dart';

/// Persists a conversation and assembles its context for the next turn.
///
/// Every message is stored with its embedding, so besides the most recent
/// turns [context] can recall older turns relevant to the new question —
/// the conversation can outgrow any context window without forgetting.
class ConversationMemory {
  /// Where messages live (may be shared with other data; messages are
  /// tagged with their conversation).
  final DocumentStore store;

  /// Embeds messages and questions.
  final Embedder embedder;

  /// Conversation id.
  final String conversationId;

  int _next;

  ConversationMemory._(
    this.store,
    this.embedder,
    this.conversationId,
    this._next,
  );

  /// Opens (or starts) conversation [conversationId].
  static Future<ConversationMemory> open({
    required DocumentStore store,
    required Embedder embedder,
    required String conversationId,
  }) async {
    final bytes = utf8.encode(conversationId).length;
    if (bytes == 0 || bytes > 100) {
      throw ArgumentError.value(
        conversationId,
        'conversationId',
        'must be 1..=100 bytes of UTF-8',
      );
    }
    if (store.dimensions != embedder.dimensions) {
      throw ArgumentError(
        'the store holds ${store.dimensions}-dimensional vectors but the '
        'embedder produces ${embedder.dimensions}',
      );
    }
    final count = await store.count(filter: _scopeOf(conversationId));
    return ConversationMemory._(store, embedder, conversationId, count);
  }

  static Filter _scopeOf(String conversationId) =>
      Filter.eq('kind', 'memory') & Filter.eq('conversation', conversationId);

  Filter get _scope => _scopeOf(conversationId);

  /// Messages stored so far.
  int get length => _next;

  // Zero-padded sequence numbers make id order chronological order.
  String _id(int seq) => '$conversationId/${seq.toString().padLeft(10, '0')}';

  static ChatMessage _message(Document d) => ChatMessage(
    ChatRole.values.byName(d.metadata!['role']! as String),
    d.text ?? '',
  );

  /// Appends messages to the conversation.
  Future<void> addAll(List<ChatMessage> messages) async {
    if (messages.isEmpty) return;
    final vectors = await embedder.embed([for (final m in messages) m.content]);
    final docs = <Document>[];
    for (var i = 0; i < messages.length; i++) {
      docs.add(
        Document(
          _id(_next + i),
          text: messages[i].content,
          metadata: {
            'kind': 'memory',
            'conversation': conversationId,
            'role': messages[i].role.name,
            'seq': _next + i,
            'at': DateTime.now().millisecondsSinceEpoch,
          },
          vector: vectors[i],
        ),
      );
    }
    await store.upsert(docs);
    _next += messages.length;
  }

  /// Appends one message.
  Future<void> add(ChatMessage message) => addAll([message]);

  /// The last [n] messages, oldest first.
  Future<List<ChatMessage>> recent([int n = 10]) async {
    if (n <= 0 || _next == 0) return const [];
    final docs = await store.list(
      filter: _scope & Filter.gte('seq', _next - n),
    );
    return [for (final d in docs) _message(d)];
  }

  /// Up to [k] earlier messages most relevant to [query], oldest first,
  /// excluding the last [skipRecent] (which [recent] already covers).
  Future<List<ChatMessage>> recall(
    String query, {
    int k = 4,
    int skipRecent = 0,
  }) async {
    final cutoff = _next - skipRecent;
    if (k <= 0 || cutoff <= 0) return const [];
    final hits = [
      ...await store.query(
        CollectionQuery(
          vector: await embedder.embedOne(query, purpose: EmbedPurpose.query),
          text: query,
          filter: _scope & Filter.lt('seq', cutoff),
          k: k,
        ),
      ),
    ];
    hits.sort(
      (a, b) =>
          (a.metadata!['seq']! as num).compareTo(b.metadata!['seq']! as num),
    );
    return [
      for (final h in hits)
        ChatMessage(
          ChatRole.values.byName(h.metadata!['role']! as String),
          h.text ?? '',
        ),
    ];
  }

  /// Context for answering [query]: up to [recalled] relevant older
  /// messages (wrapped in a system note so the model knows they are
  /// excerpts), then the last [recentCount] messages.
  Future<List<ChatMessage>> context(
    String query, {
    int recentCount = 8,
    int recalled = 4,
  }) async {
    final older = await recall(query, k: recalled, skipRecent: recentCount);
    final latest = await recent(recentCount);
    return [
      if (older.isNotEmpty)
        ChatMessage.system(
          'Earlier in this conversation (excerpts relevant to the current '
          'question):\n${older.map((m) => '${m.role.name}: ${m.content}').join('\n')}',
        ),
      ...latest,
    ];
  }

  /// Deletes the conversation; returns how many messages were removed.
  Future<int> clear() async {
    final docs = await store.list(filter: _scope);
    _next = 0;
    if (docs.isEmpty) return 0;
    return store.delete([for (final d in docs) d.id]);
  }
}
