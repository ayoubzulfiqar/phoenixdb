/// Long-term conversation memory: recent turns plus semantically recalled
/// older ones.
library;

import 'dart:convert';
import 'dart:math';

import '../collection.dart';
import 'chat.dart';
import 'embedder.dart';
import 'http.dart' show LlmException;

/// Persists a conversation and assembles its context for the next turn.
///
/// Every message is stored with its embedding, so besides the most recent
/// turns [context] can recall older turns relevant to the new question —
/// the conversation can outgrow any context window without forgetting.
///
/// Messages are keyed by a monotonic timestamp rather than a position, so two
/// handles on one conversation (a UI isolate and a background job, say) can
/// both append without overwriting each other, and deleting messages never
/// makes a later append collide with a surviving one. Each append first reads
/// the newest stored message, so the order holds across handles even where the
/// platform clock is coarse; two genuinely simultaneous appends may still tie,
/// and then their relative order is arbitrary but neither is lost.
class ConversationMemory {
  /// Where messages live (may be shared with other data; messages are
  /// tagged with their conversation).
  final DocumentStore store;

  /// Embeds messages and questions.
  final Embedder embedder;

  /// Conversation id.
  final String conversationId;

  /// Messages this handle has seen: the count when it opened plus what it has
  /// appended. Another handle's appends are not reflected; [count] asks the
  /// store.
  int length;

  final Random _random = Random();
  int _lastStamp = 0;

  ConversationMemory._(
    this.store,
    this.embedder,
    this.conversationId,
    this.length,
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
    final existing = await store.count(filter: _scopeOf(conversationId));
    return ConversationMemory._(store, embedder, conversationId, existing);
  }

  static Filter _scopeOf(String conversationId) =>
      Filter.eq('kind', 'memory') & Filter.eq('conversation', conversationId);

  Filter get _scope => _scopeOf(conversationId);

  /// Messages stored in this conversation right now, from the store.
  Future<int> count() => store.count(filter: _scope);

  /// A strictly increasing stamp, wide enough to sort as text.
  ///
  /// Microseconds since the epoch, forced to advance even when several
  /// messages are appended inside one clock tick.
  int _stamp() {
    final now = DateTime.now().microsecondsSinceEpoch;
    _lastStamp = now > _lastStamp ? now : _lastStamp + 1;
    return _lastStamp;
  }

  /// Catches this handle up to the newest message already stored, so stamps
  /// keep increasing across handles as well as within one.
  ///
  /// Without it, ordering would only be as fine as the platform clock: on
  /// Windows a tick is a millisecond or more, so two handles appending in the
  /// same tick would get equal stamps and their order would come down to the
  /// random part of the id. One extra read per append buys a real order.
  Future<void> _catchUp() async {
    final newest = await store.list(
      filter: _scope,
      limit: 1,
      newestFirst: true,
    );
    if (newest.isEmpty) return;
    final stored = _stampOf(newest.first);
    if (stored > _lastStamp) _lastStamp = stored;
  }

  /// Zero-padded so id order is chronological order, with a random tail so two
  /// handles writing in the same microsecond cannot land on one id.
  String _id(int stamp) {
    final tail = _random.nextInt(0x10000).toRadixString(16).padLeft(4, '0');
    return '$conversationId/${stamp.toString().padLeft(19, '0')}-$tail';
  }

  static ChatMessage _message(Document d) {
    final role = d.metadata?['role'];
    return ChatMessage(
      role is String ? ChatRole.values.byName(role) : ChatRole.user,
      d.text ?? '',
    );
  }

  static int _stampOf(Document d) {
    final seq = d.metadata?['seq'];
    return seq is num ? seq.toInt() : 0;
  }

  /// Appends messages to the conversation.
  Future<void> addAll(List<ChatMessage> messages) async {
    if (messages.isEmpty) return;
    await _catchUp();
    final vectors = await embedder.embed([for (final m in messages) m.content]);
    if (vectors.length != messages.length) {
      throw LlmException(
        'the embedder returned ${vectors.length} vectors for '
        '${messages.length} messages',
      );
    }
    final docs = <Document>[];
    for (var i = 0; i < messages.length; i++) {
      final stamp = _stamp();
      docs.add(
        Document(
          _id(stamp),
          text: messages[i].content,
          metadata: {
            'kind': 'memory',
            'conversation': conversationId,
            'role': messages[i].role.name,
            'seq': stamp,
            'at': DateTime.now().millisecondsSinceEpoch,
          },
          vector: vectors[i],
        ),
      );
    }
    await store.upsert(docs);
    length += messages.length;
  }

  /// Appends one message.
  Future<void> add(ChatMessage message) => addAll([message]);

  /// The last [n] messages, oldest first.
  Future<List<ChatMessage>> recent([int n = 10]) async {
    if (n <= 0) return const [];
    // Newest-first with a limit, so the cost is the page rather than the
    // whole conversation.
    final newest = await store.list(
      filter: _scope,
      limit: n,
      newestFirst: true,
    );
    return [for (final d in newest.reversed) _message(d)];
  }

  /// Up to [k] earlier messages most relevant to [query], oldest first,
  /// excluding the last [skipRecent] (which [recent] already covers).
  Future<List<ChatMessage>> recall(
    String query, {
    int k = 4,
    int skipRecent = 0,
  }) async {
    if (k <= 0) return const [];
    var scope = _scope;
    if (skipRecent > 0) {
      final newest = await store.list(
        filter: _scope,
        limit: skipRecent,
        newestFirst: true,
      );
      if (newest.length < skipRecent) return const [];
      // Everything strictly older than the oldest message `recent` will show.
      scope = scope & Filter.lt('seq', _stampOf(newest.last));
    }
    final hits = [
      ...await store.query(
        CollectionQuery(
          vector: await embedder.embedOne(query, purpose: EmbedPurpose.query),
          text: query,
          filter: scope,
          k: k,
        ),
      ),
    ];
    hits.sort((a, b) {
      final x = a.metadata?['seq'];
      final y = b.metadata?['seq'];
      return (x is num ? x : 0).compareTo(y is num ? y : 0);
    });
    return [
      for (final h in hits)
        ChatMessage(switch (h.metadata?['role']) {
          final String role => ChatRole.values.byName(role),
          _ => ChatRole.user,
        }, h.text ?? ''),
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
    if (docs.isEmpty) {
      length = 0;
      return 0;
    }
    final removed = await store.delete([for (final d in docs) d.id]);
    // Only after the delete succeeded: believing the conversation empty while
    // its messages survive is how a later append overwrites them.
    length = 0;
    return removed;
  }
}
