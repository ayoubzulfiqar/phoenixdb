/// Retrieval-augmented generation over a PhoenixDB collection.
library;

import 'dart:convert';
import 'dart:typed_data';

import '../collection.dart';
import 'chat.dart';
import 'chunker.dart';
import 'embedder.dart';

/// A document to ingest.
class RagDocument {
  /// Stable id; re-ingesting the same id replaces the document's chunks.
  final String id;

  /// Full text.
  final String text;

  /// Metadata copied onto every chunk (filterable at query time). The keys
  /// `doc_id`, `chunk`, `chunk_start` and `chunk_end` are reserved.
  final Map<String, Object?> metadata;

  /// Creates a document.
  const RagDocument(this.id, this.text, {this.metadata = const {}});
}

/// A retrieved chunk, numbered for citation.
class RagSource {
  /// 1-based number the model cites as `[n]`.
  final int number;

  /// The underlying search hit.
  final SearchHit hit;

  /// Creates a source.
  const RagSource(this.number, this.hit);

  /// Chunk id.
  String get id => hit.id;

  /// Id of the document the chunk came from.
  String? get documentId => hit.metadata?['doc_id'] as String?;

  /// Chunk text.
  String get text => hit.text ?? '';

  /// Chunk metadata (the document's plus the reserved chunk fields).
  Map<String, Object?> get metadata => hit.metadata ?? const {};

  /// Retrieval score.
  double get score => hit.score;

  @override
  String toString() => 'RagSource([$number] $id, score: $score)';
}

/// A generated answer with its evidence.
class RagAnswer {
  /// The model's answer.
  final String text;

  /// Every source given to the model, in rank order.
  final List<RagSource> sources;

  /// The raw model response, when not streamed.
  final ChatResponse? response;

  /// Creates an answer.
  const RagAnswer(this.text, this.sources, {this.response});

  /// The sources the answer actually cites, in order of first citation.
  List<RagSource> get citations => citedSources(text, sources);

  @override
  String toString() => text;
}

/// An answer being streamed.
class RagStream {
  /// The sources given to the model, available before any text.
  final List<RagSource> sources;

  /// The answer, as text deltas.
  final Stream<String> text;

  /// Creates a stream.
  const RagStream(this.sources, this.text);
}

final _citation = RegExp(r'\[(\d+(?:\s*,\s*\d+)*)\]');

/// The members of [sources] that [text] cites as `[n]` or `[n, m]`, in order
/// of first citation. Numbers without a matching source are ignored.
List<RagSource> citedSources(String text, List<RagSource> sources) {
  final byNumber = {for (final s in sources) s.number: s};
  final seen = <int>{};
  final out = <RagSource>[];
  for (final m in _citation.allMatches(text)) {
    for (final part in m.group(1)!.split(',')) {
      final n = int.parse(part.trim());
      final source = byNumber[n];
      if (source != null && seen.add(n)) out.add(source);
    }
  }
  return out;
}

/// Ingests documents into a [DocumentStore] and answers questions from them.
///
/// ```dart
/// final kb = await AsyncPhoenixCollection.open('kb', dimensions: 1024);
/// final rag = RagPipeline(
///   store: kb,
///   embedder: OpenAICompatibleEmbedder.voyage(
///       apiKey: voyageKey, model: 'voyage-3.5', dimensions: 1024),
///   chat: AnthropicChatModel(apiKey: anthropicKey),
/// );
/// await rag.ingest(RagDocument('handbook', handbookText));
/// final answer = await rag.ask('How many vacation days do I get?');
/// print(answer.text);
/// for (final c in answer.citations) print('[${c.number}] ${c.documentId}');
/// ```
///
/// Retrieval is hybrid by default — embeddings and BM25 keywords fused with
/// RRF — which is markedly more robust than either alone for names, codes
/// and rare terms; MMR then removes near-duplicate chunks.
class RagPipeline {
  /// Where chunks live.
  final DocumentStore store;

  /// Embeds chunks and questions.
  final Embedder embedder;

  /// Writes the answers.
  final ChatModel chat;

  /// Splits documents into chunks.
  final TextChunker chunker;

  /// Chunks retrieved per question.
  final int k;

  /// Fuse keyword (BM25) retrieval with vector retrieval.
  final bool hybrid;

  /// MMR diversity for retrieval, or `null` for plain relevance.
  final double? mmr;

  /// Instructions for the model.
  final String systemPrompt;

  /// Most source characters placed in one prompt; lower-ranked sources
  /// beyond it are left out whole rather than cut mid-text.
  final int maxContextChars;

  /// Default [systemPrompt].
  static const defaultSystemPrompt =
      'Answer the question using the numbered sources provided with it. '
      'Base the answer only on those sources; if they do not contain the '
      'answer, say so plainly instead of guessing. Cite the source of each '
      'claim with its number in square brackets, like [2] or [1][3]. Keep '
      'the answer focused and concise.';

  /// Creates a pipeline. [embedder] must produce vectors of the store's
  /// dimensionality.
  RagPipeline({
    required this.store,
    required this.embedder,
    required this.chat,
    TextChunker? chunker,
    this.k = 6,
    this.hybrid = true,
    this.mmr = 0.7,
    this.systemPrompt = defaultSystemPrompt,
    this.maxContextChars = 24000,
  }) : chunker = chunker ?? TextChunker() {
    if (store.dimensions != embedder.dimensions) {
      throw ArgumentError(
        'the store holds ${store.dimensions}-dimensional vectors but the '
        'embedder produces ${embedder.dimensions}',
      );
    }
    if (k <= 0) throw ArgumentError.value(k, 'k', 'must be positive');
  }

  static String _chunkId(String documentId, int index) =>
      '$documentId#${index.toString().padLeft(5, '0')}';

  /// Ingests one document; returns its chunk count.
  Future<int> ingest(RagDocument document) => ingestAll([document]);

  /// Ingests documents (embedding all their chunks in batched calls) and
  /// returns the total chunk count. Re-ingesting a document replaces it:
  /// new chunks are written first, then stale ones removed, so the document
  /// never disappears in between.
  Future<int> ingestAll(Iterable<RagDocument> documents) async {
    final docs = documents.toList(growable: false);
    final pending = <(RagDocument, TextChunk)>[];
    for (final d in docs) {
      // Chunk ids append `#nnnnn` and must fit the engine's 128-byte limit.
      final bytes = utf8.encode(d.id).length;
      if (bytes == 0 || bytes > 120) {
        throw ArgumentError.value(d.id, 'id', 'must be 1..=120 bytes of UTF-8');
      }
      for (final c in chunker.split(d.text)) {
        pending.add((d, c));
      }
    }
    final vectors = pending.isEmpty
        ? const <Float32List>[]
        : await embedder.embed([for (final (_, c) in pending) c.text]);
    final chunks = <Document>[
      for (var i = 0; i < pending.length; i++)
        Document(
          _chunkId(pending[i].$1.id, pending[i].$2.index),
          text: pending[i].$2.text,
          metadata: {
            ...pending[i].$1.metadata,
            'doc_id': pending[i].$1.id,
            'chunk': pending[i].$2.index,
            'chunk_start': pending[i].$2.start,
            'chunk_end': pending[i].$2.end,
          },
          vector: vectors[i],
        ),
    ];
    // Upsert in slices to keep individual transactions moderate.
    for (var i = 0; i < chunks.length; i += 256) {
      await store.upsert(
        chunks.sublist(i, i + 256 < chunks.length ? i + 256 : chunks.length),
      );
    }
    final fresh = {for (final c in chunks) c.id};
    for (final d in docs) {
      final existing = await store.list(filter: Filter.eq('doc_id', d.id));
      final stale = [
        for (final c in existing)
          if (!fresh.contains(c.id)) c.id,
      ];
      if (stale.isNotEmpty) await store.delete(stale);
    }
    return chunks.length;
  }

  /// Removes a document's chunks; returns how many were removed.
  Future<int> remove(String documentId) async {
    final existing = await store.list(filter: Filter.eq('doc_id', documentId));
    if (existing.isEmpty) return 0;
    return store.delete([for (final c in existing) c.id]);
  }

  /// The chunks most relevant to [question].
  Future<List<RagSource>> retrieve(
    String question, {
    int? k,
    Filter? filter,
  }) async {
    final vector = await embedder.embedOne(
      question,
      purpose: EmbedPurpose.query,
    );
    final hits = await store.query(
      CollectionQuery(
        vector: vector,
        text: hybrid ? question : null,
        filter: filter,
        k: k ?? this.k,
        mmr: mmr,
      ),
    );
    return [for (var i = 0; i < hits.length; i++) RagSource(i + 1, hits[i])];
  }

  /// The prompt for [question] over [sources], after [history].
  List<ChatMessage> buildMessages(
    String question,
    List<RagSource> sources, {
    List<ChatMessage> history = const [],
  }) {
    final context = StringBuffer('<sources>\n');
    var used = 0;
    for (final s in sources) {
      if (used > 0 && used + s.text.length > maxContextChars) break;
      used += s.text.length;
      context
        ..write('<source number="${s.number}"')
        ..write(s.documentId == null ? '' : ' document="${s.documentId}"')
        ..write('>\n')
        ..write(s.text)
        ..write('\n</source>\n');
    }
    context.write('</sources>\n\nQuestion: $question');
    return [...history, ChatMessage.user(context.toString())];
  }

  /// Answers [question] from the most relevant chunks.
  Future<RagAnswer> ask(
    String question, {
    int? k,
    Filter? filter,
    List<ChatMessage> history = const [],
    int? maxTokens,
  }) async {
    final sources = await retrieve(question, k: k, filter: filter);
    final response = await chat.complete(
      buildMessages(question, sources, history: history),
      system: systemPrompt,
      maxTokens: maxTokens,
    );
    return RagAnswer(response.text, sources, response: response);
  }

  /// Like [ask], streaming the answer; the sources are known up front.
  Future<RagStream> askStream(
    String question, {
    int? k,
    Filter? filter,
    List<ChatMessage> history = const [],
    int? maxTokens,
  }) async {
    final sources = await retrieve(question, k: k, filter: filter);
    return RagStream(
      sources,
      chat.stream(
        buildMessages(question, sources, history: history),
        system: systemPrompt,
        maxTokens: maxTokens,
      ),
    );
  }
}
