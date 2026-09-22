/// On-device retrieval-augmented generation: ingest a small knowledge base,
/// search it with hybrid vector + keyword retrieval, and answer questions
/// with cited sources.
///
/// Run with:
/// ```sh
/// dart run example/rag_example.dart
/// ```
///
/// It runs fully offline with [HashingEmbedder] (lexical embeddings) and a
/// stand-in model that just lists the retrieved sources. Set
/// `ANTHROPIC_API_KEY` to have Claude write the answers instead; swap in
/// `OpenAICompatibleEmbedder` (OpenAI, Voyage AI, Ollama, …) for semantic
/// embeddings.
library;

import 'dart:io';

import 'package:phoenixdb/ai.dart';
import 'package:phoenixdb/phoenixdb.dart';

const docs = [
  RagDocument(
    'vacation',
    'Vacation policy. Full-time employees receive 25 paid vacation days per '
        'year. Unused days carry over for up to twelve months.',
    metadata: {'team': 'hr', 'updated': 2026},
  ),
  RagDocument(
    'expenses',
    'Expense policy. Submit receipts within 30 days of purchase through the '
        'finance portal. Travel above 500 euros needs prior approval.',
    metadata: {'team': 'finance', 'updated': 2025},
  ),
  RagDocument(
    'security',
    'Security policy. Lock your laptop whenever you leave your desk and '
        'report lost devices to the IT desk within one hour.',
    metadata: {'team': 'it', 'updated': 2026},
  ),
];

/// Offline stand-in for a chat model: answers by quoting the first source.
class QuoteFirstSource implements ChatModel {
  @override
  String get model => 'quote-first-source';

  @override
  Future<ChatResponse> complete(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async {
    final prompt = messages.last.content;
    final match = RegExp(
      r'<source number="1"[^>]*>\n([\s\S]*?)\n</source>',
    ).firstMatch(prompt);
    return ChatResponse(
      match == null
          ? 'The sources do not cover this.'
          : 'According to the knowledge base: ${match.group(1)} [1]',
    );
  }

  @override
  Stream<String> stream(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async* {
    yield (await complete(messages, system: system)).text;
  }

  @override
  void close() {}
}

Future<void> main() async {
  final dir = Directory.systemTemp.createTempSync('phoenix_rag_example_');
  final embedder = HashingEmbedder(dimensions: 256);
  final key = Platform.environment['ANTHROPIC_API_KEY'];
  final ChatModel chat = key == null || key.isEmpty
      ? QuoteFirstSource()
      : AnthropicChatModel(apiKey: key);
  final kb = await AsyncPhoenixCollection.open(
    '${dir.path}/kb',
    dimensions: embedder.dimensions,
  );
  try {
    final rag = RagPipeline(store: kb, embedder: embedder, chat: chat, k: 2);
    final chunks = await rag.ingestAll(docs);
    print('Ingested ${docs.length} documents as $chunks chunks.\n');

    for (final question in [
      'How many vacation days do employees get?',
      'What is the deadline for submitting receipts?',
    ]) {
      final answer = await rag.ask(question);
      print('Q: $question');
      print('A: ${answer.text}');
      for (final source in answer.citations) {
        print('   [${source.number}] ${source.documentId}');
      }
      print('');
    }

    // Metadata filters narrow retrieval before ranking.
    final recent = await rag.retrieve(
      'policy',
      filter: Filter.gte('updated', 2026) & ~Filter.eq('team', 'it'),
    );
    print(
      '2026 policies outside IT: '
      '${recent.map((s) => s.documentId).toSet().join(', ')}',
    );

    // The same store answers structured questions directly.
    for (final team in ['hr', 'finance', 'it']) {
      final n = await kb.count(filter: Filter.eq('team', team));
      print('$team: $n chunk(s)');
    }
  } finally {
    await kb.close();
    chat.close();
    dir.deleteSync(recursive: true);
  }
}
