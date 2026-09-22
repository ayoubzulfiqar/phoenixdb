/// AI toolkit for PhoenixDB: embeddings, LLM clients, retrieval-augmented
/// generation, semantic caching and conversation memory, all stored in
/// on-device [PhoenixCollection]s.
///
/// ```dart
/// import 'package:phoenixdb/ai.dart';
/// import 'package:phoenixdb/phoenixdb.dart';
///
/// final kb = await AsyncPhoenixCollection.open('kb', dimensions: 1024);
/// final rag = RagPipeline(
///   store: kb,
///   embedder: OpenAICompatibleEmbedder.voyage(
///       apiKey: voyageKey, model: 'voyage-3.5', dimensions: 1024),
///   chat: AnthropicChatModel(apiKey: anthropicKey),
/// );
/// await rag.ingest(RagDocument('faq', faqText, metadata: {'lang': 'en'}));
/// final answer = await rag.ask('How do I reset my password?');
/// print(answer.text);                       // cites sources as [1], [2]…
/// print(answer.citations.map((s) => s.documentId));
/// ```
///
/// * **Chat models** — [AnthropicChatModel] (Claude, via the Messages API)
///   and [OpenAICompatibleChatModel] (OpenAI, Ollama, Gemini, vLLM, LM
///   Studio, Mistral, Groq, OpenRouter, …), behind one [ChatModel]
///   interface with streaming.
/// * **Embedders** — [OpenAICompatibleEmbedder] (OpenAI, Voyage AI, Ollama,
///   …), the offline [HashingEmbedder], and [CachedEmbedder], which
///   persists vectors in a [PhoenixDatabase] so no text is embedded twice.
/// * **Retrieval** — [TextChunker] and [RagPipeline]: hybrid vector + BM25
///   retrieval with MMR, grounded answers with numbered citations.
/// * **Caching and memory** — [SemanticCache] reuses answers to prompts
///   that mean the same thing; [ConversationMemory] keeps recent turns and
///   recalls relevant older ones.
///
/// Network clients use `dart:io`, so this library runs on the Dart VM and
/// Flutter's native platforms (not the web — neither does PhoenixDB).
library;

import 'phoenixdb.dart';

export 'src/ai/anthropic.dart' show AnthropicChatModel, ClaudeEffort;
export 'src/ai/chat.dart'
    show ChatModel, ChatMessage, ChatRole, ChatResponse, ChatUsage;
export 'src/ai/chunker.dart' show TextChunker, TextChunk;
export 'src/ai/embedder.dart'
    show Embedder, EmbedPurpose, EmbedOne, HashingEmbedder, CachedEmbedder;
export 'src/ai/http.dart'
    show LlmException, LlmHttpException, LlmRefusalException;
export 'src/ai/memory.dart' show ConversationMemory;
export 'src/ai/openai_compatible.dart'
    show OpenAICompatibleChatModel, OpenAICompatibleEmbedder;
export 'src/ai/rag.dart'
    show
        RagPipeline,
        RagDocument,
        RagSource,
        RagAnswer,
        RagStream,
        citedSources;
export 'src/ai/semantic_cache.dart' show SemanticCache, CachedChatModel;
