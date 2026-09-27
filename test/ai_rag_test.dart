/// The retrieval side of the AI toolkit against real collections: chunking,
/// embedders, RAG with citations, semantic caching and conversation memory.
library;

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:phoenixdb/ai.dart';
import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

/// A scripted model that records what it was asked.
class FakeChat implements ChatModel {
  final String Function(List<ChatMessage> messages, String? system) reply;
  final calls = <(List<ChatMessage>, String?)>[];
  FakeChat(this.reply);

  @override
  String get model => 'fake';

  @override
  Future<ChatResponse> complete(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async {
    calls.add((messages, system));
    return ChatResponse(reply(messages, system), stopReason: 'end_turn');
  }

  @override
  Stream<String> stream(
    List<ChatMessage> messages, {
    String? system,
    int? maxTokens,
  }) async* {
    calls.add((messages, system));
    for (final word in reply(messages, system).split(' ')) {
      yield '$word ';
    }
  }

  @override
  void close() {}
}

double cosine(List<double> a, List<double> b) {
  var d = 0.0;
  for (var i = 0; i < a.length; i++) {
    d += a[i] * b[i];
  }
  return d;
}

void main() {
  late Directory dir;
  setUp(() => dir = Directory.systemTemp.createTempSync('phoenix_ai_'));
  tearDown(() {
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold files briefly.
    }
  });

  group('TextChunker', () {
    test('chunks are bounded, exact and cover the text', () {
      final rng = Random(3);
      const words = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'];
      final text = [
        for (var p = 0; p < 12; p++)
          [
            for (var s = 0; s < 1 + rng.nextInt(6); s++)
              '${[for (var w = 0; w < 3 + rng.nextInt(12); w++) words[rng.nextInt(6)]].join(' ')}.',
          ].join(' '),
      ].join('\n\n');
      final chunker = TextChunker(chunkSize: 200, overlap: 40);
      final chunks = chunker.split(text);
      expect(chunks.length, greaterThan(3));
      var covered = 0;
      for (final c in chunks) {
        expect(c.text.length, lessThanOrEqualTo(200));
        expect(text.substring(c.start, c.end), c.text, reason: 'exact offsets');
        expect(c.text.trim(), c.text);
        final gap = c.start > covered ? text.substring(covered, c.start) : '';
        expect(gap.trim(), isEmpty, reason: 'only whitespace between chunks');
        covered = max(covered, c.end);
      }
      expect(text.substring(covered).trim(), isEmpty);
      expect(
        chunks.skip(1).any((c) => c.start < chunks[c.index - 1].end),
        isTrue,
        reason: 'consecutive chunks overlap',
      );
    });

    test('long unbroken runs are hard-split without breaking surrogates', () {
      final text = '😀' * 150; // 300 UTF-16 code units, no separators
      final chunks = TextChunker(chunkSize: 101, overlap: 0).split(text);
      expect(chunks.map((c) => c.text).join(), text);
      for (final c in chunks) {
        expect(c.text.length, lessThanOrEqualTo(101));
        expect(c.text.runes.every((r) => r == 0x1F600), isTrue);
      }
      expect(TextChunker().split('   \n\n  '), isEmpty);
      expect(
        () => TextChunker(chunkSize: 10, overlap: 10),
        throwsArgumentError,
      );
    });
  });

  group('embedders', () {
    test('HashingEmbedder is deterministic, normalised and lexical', () async {
      final e = HashingEmbedder(dimensions: 128);
      final [a, b, c] = await e.embed([
        'the database stores vectors on device',
        'vectors are stored by the on-device database',
        'bananas are yellow fruit',
      ]);
      expect(cosine(a, a), closeTo(1, 1e-5));
      expect(cosine(a, b), greaterThan(cosine(a, c)));
      expect(e.embedSync('the database stores vectors on device'), a);
    });

    test('CachedEmbedder embeds each text once, and persists', () async {
      var calls = 0;
      final inner = _CountingEmbedder(HashingEmbedder(dimensions: 16), () {
        calls++;
      });
      final db = PhoenixDatabase.open('${dir.path}/emb.pdb');
      final cached = CachedEmbedder(inner, namespace: 'hash16', database: db);
      final first = await cached.embed(['x', 'y', 'x']);
      expect(calls, 1);
      expect(inner.texts, ['x', 'y'], reason: 'duplicates embedded once');
      final again = await cached.embed(['y', 'x']);
      expect(calls, 1, reason: 'served from memory');
      expect(again[1], first[0]);
      expect((cached.hits, cached.misses), (2, 3));

      final fresh = CachedEmbedder(inner, namespace: 'hash16', database: db);
      await fresh.embed(['x']);
      expect(calls, 1, reason: 'served from the database');
      final other = CachedEmbedder(inner, namespace: 'other', database: db);
      await other.embed(['x']);
      expect(calls, 2, reason: 'namespaces do not share entries');
      await cached.embed(['x'], purpose: EmbedPurpose.query);
      expect(calls, 3, reason: 'purposes do not share entries');
      db.close();
    });
  });

  group('RagPipeline', () {
    late PhoenixCollection kb;
    late FakeChat chat;
    late RagPipeline rag;

    setUp(() {
      kb = PhoenixCollection.open(
        '${dir.path}/kb',
        dimensions: 256,
        sync: false,
      );
      chat = FakeChat((messages, _) {
        final prompt = messages.last.content;
        return prompt.contains('vacation')
            ? 'Employees get 25 vacation days [1]. See also [9].'
            : 'I do not know.';
      });
      rag = RagPipeline(
        store: kb.asStore(),
        embedder: HashingEmbedder(),
        chat: chat,
        // Small enough that each handbook paragraph is its own chunk.
        chunker: TextChunker(chunkSize: 80, overlap: 20),
        k: 3,
      );
    });
    tearDown(() => kb.close());

    const handbook =
        'Vacation policy. Employees get 25 vacation days per year.\n\n'
        'Expenses. Submit receipts within 30 days of purchase.\n\n'
        'Security. Lock your laptop when you leave your desk.';

    test('ingests, answers with citations and filters by metadata', () async {
      final n = await rag.ingestAll([
        const RagDocument('handbook', handbook, metadata: {'team': 'hr'}),
        const RagDocument(
          'menu',
          'The cafeteria serves soup on Mondays.',
          metadata: {'team': 'facilities'},
        ),
      ]);
      expect(n, greaterThanOrEqualTo(4));
      expect(kb.count(filter: Filter.eq('doc_id', 'handbook')), n - 1);

      final answer = await rag.ask('How many vacation days do employees get?');
      expect(answer.text, contains('[1]'));
      expect(answer.sources.first.documentId, 'handbook');
      expect(answer.sources.first.text, contains('25 vacation days'));
      expect(answer.citations.map((s) => s.number), [
        1,
      ], reason: 'the out-of-range [9] is ignored');

      final (messages, system) = chat.calls.last;
      expect(system, RagPipeline.defaultSystemPrompt);
      expect(messages.last.content, contains('<source number="1"'));
      expect(messages.last.content, endsWith('employees get?'));

      final facilities = await rag.retrieve(
        'soup',
        filter: Filter.eq('team', 'facilities'),
      );
      expect(facilities.map((s) => s.documentId).toSet(), {'menu'});
    });

    test('re-ingesting replaces chunks; remove deletes them', () async {
      final long = await rag.ingest(const RagDocument('handbook', handbook));
      final short = await rag.ingest(
        const RagDocument('handbook', 'Vacation policy. 30 days now.'),
      );
      expect(short, lessThan(long));
      expect(kb.count(filter: Filter.eq('doc_id', 'handbook')), short);
      final hits = await rag.retrieve('receipts');
      expect(hits.every((s) => !s.text.contains('receipts')), isTrue);
      expect(await rag.remove('handbook'), short);
      expect(kb.count(), 0);
      expect(
        () => rag.ingest(RagDocument('x' * 121, 'text')),
        throwsArgumentError,
      );
    });

    test('streams the answer with sources up front', () async {
      await rag.ingest(const RagDocument('handbook', handbook));
      final stream = await rag.askStream('vacation days?');
      expect(stream.sources, isNotEmpty);
      final text = await stream.text.join();
      expect(citedSources(text, stream.sources).single.number, 1);
    });

    test('a dimension mismatch is caught at construction', () {
      expect(
        () => RagPipeline(
          store: kb.asStore(),
          embedder: HashingEmbedder(dimensions: 8),
          chat: chat,
        ),
        throwsArgumentError,
      );
    });
  });

  group('SemanticCache', () {
    test('serves similar prompts, respects namespaces and TTL', () async {
      final kb = PhoenixCollection.open(
        '${dir.path}/cache',
        dimensions: 256,
        sync: false,
      );
      try {
        final cache = SemanticCache(
          store: kb.asStore(),
          embedder: HashingEmbedder(),
          threshold: 0.9,
        );
        await cache.put('What is the capital of France?', 'Paris.');
        expect(await cache.lookup('What is the capital of France?'), 'Paris.');
        expect(await cache.lookup('what is the capital of france'), 'Paris.');
        expect(await cache.lookup('How do volcanoes form?'), isNull);

        final other = SemanticCache(
          store: kb.asStore(),
          embedder: HashingEmbedder(),
          namespace: 'other-model',
        );
        expect(await other.lookup('What is the capital of France?'), isNull);

        final expired = SemanticCache(
          store: kb.asStore(),
          embedder: HashingEmbedder(),
          ttl: const Duration(milliseconds: -1),
        );
        expect(await expired.lookup('What is the capital of France?'), isNull);

        var calls = 0;
        final model = cache.wrap(
          FakeChat((_, _) {
            calls++;
            return 'Fresh answer';
          }),
        );
        final q = [const ChatMessage.user('Tell me a fact about Rust')];
        expect((await model.complete(q)).text, 'Fresh answer');
        final second = await model.complete(q);
        expect(second.text, 'Fresh answer');
        expect(second.stopReason, 'cache_hit');
        expect(calls, 1);
        expect(await model.stream(q).join(), 'Fresh answer');
        expect(calls, 1);

        expect(await cache.clear(), 2);
        expect(await cache.lookup('What is the capital of France?'), isNull);
      } finally {
        kb.close();
      }
    });
  });

  group('audited retrieval behaviour', _ragAudit);

  group('ConversationMemory', () {
    test('keeps order, recalls relevant turns and survives reopen', () async {
      final kb = await AsyncPhoenixCollection.open(
        '${dir.path}/mem',
        dimensions: 256,
        sync: false,
      );
      try {
        final memory = await ConversationMemory.open(
          store: kb,
          embedder: HashingEmbedder(),
          conversationId: 'chat-1',
        );
        await memory.addAll(const [
          ChatMessage.user('My dog is called Biscuit.'),
          ChatMessage.assistant('Biscuit is a lovely name!'),
          ChatMessage.user('I live in Lisbon.'),
          ChatMessage.assistant('Lisbon is beautiful.'),
          ChatMessage.user('I work as a baker.'),
          ChatMessage.assistant('Baking is a great craft.'),
        ]);
        expect(memory.length, 6);
        expect((await memory.recent(2)).map((m) => m.content), [
          'I work as a baker.',
          'Baking is a great craft.',
        ]);

        final context = await memory.context(
          "What is my dog's name?",
          recentCount: 2,
          recalled: 1,
        );
        expect(context.first.role, ChatRole.system);
        expect(context.first.content, contains('Biscuit'));
        expect(context.skip(1).map((m) => m.content), [
          'I work as a baker.',
          'Baking is a great craft.',
        ]);

        final reopened = await ConversationMemory.open(
          store: kb,
          embedder: HashingEmbedder(),
          conversationId: 'chat-1',
        );
        expect(reopened.length, 6);
        await reopened.add(const ChatMessage.user('Thanks!'));
        expect((await reopened.recent(1)).single.content, 'Thanks!');

        final other = await ConversationMemory.open(
          store: kb,
          embedder: HashingEmbedder(),
          conversationId: 'chat-2',
        );
        expect(other.length, 0);
        expect(await other.recall('dog'), isEmpty);

        expect(await reopened.clear(), 7);
        expect(await kb.count(), 0);
      } finally {
        await kb.close();
      }
    });
  });
}

class _CountingEmbedder implements Embedder {
  final Embedder inner;
  final void Function() onCall;
  final texts = <String>[];
  _CountingEmbedder(this.inner, this.onCall);

  @override
  int get dimensions => inner.dimensions;

  @override
  Future<List<Float32List>> embed(
    List<String> texts, {
    EmbedPurpose purpose = EmbedPurpose.document,
  }) {
    onCall();
    this.texts.addAll(texts);
    return inner.embed(texts, purpose: purpose);
  }

  @override
  void close() {}
}

// ---------------------------------------------------------------------------
// Regression tests for the audit findings.
// ---------------------------------------------------------------------------

/// An embedder that answers differently per purpose, like Voyage and E5 do.
/// Every retrieval test elsewhere uses [HashingEmbedder], which ignores
/// purpose and so cannot catch a query/document mix-up.
class AsymmetricEmbedder implements Embedder {
  final HashingEmbedder _inner;
  AsymmetricEmbedder(this._inner);

  @override
  int get dimensions => _inner.dimensions;

  @override
  Future<List<Float32List>> embed(
    List<String> texts, {
    EmbedPurpose purpose = EmbedPurpose.document,
  }) async => [for (final t in texts) _inner.embedSync('${purpose.name}: $t')];

  @override
  void close() {}
}

/// Returns fewer vectors than it was given.
class ShortEmbedder implements Embedder {
  @override
  int get dimensions => 256;

  @override
  Future<List<Float32List>> embed(
    List<String> texts, {
    EmbedPurpose purpose = EmbedPurpose.document,
  }) async => [Float32List(256)];

  @override
  void close() {}
}

void _ragAudit() {
  late Directory dir;
  setUp(() => dir = Directory.systemTemp.createTempSync('phoenix_ai2_'));
  tearDown(() {
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold files briefly.
    }
  });

  PhoenixCollection open(String name, {int dimensions = 256}) =>
      PhoenixCollection.open(
        '${dir.path}/$name',
        dimensions: dimensions,
        sync: false,
      );

  test('a repeated document id in one batch leaves one version', () async {
    final kb = open('dup');
    try {
      final rag = RagPipeline(
        store: kb.asStore(),
        embedder: HashingEmbedder(),
        chat: FakeChat((_, _) => 'x'),
        chunker: TextChunker(chunkSize: 40, overlap: 0),
      );
      // The long version would leave chunks the short one never overwrites.
      await rag.ingestAll([
        RagDocument('a', List.filled(12, 'alpha beta gamma delta').join(' ')),
        const RagDocument('a', 'only this survives'),
      ]);
      final chunks = kb.list(filter: Filter.eq('doc_id', 'a'));
      expect(chunks, hasLength(1));
      expect(chunks.single.text, 'only this survives');
    } finally {
      kb.close();
    }
  });

  test('citations tolerate anything a model might write', () {
    final sources = [
      RagSource(1, const SearchHit(id: 'a', score: 1)),
      RagSource(2, const SearchHit(id: 'b', score: 1)),
    ];
    // Out of range, wider than 64 bits, zero, and a spaced pair.
    const text = 'see [2], [9], [12345678901234567890], [0] and [1, 2]';
    expect(citedSources(text, sources).map((s) => s.number), [
      2,
      1,
    ], reason: 'first citation order, unknown numbers ignored');
  });

  test('only the sources that fitted the budget are reported', () async {
    final kb = open('budget');
    try {
      final chat = FakeChat((_, _) => 'answer [1]');
      final rag = RagPipeline(
        store: kb.asStore(),
        embedder: HashingEmbedder(),
        chat: chat,
        chunker: TextChunker(chunkSize: 400, overlap: 0),
        k: 5,
        // Room for one chunk and its wrapper, not two.
        maxContextChars: 420,
      );
      await rag.ingestAll([
        for (var i = 0; i < 4; i++)
          RagDocument('d$i', 'vacation policy ${'padding ' * 45}'),
      ]);
      final answer = await rag.ask('vacation policy');
      final prompt = chat.calls.last.$1.last.content;
      expect(answer.sources, hasLength(1), reason: 'what the model saw');
      expect(
        prompt.split('<source number=').length - 1,
        answer.sources.length,
        reason: 'the reported sources are exactly the ones in the prompt',
      );
      expect(prompt.length, lessThanOrEqualTo(600));
    } finally {
      kb.close();
    }
  });

  test('an embedder that returns the wrong count is reported', () async {
    final kb = open('short');
    try {
      final rag = RagPipeline(
        store: kb.asStore(),
        embedder: ShortEmbedder(),
        chat: FakeChat((_, _) => 'x'),
        chunker: TextChunker(chunkSize: 20, overlap: 0),
      );
      await expectLater(
        rag.ingest(const RagDocument('a', 'one two three four five six seven')),
        throwsA(isA<LlmException>()),
      );
      expect(kb.count(), 0);
    } finally {
      kb.close();
    }
  });

  test('the semantic cache hits with an asymmetric embedder', () async {
    final kb = open('cache');
    try {
      final cache = SemanticCache(
        store: kb.asStore(),
        embedder: AsymmetricEmbedder(HashingEmbedder()),
        threshold: 0.95,
      );
      await cache.put('what is the capital of France?', 'Paris.');
      expect(
        await cache.lookup('what is the capital of France?'),
        'Paris.',
        reason: 'a stored prompt must match itself whatever the purpose field',
      );
      expect(cache.hits, 1);
    } finally {
      kb.close();
    }
  });

  test('a zero-norm prompt misses instead of matching anything', () async {
    final kb = open('nan');
    try {
      final cache = SemanticCache(
        store: kb.asStore(),
        embedder: HashingEmbedder(),
      );
      await cache.put('a real question', 'a real answer');
      // No letters or digits, so the hashing embedder returns all zeros and
      // cosine similarity is NaN.
      expect(await cache.lookup('???'), isNull);
      expect(await cache.lookup('---'), isNull);
      expect(cache.misses, 2);
    } finally {
      kb.close();
    }
  });

  test('empty and interrupted answers are never cached', () async {
    final kb = open('empty');
    try {
      final cache = SemanticCache(
        store: kb.asStore(),
        embedder: HashingEmbedder(),
      );
      await cache.put('a question', '');
      expect(await cache.lookup('a question'), isNull, reason: 'not stored');

      // A stream that fails partway must not leave a truncated answer behind.
      final model = cache.wrap(
        FakeChat((_, _) => throw const LlmException('connection lost')),
      );
      final q = [const ChatMessage.user('tell me about Rust')];
      await expectLater(model.stream(q).join(), throwsA(isA<LlmException>()));
      expect(await cache.lookup('user: tell me about Rust'), isNull);
    } finally {
      kb.close();
    }
  });

  test('two memory handles on one conversation do not overwrite', () async {
    final kb = open('mem');
    try {
      final store = kb.asStore();
      final a = await ConversationMemory.open(
        store: store,
        embedder: HashingEmbedder(),
        conversationId: 'chat',
      );
      final b = await ConversationMemory.open(
        store: store,
        embedder: HashingEmbedder(),
        conversationId: 'chat',
      );
      // Interleaved appends through both handles.
      await a.add(const ChatMessage.user('one'));
      await b.add(const ChatMessage.user('two'));
      await a.add(const ChatMessage.user('three'));
      await b.add(const ChatMessage.user('four'));
      expect(await a.count(), 4, reason: 'nothing was overwritten');
      expect((await a.recent(4)).map((m) => m.content), [
        'one',
        'two',
        'three',
        'four',
      ]);

      // Deleting older messages must not make the next append collide.
      final oldest = kb.list(limit: 2).map((d) => d.id).toList();
      kb.delete(oldest);
      final c = await ConversationMemory.open(
        store: store,
        embedder: HashingEmbedder(),
        conversationId: 'chat',
      );
      await c.add(const ChatMessage.user('five'));
      expect(await c.count(), 3);
      expect((await c.recent(3)).map((m) => m.content), [
        'three',
        'four',
        'five',
      ]);
      expect(await c.clear(), 3);
      expect(await c.count(), 0);
    } finally {
      kb.close();
    }
  });

  test('a one-unit chunk window keeps surrogate pairs whole', () {
    final chunks = TextChunker(chunkSize: 1, overlap: 0).split('😀😀');
    expect(chunks.map((c) => c.text).join(), '😀😀');
    expect(
      chunks.every((c) => c.text.runes.every((r) => r == 0x1F600)),
      isTrue,
    );
  });

  test('cached vectors are copies, not the cache itself', () async {
    final cached = CachedEmbedder(
      HashingEmbedder(dimensions: 8),
      namespace: 'copies',
    );
    final first = await cached.embed(['text']);
    first.single[0] = 42; // a caller normalising in place
    final second = await cached.embed(['text']);
    expect(second.single[0], isNot(42), reason: 'the cache was not corrupted');
  });
}
