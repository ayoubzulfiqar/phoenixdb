/// Document collections through the full Dart -> FFI -> Rust path: CRUD,
/// the filter DSL, hybrid search, persistence and the async worker.
library;

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

Float32List vec(List<double> v) => Float32List.fromList(v);

void main() {
  late Directory dir;
  late String path;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('phoenix_coll_');
    path = '${dir.path}/kb';
  });

  tearDown(() {
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold files briefly.
    }
  });

  List<Document> corpus() => [
    Document(
      'rust',
      text: 'Rust is a systems programming language focused on safety',
      metadata: {
        'topic': 'lang',
        'year': 2015,
        'tags': ['systems', 'safe'],
      },
      vector: vec([1, 0, 0]),
    ),
    Document(
      'dart',
      text: 'Dart compiles to native code and powers Flutter apps',
      metadata: {
        'topic': 'lang',
        'year': 2011,
        'tags': ['ui'],
        'author': {'org': 'google'},
      },
      vector: vec([0.9, 0.1, 0]),
    ),
    Document(
      'hnsw',
      text: 'HNSW graphs make approximate nearest neighbour search fast',
      metadata: {'topic': 'search', 'year': 2016},
      vector: vec([0, 1, 0]),
    ),
    Document(
      'bm25',
      text: 'BM25 ranks documents by keyword relevance for search engines',
      metadata: {'topic': 'search', 'year': 1994, 'draft': true},
      vector: vec([0, 0.8, 0.2]),
    ),
  ];

  test('documents round-trip and the filter DSL matches the engine', () {
    final kb = PhoenixCollection.open(path, dimensions: 3, sync: false);
    try {
      kb.upsert(corpus());
      expect(kb.count(), 4);

      final dart = kb.get('dart', withVector: true)!;
      expect(dart.text, startsWith('Dart compiles'));
      expect(dart.metadata!['author'], {'org': 'google'});
      expect(dart.vector, vec([0.9, 0.1, 0]));
      expect(kb.get('dart')!.vector, isNull);
      expect(kb.get('nope'), isNull);

      int n(Filter f) => kb.count(filter: f);
      expect(n(Filter.eq('topic', 'lang')), 2);
      expect(n(Filter.eq('tags', 'ui')), 1, reason: 'array contains');
      expect(n(Filter.eq('author.org', 'google')), 1, reason: 'dot path');
      expect(n(Filter.gte('year', 2012)), 2);
      expect(n(Filter.between('year', 2011, 2015)), 2);
      expect(n(Filter.inList('topic', ['search', 'x'])), 2);
      expect(n(Filter.notIn('topic', ['search'])), 2);
      expect(n(Filter.ne('draft', true)), 3, reason: r'$ne matches absent');
      expect(n(Filter.exists('author')), 1);
      expect(n(Filter.exists('draft', false)), 3);
      expect(n(Filter.eq('topic', 'lang') & Filter.lt('year', 2012)), 1);
      expect(n(Filter.eq('topic', 'lang') | Filter.eq('draft', true)), 3);
      expect(n(~Filter.eq('topic', 'lang')), 2);
      expect(
        n(
          const Filter.raw({
            'year': {r'$gt': 2000},
            'topic': 'search',
          }),
        ),
        1,
      );
      expect(
        () => n(const Filter.raw({r'$xor': []})),
        throwsA(isA<PhoenixException>()),
      );

      expect(kb.list(filter: Filter.eq('topic', 'search')).map((d) => d.id), [
        'bm25',
        'hnsw',
      ], reason: 'id order');
      expect(kb.list(limit: 1, after: 'bm25').single.id, 'dart');

      expect(kb.delete(['rust', 'missing']), 1);
      expect(kb.count(), 3);
      expect(kb.stats().vectors, 3);
    } finally {
      kb.close();
    }
  });

  test(
    'hybrid search fuses vectors and keywords, filtered and diversified',
    () {
      final kb = PhoenixCollection.open(path, dimensions: 3, sync: false);
      try {
        kb.upsert(corpus());

        final keyword = kb.search(text: 'search engines', k: 2);
        expect(keyword.first.id, 'bm25');
        expect(keyword.first.textScore, isNotNull);
        expect(keyword.first.vectorScore, isNull);

        final semantic = kb.search(vector: vec([1, 0.05, 0]), k: 2);
        expect(semantic.map((h) => h.id), ['rust', 'dart']);
        expect(semantic.first.distance, lessThan(semantic.last.distance!));

        final hybrid = kb.search(
          vector: vec([0.1, 1, 0]),
          text: 'keyword relevance',
          k: 2,
        );
        expect(hybrid.map((h) => h.id), containsAll(['hnsw', 'bm25']));

        final filtered = kb.search(
          vector: vec([1, 0, 0]),
          filter: Filter.eq('topic', 'search'),
          k: 5,
        );
        expect(filtered.map((h) => h.id).toSet(), {'hnsw', 'bm25'});

        final lean = kb.search(
          text: 'rust',
          includeText: false,
          includeMetadata: false,
          includeVector: true,
        );
        expect(lean.single.text, isNull);
        expect(lean.single.metadata, isNull);
        expect(lean.single.vector, vec([1, 0, 0]));

        // alpha = 1: pure vector ranking; MMR then prefers diversity.
        final plain = kb.search(
          vector: vec([1, 0, 0]),
          text: 'language',
          fusion: const Fusion.weighted(1),
          k: 2,
        );
        expect(plain.map((h) => h.id), ['rust', 'dart']);
        final diverse = kb.search(
          vector: vec([1, 0, 0]),
          text: 'language',
          fusion: const Fusion.weighted(1),
          mmr: 0.3,
          k: 2,
        );
        expect(diverse.first.id, 'rust');
        expect(diverse.last.id, isNot('dart'), reason: 'dart ≈ rust');

        expect(kb.search(k: 10).length, 4, reason: 'no query lists all');
        expect(kb.search(text: 'rust', minScore: 1e9), isEmpty);
      } finally {
        kb.close();
      }
    },
  );

  test('arguments are validated before the native call', () {
    final kb = PhoenixCollection.open(path, dimensions: 3, sync: false);
    try {
      expect(
        () => kb.upsert([
          Document('x', vector: vec([1, 2])),
        ]),
        throwsArgumentError,
      );
      expect(() => kb.search(vector: vec([1])), throwsArgumentError);
      expect(
        () => kb.upsert([
          Document('x', metadata: {'bad': Object()}),
        ]),
        throwsArgumentError,
      );
      expect(() => kb.upsert([Document('')]), throwsA(isA<PhoenixException>()));
      expect(
        () => kb.upsert([
          Document('x', metadata: {'a.b': 1}),
        ]),
        throwsA(isA<PhoenixException>()),
      );
      expect(kb.count(), 0, reason: 'failed upserts wrote nothing');
    } finally {
      kb.close();
    }
    expect(kb.count, throwsA(isA<PhoenixException>()));
    kb.close(); // idempotent
  });

  test('collections persist, adopt their layout and share opens', () {
    final kb = PhoenixCollection.open(path, dimensions: 3);
    kb.upsert(corpus());
    final twin = PhoenixCollection.open(path);
    expect(twin.dimensions, 3, reason: 'dimensions 0 adopts the layout');
    expect(twin.count(), 4, reason: 'a second open shares the engine');
    twin.close();
    kb.close();

    final reopened = PhoenixCollection.open(path);
    try {
      expect(reopened.stats().documents, 4);
      expect(reopened.stats().repaired, 0);
      expect(reopened.search(text: 'flutter').single.id, 'dart');
      expect(
        () => PhoenixCollection.open(path, dimensions: 8),
        throwsA(isA<PhoenixException>()),
      );
    } finally {
      reopened.close();
    }
  });

  test('bulk ingest pages lazily', () {
    final kb = PhoenixCollection.open(path, dimensions: 8, sync: false);
    final rng = Random(7);
    try {
      kb.upsert([
        for (var i = 0; i < 600; i++)
          Document(
            'd${i.toString().padLeft(4, '0')}',
            text: 'document number $i',
            metadata: {'bucket': i % 3},
            vector: Float32List.fromList([
              for (var j = 0; j < 8; j++) rng.nextDouble(),
            ]),
          ),
      ]);
      final ids = kb
          .documents(filter: Filter.eq('bucket', 1), pageSize: 64)
          .map((d) => d.id)
          .toList();
      expect(ids.length, 200);
      expect(ids.first, 'd0001');
      expect(ids, orderedEquals([...ids]..sort()));
      final q = Float32List.fromList([for (var j = 0; j < 8; j++) 0.5]);
      final hits = kb.search(vector: q, filter: Filter.eq('bucket', 2), k: 10);
      expect(hits.length, 10);
      expect(hits.every((h) => h.metadata!['bucket'] == 2), isTrue);
    } finally {
      kb.close();
    }
  });

  test('async collection has parity with the sync client', () async {
    final kb = await AsyncPhoenixCollection.open(
      path,
      dimensions: 3,
      sync: false,
    );
    try {
      await kb.upsert(corpus());
      expect(await kb.count(filter: Filter.eq('topic', 'lang')), 2);
      final hits = await kb.search(
        vector: vec([0, 1, 0]),
        text: 'nearest neighbour',
        k: 1,
      );
      expect(hits.single.id, 'hnsw');
      expect((await kb.get('bm25'))!.metadata!['draft'], isTrue);
      expect(await kb.documents(pageSize: 3).map((d) => d.id).toList(), [
        'bm25',
        'dart',
        'hnsw',
        'rust',
      ]);
      expect(await kb.delete(['bm25']), 1);
      expect((await kb.stats()).documents, 3);
      expect(
        () => kb.upsert([
          Document('x', vector: vec([1])),
        ]),
        throwsArgumentError,
      );
      await kb.flush();
    } finally {
      await kb.close();
    }
    expect(kb.isClosed, isTrue);

    final adopted = await AsyncPhoenixCollection.open(path);
    try {
      expect(adopted.dimensions, 3);
    } finally {
      await adopted.close();
    }
    await expectLater(
      AsyncPhoenixCollection.open(path, dimensions: 5),
      throwsA(isA<PhoenixException>()),
    );
  });
}
