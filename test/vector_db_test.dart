/// Tests for the embedded vector-search API.
///
/// These require the native library; run `./build.sh` (or `build.ps1`) first.
/// When it is missing the whole suite is skipped rather than failing, matching
/// the convention in `phoenixdb_test.dart`.
library;

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

/// Deterministic pseudo-random vectors, so a failure reproduces exactly.
Float32List randomVector(int dim, int seed) {
  final rng = Random(seed);
  final out = Float32List(dim);
  for (var i = 0; i < dim; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

/// A unit vector pointing along [axis], for exactly-checkable geometry.
Float32List basisVector(int dim, int axis) {
  final out = Float32List(dim);
  out[axis] = 1.0;
  return out;
}

void main() {
  late Directory dir;
  String? skipReason;
  var counter = 0;

  setUpAll(() {
    try {
      final probe = Directory.systemTemp.createTempSync('phoenix_vec_probe');
      final p = PhoenixVectorDB.open(
        '${probe.path}/probe.pvec',
        dimensions: 4,
      );
      p.close();
      probe.deleteSync(recursive: true);
    } on PhoenixLoadException catch (e) {
      skipReason = 'native library unavailable: ${e.message}';
    }
  });

  setUp(() {
    if (skipReason != null) return;
    dir = Directory.systemTemp.createTempSync('phoenixdb_vector_test');
  });

  tearDown(() {
    if (skipReason != null) return;
    if (dir.existsSync()) {
      try {
        dir.deleteSync(recursive: true);
      } on FileSystemException {
        // Windows keeps a mapped file locked briefly after close; a leaked
        // temp directory must not fail an otherwise passing test.
      }
    }
  });

  /// Opens a fresh index inside the per-test temp directory.
  PhoenixVectorDB openIndex({
    int dimensions = 4,
    VectorMetric metric = VectorMetric.cosine,
  }) => PhoenixVectorDB.open(
    '${dir.path}/index${counter++}.pvec',
    dimensions: dimensions,
    metric: metric,
  );

  group('lifecycle', () {
    test('open reports its geometry', () {
      final db = openIndex(dimensions: 16, metric: VectorMetric.euclidean);
      addTearDown(db.close);
      expect(db.dimensions, 16);
      expect(db.metric, VectorMetric.euclidean);
      expect(db.isClosed, isFalse);
      expect(db.count(), 0);
    }, skip: skipReason);

    test('close is idempotent', () {
      final db = openIndex();
      db.close();
      expect(db.isClosed, isTrue);
      // A second close must be a no-op, not a double free.
      db.close();
      expect(db.isClosed, isTrue);
    }, skip: skipReason);

    test('operations after close are rejected', () {
      final db = openIndex();
      db.close();
      expect(
        () => db.insert('a', basisVector(4, 0)),
        throwsA(isA<PhoenixException>()),
      );
      expect(() => db.count(), throwsA(isA<PhoenixException>()));
    }, skip: skipReason);

    test('a non-positive dimension is rejected before any native call', () {
      expect(
        () => PhoenixVectorDB.open('${dir.path}/bad.pvec', dimensions: 0),
        throwsA(isA<ArgumentError>()),
      );
    }, skip: skipReason);

    test('native limits are reported', () {
      final db = openIndex();
      addTearDown(db.close);
      expect(db.maxDimensions, 65536);
      expect(db.maxK, 4096);
      expect(db.maxIdLength, 128);
      expect(db.kernel, isIn(['avx2+fma', 'neon', 'portable']));
    }, skip: skipReason);
  });

  group('multi-vector insert and retrieval', () {
    test('inserts are counted and retrievable', () {
      final db = openIndex(dimensions: 8);
      addTearDown(db.close);

      final vectors = <String, Float32List>{
        for (var i = 0; i < 25; i++) 'doc-$i': randomVector(8, i),
      };
      db.insertAll(vectors);

      expect(db.count(), 25);
      for (final entry in vectors.entries) {
        expect(db.contains(entry.key), isTrue, reason: entry.key);
        final stored = db.get(entry.key);
        expect(stored, isNotNull);
        // f32 storage round-trips a Dart double exactly, because Float32List
        // already holds single precision.
        expect(stored, equals(entry.value), reason: entry.key);
      }
    }, skip: skipReason);

    test('missing ids read as null and report absence', () {
      final db = openIndex();
      addTearDown(db.close);
      expect(db.get('nope'), isNull);
      expect(db.contains('nope'), isFalse);
      expect(db.remove('nope'), isFalse);
    }, skip: skipReason);

    test('re-inserting an id replaces it without duplicating', () {
      final db = openIndex(dimensions: 3, metric: VectorMetric.euclidean);
      addTearDown(db.close);

      db.insert('k', Float32List.fromList([0, 0, 0]));
      db.insert('k', Float32List.fromList([9, 9, 9]));

      expect(db.count(), 1);
      expect(db.get('k'), equals(Float32List.fromList([9, 9, 9])));
      final hits = db.searchVector(Float32List.fromList([9, 9, 9]), k: 10);
      expect(hits, hasLength(1));
      expect(hits.single.id, 'k');
    }, skip: skipReason);

    test('a wrong-width vector is rejected at the call site', () {
      final db = openIndex(dimensions: 4);
      addTearDown(db.close);
      expect(
        () => db.insert('short', Float32List.fromList([1, 2])),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => db.insert('long', Float32List(9)),
        throwsA(isA<ArgumentError>()),
      );
      expect(db.count(), 0);
    }, skip: skipReason);

    test('a rejected batch leaves the index untouched', () {
      final db = openIndex(dimensions: 4);
      addTearDown(db.close);
      expect(
        () => db.insertAll({
          'good': Float32List(4),
          'bad': Float32List(2),
        }),
        throwsA(isA<ArgumentError>()),
      );
      // Validation runs over the whole batch first, so nothing was written.
      expect(db.count(), 0);
    }, skip: skipReason);

    test('a non-finite component is refused by the engine', () {
      final db = openIndex(dimensions: 3, metric: VectorMetric.euclidean);
      addTearDown(db.close);
      expect(
        () => db.insert('nan', Float32List.fromList([1, double.nan, 3])),
        throwsA(isA<PhoenixException>()),
      );
      expect(db.count(), 0);
    }, skip: skipReason);
  });

  group('similarity retrieval and scoring', () {
    test('cosine ranks the exact match first with score 1', () {
      final db = openIndex(dimensions: 4);
      addTearDown(db.close);
      db.insert('x', basisVector(4, 0));
      db.insert('y', basisVector(4, 1));
      db.insert('z', basisVector(4, 2));

      final hits = db.searchVector(basisVector(4, 0), k: 3);
      expect(hits, hasLength(3));
      expect(hits.first.id, 'x');
      expect(hits.first.distance, closeTo(0.0, 1e-5));
      expect(hits.first.score, closeTo(1.0, 1e-5));
      // The other two are orthogonal: cosine distance 1, similarity 0.
      for (final hit in hits.skip(1)) {
        expect(hit.distance, closeTo(1.0, 1e-5));
        expect(hit.score, closeTo(0.0, 1e-5));
      }
    }, skip: skipReason);

    test('cosine is scale-invariant', () {
      final db = openIndex(dimensions: 3);
      addTearDown(db.close);
      db.insert('unit', Float32List.fromList([1, 0, 0]));
      db.insert('big', Float32List.fromList([250, 0, 0]));

      final hits = db.searchVector(Float32List.fromList([0.004, 0, 0]), k: 2);
      expect(hits, hasLength(2));
      for (final hit in hits) {
        expect(hit.distance, closeTo(0.0, 1e-4), reason: hit.id);
        expect(hit.score, closeTo(1.0, 1e-4), reason: hit.id);
      }
    }, skip: skipReason);

    test('cosine distance at 90 and 180 degrees', () {
      final db = openIndex(dimensions: 2);
      addTearDown(db.close);
      db.insert('same', Float32List.fromList([1, 0]));
      db.insert('perp', Float32List.fromList([0, 1]));
      db.insert('opposite', Float32List.fromList([-1, 0]));

      final byId = {
        for (final hit in db.searchVector(Float32List.fromList([1, 0]), k: 3))
          hit.id: hit,
      };
      expect(byId['same']!.distance, closeTo(0.0, 1e-5));
      expect(byId['perp']!.distance, closeTo(1.0, 1e-5));
      expect(byId['opposite']!.distance, closeTo(2.0, 1e-5));
    }, skip: skipReason);

    test('euclidean reports the true distance, not the square', () {
      final db = openIndex(dimensions: 2, metric: VectorMetric.euclidean);
      addTearDown(db.close);
      db.insert('origin', Float32List.fromList([0, 0]));

      // 3-4-5 triangle: the engine orders by 25 but must report 5.
      final hit = db.searchVector(Float32List.fromList([3, 4]), k: 1).single;
      expect(hit.distance, closeTo(5.0, 1e-4));
      expect(hit.score, closeTo(1 / 6, 1e-4));
    }, skip: skipReason);

    test('dot product ranks by magnitude, unlike cosine', () {
      final db = openIndex(dimensions: 2, metric: VectorMetric.dotProduct);
      addTearDown(db.close);
      db.insert('small', Float32List.fromList([1, 1]));
      db.insert('large', Float32List.fromList([5, 5]));

      final hits = db.searchVector(Float32List.fromList([1, 1]), k: 2);
      expect(hits.first.id, 'large');
      expect(hits.first.score, closeTo(10.0, 1e-4));
      expect(hits.last.score, closeTo(2.0, 1e-4));
    }, skip: skipReason);

    test('results are ordered nearest first, with score mirroring rank', () {
      final db = openIndex(dimensions: 12);
      addTearDown(db.close);
      for (var i = 0; i < 60; i++) {
        db.insert('v$i', randomVector(12, i * 7 + 1));
      }

      final hits = db.searchVector(randomVector(12, 999), k: 10);
      expect(hits, hasLength(10));
      for (var i = 1; i < hits.length; i++) {
        expect(
          hits[i].distance,
          greaterThanOrEqualTo(hits[i - 1].distance),
          reason: 'distances must ascend',
        );
        expect(
          hits[i].score,
          lessThanOrEqualTo(hits[i - 1].score),
          reason: 'scores must descend',
        );
      }
    }, skip: skipReason);

    test('every stored vector retrieves itself as its own neighbour', () {
      // The strongest end-to-end correctness signal: self-retrieval must be
      // exact for every point, not merely likely.
      final db = openIndex(dimensions: 32, metric: VectorMetric.euclidean);
      addTearDown(db.close);
      final vectors = <String, Float32List>{
        for (var i = 0; i < 120; i++) 'v$i': randomVector(32, i + 500),
      };
      db.insertAll(vectors);

      for (final entry in vectors.entries) {
        final hit = db.search(VectorQuery(entry.value, k: 1)).single;
        expect(hit.id, entry.key);
        expect(hit.distance, closeTo(0.0, 1e-3));
      }
    }, skip: skipReason);

    test('k larger than the collection returns everything live', () {
      final db = openIndex(dimensions: 4);
      addTearDown(db.close);
      db.insert('a', basisVector(4, 0));
      db.insert('b', basisVector(4, 1));
      expect(db.searchVector(basisVector(4, 0), k: 50), hasLength(2));
    }, skip: skipReason);

    test('searching an empty index returns no matches', () {
      final db = openIndex();
      addTearDown(db.close);
      expect(db.searchVector(basisVector(4, 0), k: 5), isEmpty);
    }, skip: skipReason);

    test('invalid k and query width are rejected', () {
      final db = openIndex(dimensions: 4);
      addTearDown(db.close);
      db.insert('a', basisVector(4, 0));
      expect(
        () => db.searchVector(basisVector(4, 0), k: 0),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => db.searchVector(basisVector(4, 0), k: db.maxK + 1),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => db.searchVector(Float32List(3), k: 1),
        throwsA(isA<ArgumentError>()),
      );
    }, skip: skipReason);

    test('a higher ef still returns a correct ranking', () {
      final db = openIndex(dimensions: 16, metric: VectorMetric.euclidean);
      addTearDown(db.close);
      for (var i = 0; i < 80; i++) {
        db.insert('v$i', randomVector(16, i + 4000));
      }
      final query = randomVector(16, 4000); // identical to v0
      final wide = db.search(VectorQuery(query, k: 5, efSearch: 200));
      final narrow = db.search(VectorQuery(query, k: 5, efSearch: 16));
      expect(wide.first.id, 'v0');
      expect(narrow.first.id, 'v0');
    }, skip: skipReason);
  });

  group('removal and compaction', () {
    test('removed vectors disappear from results', () {
      final db = openIndex(dimensions: 4, metric: VectorMetric.euclidean);
      addTearDown(db.close);
      db.insert('gone', Float32List.fromList([1, 1, 1, 1]));
      db.insert('stays', Float32List.fromList([2, 2, 2, 2]));

      expect(db.remove('gone'), isTrue);
      expect(db.remove('gone'), isFalse, reason: 'already removed');
      expect(db.count(), 1);
      expect(db.get('gone'), isNull);

      final hits = db.searchVector(Float32List.fromList([1, 1, 1, 1]), k: 5);
      expect(hits.map((h) => h.id), isNot(contains('gone')));
      expect(hits.map((h) => h.id), contains('stays'));
    }, skip: skipReason);

    test('stats track tombstones and compact reclaims them', () {
      final db = openIndex(dimensions: 4, metric: VectorMetric.euclidean);
      addTearDown(db.close);
      for (var i = 0; i < 20; i++) {
        db.insert('v$i', Float32List.fromList([i.toDouble(), 0, 0, 0]));
      }
      for (var i = 0; i < 8; i++) {
        db.remove('v$i');
      }

      var stats = db.stats();
      expect(stats.live, 12);
      expect(stats.total, 20);
      expect(stats.deleted, 8);
      expect(stats.deletedRatio, closeTo(0.4, 1e-9));

      expect(db.compact(), 8);
      stats = db.stats();
      expect(stats.live, 12);
      expect(stats.total, 12);
      expect(stats.deleted, 0);
      expect(stats.deletedRatio, 0);

      // Search must stay correct after ordinals are renumbered.
      final hit = db
          .searchVector(Float32List.fromList([15, 0, 0, 0]), k: 1)
          .single;
      expect(hit.id, 'v15');
      expect(db.contains('v3'), isFalse);
      expect(db.compact(), 0, reason: 'nothing left to reclaim');
    }, skip: skipReason);
  });

  group('persistence', () {
    test('an index reopens with its vectors and rankings intact', () {
      final path = '${dir.path}/persist.pvec';
      final query = randomVector(16, 12345);
      late final List<String> before;

      final first = PhoenixVectorDB.open(path, dimensions: 16);
      for (var i = 0; i < 40; i++) {
        first.insert('v$i', randomVector(16, i + 900));
      }
      before = first.searchVector(query, k: 5).map((h) => h.id).toList();
      first.save();
      first.close();

      expect(File('$path.hnsw').existsSync(), isTrue, reason: 'snapshot');

      final second = PhoenixVectorDB.open(path, dimensions: 16);
      addTearDown(second.close);
      expect(second.count(), 40);
      expect(second.searchVector(query, k: 5).map((h) => h.id), before);
    }, skip: skipReason);

    test('a corrupt snapshot is rebuilt from the vectors', () {
      final path = '${dir.path}/rebuild.pvec';
      final first = PhoenixVectorDB.open(
        path,
        dimensions: 4,
        metric: VectorMetric.euclidean,
      );
      for (var i = 0; i < 15; i++) {
        first.insert('v$i', Float32List.fromList([i.toDouble(), 1, 2, 3]));
      }
      first.save();
      first.close();

      // The vectors are the source of truth, so a damaged graph snapshot must
      // be recoverable rather than fatal.
      File('$path.hnsw').writeAsStringSync('not a snapshot');

      final second = PhoenixVectorDB.open(
        path,
        dimensions: 4,
        metric: VectorMetric.euclidean,
      );
      addTearDown(second.close);
      expect(second.count(), 15);
      expect(
        second
            .searchVector(Float32List.fromList([9, 1, 2, 3]), k: 1)
            .single
            .id,
        'v9',
      );
    }, skip: skipReason);

    test('reopening with the wrong geometry is refused', () {
      final path = '${dir.path}/geometry.pvec';
      final first = PhoenixVectorDB.open(path, dimensions: 8);
      first.insert('a', Float32List(8));
      first.save();
      first.close();

      // Reinterpreting 8-float records as 16-float ones, or reordering by a
      // different metric, would produce plausible nonsense.
      expect(
        () => PhoenixVectorDB.open(path, dimensions: 16),
        throwsA(isA<PhoenixException>()),
      );
      expect(
        () => PhoenixVectorDB.open(
          path,
          dimensions: 8,
          metric: VectorMetric.euclidean,
        ),
        throwsA(isA<PhoenixException>()),
      );
      final reopened = PhoenixVectorDB.open(path, dimensions: 8);
      addTearDown(reopened.close);
      expect(reopened.count(), 1);
    }, skip: skipReason);

    test('tombstones survive a reopen', () {
      final path = '${dir.path}/tombstone.pvec';
      final first = PhoenixVectorDB.open(
        path,
        dimensions: 2,
        metric: VectorMetric.euclidean,
      );
      first.insert('keep', Float32List.fromList([1, 1]));
      first.insert('drop', Float32List.fromList([2, 2]));
      first.remove('drop');
      first.save();
      first.close();

      final second = PhoenixVectorDB.open(
        path,
        dimensions: 2,
        metric: VectorMetric.euclidean,
      );
      addTearDown(second.close);
      expect(second.count(), 1);
      expect(second.contains('keep'), isTrue);
      expect(second.contains('drop'), isFalse);
    }, skip: skipReason);
  });

  group('async isolate API', () {
    test('insert and search run off the calling isolate', () async {
      final db = await AsyncPhoenixVectorDB.open(
        '${dir.path}/async.pvec',
        dimensions: 4,
      );
      addTearDown(db.close);

      await db.insert('x', basisVector(4, 0));
      await db.insert('y', basisVector(4, 1));
      expect(await db.count(), 2);

      final hits = await db.search(VectorQuery(basisVector(4, 0), k: 2));
      expect(hits.first.id, 'x');
      expect(hits.first.distance, closeTo(0.0, 1e-5));
      expect(hits.first.score, closeTo(1.0, 1e-5));
    }, skip: skipReason);

    test('batch insert, stats, compaction and save', () async {
      final db = await AsyncPhoenixVectorDB.open(
        '${dir.path}/async_batch.pvec',
        dimensions: 8,
        metric: VectorMetric.euclidean,
      );
      addTearDown(db.close);

      await db.insertAll({
        for (var i = 0; i < 30; i++) 'v$i': randomVector(8, i + 77),
      });
      expect(await db.count(), 30);

      expect(await db.remove('v0'), isTrue);
      expect(await db.contains('v0'), isFalse);
      final stats = await db.stats();
      expect(stats.live, 29);
      expect(stats.deleted, 1);

      expect(await db.compact(), 1);
      await db.flush();
      await db.save();
      expect(await db.count(), 29);
    }, skip: skipReason);

    test('many concurrent searches all resolve correctly', () async {
      // The point of the worker isolate: overlapping in-flight requests must
      // each get their own answer back, in any completion order.
      final db = await AsyncPhoenixVectorDB.open(
        '${dir.path}/async_concurrent.pvec',
        dimensions: 16,
        metric: VectorMetric.euclidean,
      );
      addTearDown(db.close);

      final vectors = <String, Float32List>{
        for (var i = 0; i < 50; i++) 'v$i': randomVector(16, i + 2000),
      };
      await db.insertAll(vectors);

      final results = await Future.wait([
        for (final entry in vectors.entries)
          db.search(VectorQuery(entry.value, k: 1)),
      ]);
      final ids = vectors.keys.toList();
      for (var i = 0; i < results.length; i++) {
        expect(results[i].single.id, ids[i], reason: 'request $i');
      }
    }, skip: skipReason);

    test('a wrong-width vector is rejected without an isolate round trip',
        () async {
      final db = await AsyncPhoenixVectorDB.open(
        '${dir.path}/async_bad.pvec',
        dimensions: 4,
      );
      addTearDown(db.close);
      expect(
        () => db.insert('bad', Float32List(2)),
        throwsA(isA<ArgumentError>()),
      );
      await expectLater(
        db.search(VectorQuery(Float32List(9), k: 1)),
        throwsA(isA<ArgumentError>()),
      );
    }, skip: skipReason);

    test('calls after close fail rather than hang', () async {
      final db = await AsyncPhoenixVectorDB.open(
        '${dir.path}/async_closed.pvec',
        dimensions: 4,
      );
      await db.insert('a', basisVector(4, 0));
      await db.close();
      expect(db.isClosed, isTrue);
      await db.close(); // idempotent
      await expectLater(db.count(), throwsA(isA<PhoenixException>()));
    }, skip: skipReason);

    test('data written asynchronously is readable synchronously', () async {
      final path = '${dir.path}/async_handoff.pvec';
      final async = await AsyncPhoenixVectorDB.open(path, dimensions: 4);
      await async.insert('shared', Float32List.fromList([1, 0, 0, 0]));
      await async.save();
      await async.close();

      final sync = PhoenixVectorDB.open(path, dimensions: 4);
      addTearDown(sync.close);
      expect(sync.count(), 1);
      expect(sync.get('shared'), equals(Float32List.fromList([1, 0, 0, 0])));
    }, skip: skipReason);
  });

  group('value semantics', () {
    test('VectorMatch compares and sorts by distance', () {
      const near = VectorMatch(id: 'a', distance: 0.1, score: 0.9);
      const far = VectorMatch(id: 'b', distance: 0.8, score: 0.2);
      const alsoNear = VectorMatch(id: 'a', distance: 0.1, score: 0.9);

      expect(near, equals(alsoNear));
      expect(near.hashCode, alsoNear.hashCode);
      expect(near.compareTo(far), lessThan(0));

      final sorted = [far, near]..sort();
      expect(sorted.first.id, 'a');
    });

    test('VectorQuery.fromList copies into a Float32List', () {
      final query = VectorQuery.fromList([1.0, 2.0, 3.0], k: 3);
      expect(query.vector, isA<Float32List>());
      expect(query.dimensions, 3);
      expect(query.k, 3);
      expect(query.efSearch, isNull);
    });

    test('VectorStats.deletedRatio handles an empty index', () {
      const empty = VectorStats(live: 0, total: 0, deleted: 0);
      expect(empty.deletedRatio, 0);
      const half = VectorStats(live: 5, total: 10, deleted: 5);
      expect(half.deletedRatio, 0.5);
    });

    test('metric codes match the native wire values', () {
      // These are the values passed to phoenix_vector_init and must never be
      // reordered.
      expect(VectorMetric.cosine.code, 0);
      expect(VectorMetric.euclidean.code, 1);
      expect(VectorMetric.dotProduct.code, 2);
    });
  });
}
