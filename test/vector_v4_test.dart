/// Tests for the 4.0 vector surface: batch insert, subset (filtered) search,
/// batch search, shared engines, and the supervised async worker.
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
    dir = Directory.systemTemp.createTempSync('phoenix_vec4_');
    path = '${dir.path}/v.pvec';
  });

  tearDown(() {
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold files briefly.
    }
  });

  test('insertAll is one atomic batch', () {
    final db = PhoenixVectorDB.open(path, dimensions: 2);
    try {
      db.insertAll({
        for (var i = 0; i < 100; i++) 'p$i': vec([i.toDouble(), 1]),
      });
      expect(db.count(), 100);
      expect(
        () => db.insertAll({
          'ok': vec([1, 2]),
          'bad': vec([double.nan, 1]),
        }),
        throwsA(isA<PhoenixException>()),
      );
      expect(db.contains('ok'), isFalse, reason: 'nothing from a bad batch');
      db.insertAll({});
    } finally {
      db.close();
    }
  });

  test('searchIds is exact over the subset and ignores unknown ids', () {
    final db = PhoenixVectorDB.open(
      path,
      dimensions: 2,
      metric: VectorMetric.euclidean,
    );
    try {
      db.insertAll({
        for (var i = 0; i < 1000; i++) 'p$i': vec([i.toDouble(), 0]),
      });
      final hits = db.searchIds(vec([500, 0]), [
        'p3',
        'p900',
        'p510',
        'ghost',
      ], k: 5);
      expect(hits.map((m) => m.id), ['p510', 'p900', 'p3']);
      expect(db.searchIds(vec([0, 0]), const []), isEmpty);
    } finally {
      db.close();
    }
  });

  test('searchBatch matches individual searches', () {
    final db = PhoenixVectorDB.open(path, dimensions: 8);
    final rng = Random(7);
    Float32List random() =>
        Float32List.fromList(List.generate(8, (_) => rng.nextDouble()));
    try {
      final points = {for (var i = 0; i < 700; i++) 'p$i': random()};
      db.insertAll(points);
      final queries = [random(), random(), random()];
      final batch = db.searchBatch(queries, k: 4, ef: 128);
      expect(batch.length, 3);
      for (var i = 0; i < queries.length; i++) {
        final single = db.searchVector(queries[i], k: 4, ef: 128);
        expect(batch[i].map((m) => m.id), single.map((m) => m.id));
      }
      expect(db.searchBatch(const []), isEmpty);
    } finally {
      db.close();
    }
  });

  test('two handles on one index share the engine', () {
    final a = PhoenixVectorDB.open(path, dimensions: 3);
    final b = PhoenixVectorDB.open(path, dimensions: 3);
    try {
      a.insert('x', vec([1, 0, 0]));
      expect(b.contains('x'), isTrue);
      expect(
        () => PhoenixVectorDB.open(path, dimensions: 4),
        throwsA(isA<PhoenixException>()),
        reason: 'a different geometry is refused, not reinterpreted',
      );
    } finally {
      a.close();
      expect(b.contains('x'), isTrue, reason: 'b survives a closing');
      b.close();
    }
  });

  test('async client: batch ops, subset search, idempotent close', () async {
    final db = await AsyncPhoenixVectorDB.open(
      path,
      dimensions: 2,
      metric: VectorMetric.euclidean,
    );
    await db.insertAll({
      for (var i = 0; i < 50; i++) 'p$i': vec([i.toDouble(), 1]),
    });
    expect(await db.count(), 50);
    final subset = await db.searchIds(vec([10, 1]), ['p9', 'p30'], k: 2);
    expect(subset.first.id, 'p9');
    final batch = await db.searchBatch([
      vec([0, 1]),
      vec([49, 1]),
    ], k: 1);
    expect(batch.map((r) => r.first.id), ['p0', 'p49']);
    await db.close();
    expect(db.isClosed, isTrue);
    await db.close();
    expect(db.count(), throwsA(isA<PhoenixException>()));
  });

  test('async client reports a failed open instead of hanging', () async {
    await expectLater(
      AsyncPhoenixVectorDB.open(
        '${dir.path}/missing-dir-is-fine/v.pvec',
        dimensions: 70000, // over the 65 536 limit
      ),
      throwsA(isA<PhoenixException>()),
    );
  });
}
