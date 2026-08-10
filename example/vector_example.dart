/// Embedded vector search: index a few documents and query them by meaning.
///
/// Run with:
/// ```sh
/// dart run example/vector_example.dart
/// ```
///
/// The "embeddings" here are hand-written so the example is deterministic and
/// dependency-free. In a real app they would come from a model such as
/// all-MiniLM-L6-v2 (384 dimensions) or text-embedding-3-small (1536).
library;

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';

/// Toy 8-dimensional "embedding": a bag-of-topics vector.
///
/// Each slot is one topic. Real embeddings are learned and dense; this is just
/// enough structure for cosine similarity to produce meaningful rankings.
Float32List embed({
  double database = 0,
  double rust = 0,
  double dart = 0,
  double mobile = 0,
  double search = 0,
  double ml = 0,
  double web = 0,
  double devops = 0,
}) => Float32List.fromList([
  database,
  rust,
  dart,
  mobile,
  search,
  ml,
  web,
  devops,
]);

Future<void> main() async {
  final dir = Directory.systemTemp.createTempSync('phoenix_vector_example');
  final path = '${dir.path}/documents.pvec';

  // --------------------------------------------------------------------
  // 1. Open an index. Dimensionality and metric are fixed at creation.
  // --------------------------------------------------------------------
  final db = PhoenixVectorDB.open(
    path,
    dimensions: 8,
    metric: VectorMetric.cosine,
    maxElements: 1000,
  );

  stdout.writeln('PhoenixDB vector search');
  stdout.writeln('  dimensions : ${db.dimensions}');
  stdout.writeln('  metric     : ${db.metric.label}');
  stdout.writeln('  SIMD kernel: ${db.kernel}');
  stdout.writeln('');

  // --------------------------------------------------------------------
  // 2. Index some documents.
  // --------------------------------------------------------------------
  db.insertAll({
    'phoenixdb': embed(database: 1.0, rust: 0.8, dart: 0.7, mobile: 0.5),
    'sqlite': embed(database: 1.0, mobile: 0.6),
    'tokio': embed(rust: 1.0, web: 0.4),
    'flutter': embed(dart: 1.0, mobile: 1.0, web: 0.4),
    'faiss': embed(search: 1.0, ml: 0.9),
    'elasticsearch': embed(search: 1.0, database: 0.5, web: 0.5),
    'pytorch': embed(ml: 1.0),
    'kubernetes': embed(devops: 1.0, web: 0.3),
  });
  stdout.writeln('indexed ${db.count()} documents\n');

  // --------------------------------------------------------------------
  // 3. Query by similarity.
  // --------------------------------------------------------------------
  final queries = <String, Float32List>{
    'an embedded database for mobile': embed(
      database: 1.0,
      mobile: 0.8,
    ),
    'vector similarity search': embed(search: 1.0, ml: 0.7),
    'systems programming in Rust': embed(rust: 1.0),
  };

  for (final entry in queries.entries) {
    stdout.writeln('query: "${entry.key}"');
    final hits = db.search(VectorQuery(entry.value, k: 3));
    for (final (index, hit) in hits.indexed) {
      // score is cosine similarity here: 1.0 is identical, 0.0 orthogonal.
      final bar = '#' * (hit.score.clamp(0.0, 1.0) * 20).round();
      stdout.writeln(
        '  ${index + 1}. ${hit.id.padRight(14)} '
        '${hit.score.toStringAsFixed(4)} $bar',
      );
    }
    stdout.writeln('');
  }

  // --------------------------------------------------------------------
  // 4. Update, remove, compact.
  // --------------------------------------------------------------------
  // Re-inserting an id replaces it; the old record is tombstoned.
  db.insert('pytorch', embed(ml: 1.0, rust: 0.2, search: 0.3));
  db.remove('kubernetes');

  var stats = db.stats();
  stdout.writeln(
    'after edits: ${stats.live} live, ${stats.total} on disk, '
    '${stats.deleted} tombstoned '
    '(${(stats.deletedRatio * 100).toStringAsFixed(0)}%)',
  );

  // Tombstones cost search time, so reclaim them once they accumulate.
  if (stats.deletedRatio > 0.2) {
    final reclaimed = db.compact();
    stats = db.stats();
    stdout.writeln('compacted: reclaimed $reclaimed record(s), '
        '${stats.total} now on disk');
  }

  // --------------------------------------------------------------------
  // 5. Persist and reopen.
  // --------------------------------------------------------------------
  db.save(); // syncs the vectors and writes the HNSW graph snapshot
  db.close();

  final reopened = PhoenixVectorDB.open(path, dimensions: 8);
  stdout.writeln('\nreopened with ${reopened.count()} documents');
  final top = reopened
      .search(VectorQuery(embed(database: 1.0, rust: 1.0), k: 1))
      .single;
  stdout.writeln('nearest to "rust database": ${top.id}');
  reopened.close();

  // --------------------------------------------------------------------
  // 6. The async API — this is what a Flutter app should use.
  // --------------------------------------------------------------------
  // Every call runs on a worker isolate, so a search over a large index
  // cannot drop a frame at a 120 FPS budget of 8.3 ms.
  final async = await AsyncPhoenixVectorDB.open(path, dimensions: 8);
  final asyncHits = await async.search(
    VectorQuery(embed(search: 1.0, ml: 1.0), k: 2),
  );
  stdout.writeln(
    '\nasync search returned: ${asyncHits.map((h) => h.id).join(', ')}',
  );

  // Concurrent queries are pipelined through the one worker.
  final rng = Random(42);
  final concurrent = await Future.wait([
    for (var i = 0; i < 5; i++)
      async.searchVector(
        Float32List.fromList(
          List.generate(8, (_) => rng.nextDouble()),
        ),
        k: 1,
      ),
  ]);
  stdout.writeln(
    '5 concurrent searches -> ${concurrent.map((r) => r.single.id).join(', ')}',
  );
  await async.close();

  dir.deleteSync(recursive: true);
  stdout.writeln('\ndone');
}
