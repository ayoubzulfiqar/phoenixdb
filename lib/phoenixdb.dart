/// PhoenixDB — an ACID-compliant embedded key/value database for Dart.
///
/// The storage engine is written in Rust (B+Tree index, MVCC transactions,
/// write-ahead log, CRC32-checksummed 4 KiB pages) and reached through a
/// zero-overhead `dart:ffi` binding.
///
/// ## Synchronous use
///
/// ```dart
/// final db = PhoenixDatabase.open('data.pdb');
/// db.insert(utf8Key('hello'), utf8Value('world'));
/// print(db.get(utf8Key('hello')));
/// db.close();
/// ```
///
/// ## Asynchronous use (recommended in Flutter)
///
/// [AsyncPhoenixDB] runs every operation on a worker isolate so disk I/O never
/// blocks the UI thread:
///
/// ```dart
/// final db = await AsyncPhoenixDB.open('data.pdb');
/// await db.transaction((txn) async {
///   await db.insert(utf8Key('a'), utf8Value('1'), txnId: txn);
///   await db.insert(utf8Key('b'), utf8Value('2'), txnId: txn);
/// });
/// await db.close();
/// ```
///
/// ## Vector search
///
/// PhoenixDB also ships an embedded k-NN index (HNSW over memory-mapped `f32`
/// vectors) for local-first semantic search:
///
/// ```dart
/// final index = await AsyncPhoenixVectorDB.open('vectors.pvec', dimensions: 384);
/// await index.insert('doc-1', embedding);
/// final hits = await index.search(VectorQuery(query, k: 5));
/// print(hits.first.id);
/// await index.close();
/// ```
///
/// ## Limits
///
/// Keys are capped at 1 MiB and values at 10 MiB by the FFI layer; the B+Tree
/// additionally caps keys at 1 KiB so a node always holds at least two entries.
/// Values larger than 1 KiB spill onto overflow pages transparently. Vectors
/// are capped at 65 536 dimensions and their ids at 128 bytes.
library;

import 'dart:convert';
import 'dart:typed_data';

export 'src/bindings.dart'
    show PhoenixStatus, PhoenixLoadException, kExpectedAbiVersion;
export 'src/isolate_worker.dart' show AsyncPhoenixDB;
export 'src/native/vector_bindings.dart' show VectorMetric;
export 'src/phoenix_vector_db.dart'
    show PhoenixVectorDB, VectorMatch, VectorQuery, VectorStats;
export 'src/phoenixdb_base.dart'
    show PhoenixDatabase, PhoenixException, KeyNotFoundException;
export 'src/prefs.dart' show PhoenixPrefs;
export 'src/prefs_codec.dart'
    show PrefType, PrefCodec, PhoenixTypeMismatch, PhoenixDecodeException;
export 'src/sql_result.dart' show SqlResult;
export 'src/vector_isolate.dart' show AsyncPhoenixVectorDB;

/// Encodes [s] as UTF-8 bytes for use as a key.
Uint8List utf8Key(String s) => Uint8List.fromList(utf8.encode(s));

/// Encodes [s] as UTF-8 bytes for use as a value.
Uint8List utf8Value(String s) => Uint8List.fromList(utf8.encode(s));

/// Decodes UTF-8 [bytes] back into a string.
String utf8Decode(Uint8List bytes) => utf8.decode(bytes);
