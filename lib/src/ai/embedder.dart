/// Text embedding models.
library;

import 'dart:collection';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import '../phoenixdb_base.dart';

/// Whether a text is being stored or used to search. Asymmetric embedding
/// models (Voyage AI, E5, …) embed the two differently.
enum EmbedPurpose {
  /// A text being indexed.
  document,

  /// A search query.
  query,
}

/// Turns texts into fixed-size vectors.
abstract interface class Embedder {
  /// Length of every vector this embedder returns.
  int get dimensions;

  /// Embeds [texts], returning one vector per text, in order.
  Future<List<Float32List>> embed(
    List<String> texts, {
    EmbedPurpose purpose = EmbedPurpose.document,
  });

  /// Releases network resources.
  void close();
}

/// Embeds one text.
extension EmbedOne on Embedder {
  /// Embeds a single [text].
  Future<Float32List> embedOne(
    String text, {
    EmbedPurpose purpose = EmbedPurpose.document,
  }) async => (await embed([text], purpose: purpose)).single;
}

int _fnv1a(String s) {
  var h = 0x811c9dc5;
  for (final unit in s.codeUnits) {
    h ^= unit;
    h = (h * 0x01000193) & 0xffffffff;
  }
  return h;
}

/// A deterministic, offline embedder based on feature hashing of words,
/// word bigrams and character trigrams.
///
/// It captures lexical overlap, not meaning — "car" and "automobile" are
/// unrelated to it — so it is a baseline for tests, demos and fully offline
/// apps, not a substitute for a neural model. Vectors are L2-normalised, so
/// it pairs with the cosine metric.
class HashingEmbedder implements Embedder {
  @override
  final int dimensions;

  /// Creates an embedder producing [dimensions]-dimensional vectors.
  HashingEmbedder({this.dimensions = 256}) {
    if (dimensions <= 0) {
      throw ArgumentError.value(dimensions, 'dimensions', 'must be positive');
    }
  }

  static final _word = RegExp(r'[\p{L}\p{N}]+', unicode: true);

  /// The vector for [text], synchronously.
  Float32List embedSync(String text) {
    final v = Float32List(dimensions);
    void add(String feature, double weight) {
      final h = _fnv1a(feature);
      final sign = (h & 0x80000000) == 0 ? 1.0 : -1.0;
      v[h % dimensions] += sign * weight;
    }

    final words = [
      for (final m in _word.allMatches(text.toLowerCase())) m.group(0)!,
    ];
    for (var i = 0; i < words.length; i++) {
      final w = words[i];
      add('w:$w', 1);
      if (i + 1 < words.length) add('b:$w ${words[i + 1]}', 0.5);
      final padded = '^$w\$';
      for (var j = 0; j + 3 <= padded.length; j++) {
        add('c:${padded.substring(j, j + 3)}', 0.25);
      }
    }
    var norm = 0.0;
    for (final x in v) {
      norm += x * x;
    }
    if (norm > 0) {
      final inv = 1 / sqrt(norm);
      for (var i = 0; i < v.length; i++) {
        v[i] *= inv;
      }
    }
    return v;
  }

  @override
  Future<List<Float32List>> embed(
    List<String> texts, {
    EmbedPurpose purpose = EmbedPurpose.document,
  }) async => [for (final t in texts) embedSync(t)];

  @override
  void close() {}
}

/// Caches another embedder's vectors, in memory and optionally in a
/// [PhoenixDatabase], so a text is embedded (and paid for) once.
///
/// Entries are keyed by [namespace] — use the model name — plus purpose and
/// text, so switching models never serves stale vectors.
class CachedEmbedder implements Embedder {
  /// The embedder that computes missing vectors.
  final Embedder inner;

  /// Distinguishes models sharing one database.
  final String namespace;

  /// Entries kept in memory.
  final int memoryEntries;

  /// Persistent store, or `null` for memory only.
  final PhoenixDatabase? database;

  final LinkedHashMap<String, Float32List> _memory = LinkedHashMap();

  /// Cache hits since creation.
  int hits = 0;

  /// Cache misses since creation.
  int misses = 0;

  /// Wraps [inner].
  CachedEmbedder(
    this.inner, {
    required this.namespace,
    this.database,
    this.memoryEntries = 4096,
  });

  @override
  int get dimensions => inner.dimensions;

  String _key(String text, EmbedPurpose purpose) =>
      '$namespace\u0000${purpose.name}\u0000$text';

  // Database keys are capped at 1 KiB, so the key is a hash; the value
  // repeats the full cache key so a hash collision is detected, not served.
  Uint8List _dbKey(String key) {
    final bytes = utf8.encode(key);
    var a = 0xcbf29ce4, b = 0x84222325;
    for (final x in bytes) {
      a = ((a ^ x) * 0x01000193) & 0xffffffff;
      b = ((b ^ x) * 0x01000197) & 0xffffffff;
    }
    return Uint8List.fromList(
      utf8.encode(
        'emb:${a.toRadixString(16).padLeft(8, '0')}'
        '${b.toRadixString(16).padLeft(8, '0')}:${bytes.length}',
      ),
    );
  }

  Float32List? _load(String key) {
    final db = database;
    if (db == null) return null;
    final raw = db.get(_dbKey(key));
    if (raw == null || raw.length < 4) return null;
    final data = ByteData.sublistView(raw);
    final keyLen = data.getUint32(0, Endian.little);
    if (4 + keyLen > raw.length) return null;
    if (utf8.decode(raw.sublist(4, 4 + keyLen), allowMalformed: true) != key) {
      return null; // collision
    }
    final floats = (raw.length - 4 - keyLen) ~/ 4;
    if (floats != dimensions) return null;
    final v = Float32List(floats);
    for (var i = 0; i < floats; i++) {
      v[i] = data.getFloat32(4 + keyLen + i * 4, Endian.little);
    }
    return v;
  }

  void _store(String key, Float32List v) {
    final db = database;
    if (db == null) return;
    final k = utf8.encode(key);
    final out = ByteData(4 + k.length + v.length * 4);
    out.setUint32(0, k.length, Endian.little);
    final bytes = out.buffer.asUint8List();
    bytes.setRange(4, 4 + k.length, k);
    for (var i = 0; i < v.length; i++) {
      out.setFloat32(4 + k.length + i * 4, v[i], Endian.little);
    }
    db.insert(_dbKey(key), bytes);
  }

  void _remember(String key, Float32List v) {
    _memory.remove(key);
    _memory[key] = v;
    while (_memory.length > memoryEntries) {
      _memory.remove(_memory.keys.first);
    }
  }

  @override
  Future<List<Float32List>> embed(
    List<String> texts, {
    EmbedPurpose purpose = EmbedPurpose.document,
  }) async {
    final out = List<Float32List?>.filled(texts.length, null);
    final missing = <int>[];
    for (var i = 0; i < texts.length; i++) {
      final key = _key(texts[i], purpose);
      final cached = _memory[key] ?? _load(key);
      if (cached != null) {
        hits++;
        _remember(key, cached);
        out[i] = cached;
      } else {
        missing.add(i);
      }
    }
    if (missing.isNotEmpty) {
      misses += missing.length;
      // Embed each distinct missing text once.
      final unique = <String>{for (final i in missing) texts[i]}.toList();
      final vectors = await inner.embed(unique, purpose: purpose);
      final byText = {
        for (var j = 0; j < unique.length; j++) unique[j]: vectors[j],
      };
      for (final i in missing) {
        final v = byText[texts[i]]!;
        final key = _key(texts[i], purpose);
        _remember(key, v);
        _store(key, v);
        out[i] = v;
      }
    }
    return out.cast<Float32List>();
  }

  @override
  void close() => inner.close();
}
