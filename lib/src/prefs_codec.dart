/// A `shared_preferences`-style typed key/value facade over PhoenixDB.
///
/// The byte-oriented [AsyncPhoenixDB] API is precise but noisy for the common
/// case of storing a handful of settings. [PhoenixPrefs] mirrors the API shape
/// of `SharedPreferencesAsync`, so the ergonomics are familiar:
///
/// ```dart
/// final prefs = await PhoenixPrefs.open('settings.pdb');
///
/// await prefs.setInt('counter', 10);
/// await prefs.setBool('repeat', true);
/// await prefs.setString('action', 'Start');
/// await prefs.setStringList('items', ['Earth', 'Moon']);
///
/// final int? counter = await prefs.getInt('counter');
/// final String? action = await prefs.getString('action');
///
/// await prefs.remove('repeat');
/// await prefs.close();
/// ```
///
/// Unlike `shared_preferences`, every write here is an ACID transaction backed
/// by a write-ahead log, so a value is durable the moment its future completes.
///
/// # Type safety
///
/// Values carry a one-byte type tag on disk. Reading a key with the wrong
/// accessor throws [PhoenixTypeMismatch] rather than returning garbage — a
/// deliberate difference from `shared_preferences`, which silently returns
/// `null`. Use [getValue] when the type is not known ahead of time.
library;

import 'dart:convert';
import 'dart:typed_data';

/// On-disk type tags. The tag is the first byte of every encoded value.
///
/// Tags are part of the storage format: never renumber an existing entry.
enum PrefType {
  /// UTF-8 string.
  string(0x01),

  /// 64-bit signed integer.
  int64(0x02),

  /// IEEE-754 double.
  float64(0x03),

  /// Boolean.
  boolean(0x04),

  /// List of UTF-8 strings.
  stringList(0x05),

  /// Opaque bytes, stored verbatim.
  bytes(0x06);

  /// Creates a tag with its on-disk byte value.
  const PrefType(this.tag);

  /// The byte written as the value's first octet.
  final int tag;

  /// Resolves a tag byte, or `null` when unrecognised.
  static PrefType? fromTag(int tag) {
    for (final t in PrefType.values) {
      if (t.tag == tag) return t;
    }
    return null;
  }
}

/// Thrown when a key is read with an accessor that does not match the type it
/// was written with.
class PhoenixTypeMismatch implements Exception {
  /// The key that was read.
  final String key;

  /// The type the caller asked for.
  final PrefType expected;

  /// The type actually stored.
  final PrefType actual;

  /// Creates a mismatch error.
  const PhoenixTypeMismatch(this.key, this.expected, this.actual);

  @override
  String toString() =>
      'PhoenixTypeMismatch: "$key" holds ${actual.name}, '
      'but was read as ${expected.name}';
}

/// Thrown when stored bytes cannot be decoded as the type their tag claims.
class PhoenixDecodeException implements Exception {
  /// The key whose value failed to decode.
  final String key;

  /// What went wrong.
  final String message;

  /// Creates a decode failure for [key].
  const PhoenixDecodeException(this.key, this.message);

  @override
  String toString() => 'PhoenixDecodeException("$key"): $message';
}

/// Encodes and decodes tagged preference values.
///
/// Layout is `[tag u8][payload…]`:
///
/// * `string`     — UTF-8 bytes
/// * `int64`      — 8 bytes, little-endian two's complement
/// * `float64`    — 8 bytes, little-endian IEEE-754
/// * `boolean`    — 1 byte, `0` or `1`
/// * `stringList` — `[count u32][len u32][utf8]…`, all little-endian
/// * `bytes`      — verbatim
///
/// Fixed-width integers keep decoding allocation-free, and every length is
/// bounds-checked against the buffer before it is used to slice.
class PrefCodec {
  const PrefCodec._();

  /// Encodes a UTF-8 string.
  static Uint8List encodeString(String v) {
    final body = utf8.encode(v);
    final out = Uint8List(1 + body.length);
    out[0] = PrefType.string.tag;
    out.setRange(1, out.length, body);
    return out;
  }

  /// Encodes a 64-bit signed integer.
  static Uint8List encodeInt(int v) {
    final out = Uint8List(9);
    out[0] = PrefType.int64.tag;
    ByteData.view(out.buffer).setInt64(1, v, Endian.little);
    return out;
  }

  /// Encodes a double.
  static Uint8List encodeDouble(double v) {
    final out = Uint8List(9);
    out[0] = PrefType.float64.tag;
    ByteData.view(out.buffer).setFloat64(1, v, Endian.little);
    return out;
  }

  /// Encodes a boolean.
  static Uint8List encodeBool(bool v) =>
      Uint8List.fromList([PrefType.boolean.tag, v ? 1 : 0]);

  /// Encodes a list of strings.
  static Uint8List encodeStringList(List<String> v) {
    final parts = v.map(utf8.encode).toList(growable: false);
    var total = 1 + 4;
    for (final p in parts) {
      total += 4 + p.length;
    }
    final out = Uint8List(total);
    final view = ByteData.view(out.buffer);
    out[0] = PrefType.stringList.tag;
    view.setUint32(1, parts.length, Endian.little);
    var offset = 5;
    for (final p in parts) {
      view.setUint32(offset, p.length, Endian.little);
      offset += 4;
      out.setRange(offset, offset + p.length, p);
      offset += p.length;
    }
    return out;
  }

  /// Encodes opaque bytes.
  static Uint8List encodeBytes(Uint8List v) {
    final out = Uint8List(1 + v.length);
    out[0] = PrefType.bytes.tag;
    out.setRange(1, out.length, v);
    return out;
  }

  /// Reads the type tag, or throws when the buffer is empty or unknown.
  static PrefType typeOf(String key, Uint8List raw) {
    if (raw.isEmpty) {
      throw PhoenixDecodeException(key, 'value is empty (no type tag)');
    }
    final t = PrefType.fromTag(raw[0]);
    if (t == null) {
      throw PhoenixDecodeException(
        key,
        'unknown type tag 0x${raw[0].toRadixString(16).padLeft(2, '0')}',
      );
    }
    return t;
  }

  /// Verifies the tag matches [want], throwing [PhoenixTypeMismatch] if not.
  static void expect(String key, Uint8List raw, PrefType want) {
    final actual = typeOf(key, raw);
    if (actual != want) throw PhoenixTypeMismatch(key, want, actual);
  }

  /// Decodes a string payload.
  static String decodeString(String key, Uint8List raw) {
    expect(key, raw, PrefType.string);
    try {
      return utf8.decode(raw.sublist(1));
    } on FormatException catch (e) {
      throw PhoenixDecodeException(key, 'invalid UTF-8: ${e.message}');
    }
  }

  /// Decodes an integer payload.
  static int decodeInt(String key, Uint8List raw) {
    expect(key, raw, PrefType.int64);
    if (raw.length != 9) {
      throw PhoenixDecodeException(
        key,
        'int64 needs 9 bytes, found ${raw.length}',
      );
    }
    return ByteData.view(
      raw.buffer,
      raw.offsetInBytes,
      raw.length,
    ).getInt64(1, Endian.little);
  }

  /// Decodes a double payload.
  static double decodeDouble(String key, Uint8List raw) {
    expect(key, raw, PrefType.float64);
    if (raw.length != 9) {
      throw PhoenixDecodeException(
        key,
        'float64 needs 9 bytes, found ${raw.length}',
      );
    }
    return ByteData.view(
      raw.buffer,
      raw.offsetInBytes,
      raw.length,
    ).getFloat64(1, Endian.little);
  }

  /// Decodes a boolean payload.
  static bool decodeBool(String key, Uint8List raw) {
    expect(key, raw, PrefType.boolean);
    if (raw.length != 2) {
      throw PhoenixDecodeException(
        key,
        'bool needs 2 bytes, found ${raw.length}',
      );
    }
    return raw[1] != 0;
  }

  /// Decodes a string-list payload, bounds-checking every element.
  static List<String> decodeStringList(String key, Uint8List raw) {
    expect(key, raw, PrefType.stringList);
    if (raw.length < 5) {
      throw PhoenixDecodeException(key, 'string list header is truncated');
    }
    final view = ByteData.view(raw.buffer, raw.offsetInBytes, raw.length);
    final count = view.getUint32(1, Endian.little);
    // Each element costs at least a 4-byte header, so this rejects an absurd
    // count before it can drive a huge allocation.
    if (count > (raw.length - 5) ~/ 4) {
      throw PhoenixDecodeException(
        key,
        'string list declares $count entries, buffer holds ${raw.length} bytes',
      );
    }
    final out = <String>[];
    var offset = 5;
    for (var i = 0; i < count; i++) {
      if (offset + 4 > raw.length) {
        throw PhoenixDecodeException(
          key,
          'element $i header runs past the end',
        );
      }
      final len = view.getUint32(offset, Endian.little);
      offset += 4;
      if (offset + len > raw.length) {
        throw PhoenixDecodeException(
          key,
          'element $i payload runs past the end',
        );
      }
      try {
        out.add(utf8.decode(raw.sublist(offset, offset + len)));
      } on FormatException catch (e) {
        throw PhoenixDecodeException(
          key,
          'element $i is invalid UTF-8: ${e.message}',
        );
      }
      offset += len;
    }
    return out;
  }

  /// Decodes an opaque byte payload.
  static Uint8List decodeBytes(String key, Uint8List raw) {
    expect(key, raw, PrefType.bytes);
    return Uint8List.fromList(raw.sublist(1));
  }

  /// Decodes any tagged value into its natural Dart type.
  static Object decodeDynamic(String key, Uint8List raw) {
    return switch (typeOf(key, raw)) {
      PrefType.string => decodeString(key, raw),
      PrefType.int64 => decodeInt(key, raw),
      PrefType.float64 => decodeDouble(key, raw),
      PrefType.boolean => decodeBool(key, raw),
      PrefType.stringList => decodeStringList(key, raw),
      PrefType.bytes => decodeBytes(key, raw),
    };
  }
}
