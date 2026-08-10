/// `shared_preferences`-style facade over an [AsyncPhoenixDB].
library;

import 'dart:convert';
import 'dart:typed_data';

import 'isolate_worker.dart';
import 'phoenixdb_base.dart';
import 'prefs_codec.dart';

/// A typed, async key/value store with a `SharedPreferencesAsync`-shaped API.
///
/// Every operation runs on a worker isolate, so disk I/O never blocks the UI
/// thread. Writes are ACID: when a `set*` future completes, the value is
/// durable on disk.
///
/// ```dart
/// final prefs = await PhoenixPrefs.open('settings.pdb');
/// await prefs.setString('action', 'Start');
/// final action = await prefs.getString('action'); // 'Start'
/// await prefs.close();
/// ```
///
/// ## Differences from `shared_preferences`
///
/// * Reading a key with the wrong type throws [PhoenixTypeMismatch] instead of
///   silently yielding `null`.
/// * [setMany] commits a batch of writes in a single atomic transaction.
/// * Keys are arbitrary strings; there is no enforced prefix.
class PhoenixPrefs {
  final AsyncPhoenixDB _db;
  final Set<String>? _allowList;
  bool _closed = false;

  PhoenixPrefs._(this._db, this._allowList);

  /// Opens (or creates) the preference store at [path].
  ///
  /// When [allowList] is non-null, only those keys may be read or written; any
  /// other key throws [ArgumentError]. This mirrors the allow-list option on
  /// `SharedPreferencesWithCache` and is a cheap guard against typo'd keys.
  ///
  /// [cachePages] tunes the native page cache (0 selects the engine default);
  /// [libraryPath] overrides native library discovery.
  static Future<PhoenixPrefs> open(
    String path, {
    Set<String>? allowList,
    int cachePages = 0,
    String? libraryPath,
  }) async {
    final db = await AsyncPhoenixDB.open(
      path,
      cachePages: cachePages,
      libraryPath: libraryPath,
    );
    return PhoenixPrefs._(db, allowList == null ? null : Set.of(allowList));
  }

  /// Wraps an already-open [AsyncPhoenixDB].
  ///
  /// The caller retains ownership: [close] on the facade does **not** close the
  /// underlying database. Use this to share one engine between the typed and
  /// byte-oriented APIs.
  static PhoenixPrefs wrap(AsyncPhoenixDB db, {Set<String>? allowList}) =>
      _BorrowedPrefs(db, allowList == null ? null : Set.of(allowList));

  /// The underlying database, for byte-level or transactional access.
  AsyncPhoenixDB get database => _db;

  /// Whether [close] has run.
  bool get isClosed => _closed;

  /// Throws when the store has been closed.
  void _ensureOpen() {
    if (_closed) {
      throw const PhoenixException(-2, 'preference store is closed');
    }
  }

  /// Validates a key: store open, key non-empty and within the allow-list.
  void _check(String key) {
    _ensureOpen();
    if (key.isEmpty) {
      throw ArgumentError.value(key, 'key', 'must not be empty');
    }
    final allow = _allowList;
    if (allow != null && !allow.contains(key)) {
      throw ArgumentError.value(
        key,
        'key',
        'not in the allow-list ${allow.toList()..sort()}',
      );
    }
  }

  Uint8List _k(String key) => Uint8List.fromList(utf8.encode(key));

  // ---- writes -------------------------------------------------------------

  /// Encodes and stores `value` under `key`.
  Future<void> _write<T>(String key, T value, Uint8List Function(T) encode) {
    _check(key);
    return _db.insert(_k(key), encode(value));
  }

  /// Stores a string.
  Future<void> setString(String key, String value) =>
      _write(key, value, PrefCodec.encodeString);

  /// Stores a 64-bit integer.
  Future<void> setInt(String key, int value) =>
      _write(key, value, PrefCodec.encodeInt);

  /// Stores a double.
  Future<void> setDouble(String key, double value) =>
      _write(key, value, PrefCodec.encodeDouble);

  /// Stores a boolean.
  Future<void> setBool(String key, bool value) =>
      _write(key, value, PrefCodec.encodeBool);

  /// Stores a list of strings.
  Future<void> setStringList(String key, List<String> value) =>
      _write(key, value, PrefCodec.encodeStringList);

  /// Stores opaque bytes.
  Future<void> setBytes(String key, Uint8List value) =>
      _write(key, value, PrefCodec.encodeBytes);

  /// Writes several entries in **one atomic transaction**.
  ///
  /// Either every entry is applied or none is — impossible with
  /// `shared_preferences`, and the reason this facade sits on a real database.
  /// Accepted value types are `String`, `int`, `double`, `bool`,
  /// `List<String>` and `Uint8List`; anything else throws [ArgumentError].
  Future<void> setMany(Map<String, Object> entries) async {
    if (entries.isEmpty) return;
    for (final key in entries.keys) {
      _check(key);
    }
    final encoded = <Uint8List, Uint8List>{};
    entries.forEach((key, value) {
      encoded[_k(key)] = _encodeDynamic(key, value);
    });
    await _db.transaction((txn) async {
      for (final entry in encoded.entries) {
        await _db.insert(entry.key, entry.value, txnId: txn);
      }
    });
  }

  static Uint8List _encodeDynamic(String key, Object value) {
    return switch (value) {
      String v => PrefCodec.encodeString(v),
      int v => PrefCodec.encodeInt(v),
      double v => PrefCodec.encodeDouble(v),
      bool v => PrefCodec.encodeBool(v),
      List<String> v => PrefCodec.encodeStringList(v),
      Uint8List v => PrefCodec.encodeBytes(v),
      _ => throw ArgumentError.value(
        value,
        'entries["$key"]',
        'unsupported type ${value.runtimeType}; expected String, int, double, '
            'bool, List<String> or Uint8List',
      ),
    };
  }

  // ---- reads --------------------------------------------------------------

  /// Fetches `key` and decodes it, or returns `null` when absent.
  ///
  /// Every typed getter is this one line; the decoders live in [PrefCodec] and
  /// are responsible for enforcing the type tag.
  Future<T?> _read<T>(String key, T Function(String, Uint8List) decode) async {
    _check(key);
    final raw = await _db.get(_k(key));
    return raw == null ? null : decode(key, raw);
  }

  /// Reads a string, or `null` when absent.
  ///
  /// Throws [PhoenixTypeMismatch] when the key holds another type.
  Future<String?> getString(String key) => _read(key, PrefCodec.decodeString);

  /// Reads an integer, or `null` when absent.
  Future<int?> getInt(String key) => _read(key, PrefCodec.decodeInt);

  /// Reads a double, or `null` when absent.
  Future<double?> getDouble(String key) => _read(key, PrefCodec.decodeDouble);

  /// Reads a boolean, or `null` when absent.
  Future<bool?> getBool(String key) => _read(key, PrefCodec.decodeBool);

  /// Reads a string list, or `null` when absent.
  Future<List<String>?> getStringList(String key) =>
      _read(key, PrefCodec.decodeStringList);

  /// Reads opaque bytes, or `null` when absent.
  Future<Uint8List?> getBytes(String key) => _read(key, PrefCodec.decodeBytes);

  /// Reads a value of unknown type, or `null` when absent.
  ///
  /// Returns whichever Dart type the value was written with.
  Future<Object?> getValue(String key) => _read(key, PrefCodec.decodeDynamic);

  /// The stored type of [key], or `null` when the key is absent.
  Future<PrefType?> typeOf(String key) => _read(key, PrefCodec.typeOf);

  /// Whether [key] has a value.
  Future<bool> containsKey(String key) async =>
      (await _read(key, (_, raw) => raw)) != null;

  // ---- removal ------------------------------------------------------------

  /// Removes [key]. Returns `true` when it existed.
  Future<bool> remove(String key) async {
    _check(key);
    try {
      return await _db.delete(_k(key));
    } on PhoenixException catch (e) {
      // A missing key is not an error for a preferences API.
      if (e.isNotFound) return false;
      rethrow;
    }
  }

  /// Removes several keys in one atomic transaction.
  Future<void> removeMany(Iterable<String> keys) async {
    final list = keys.toList(growable: false);
    if (list.isEmpty) return;
    for (final key in list) {
      _check(key);
    }
    await _db.transaction((txn) async {
      for (final key in list) {
        try {
          await _db.delete(_k(key), txnId: txn);
        } on PhoenixException catch (e) {
          if (!e.isNotFound) rethrow;
        }
      }
    });
  }

  /// Number of stored entries.
  Future<int> count() {
    _ensureOpen();
    return _db.count();
  }

  /// Flushes pending work and truncates the write-ahead log.
  Future<void> checkpoint() {
    _ensureOpen();
    return _db.checkpoint();
  }

  /// Closes the store and its worker isolate.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _db.close();
  }
}

/// A [PhoenixPrefs] that borrows its database instead of owning it.
class _BorrowedPrefs extends PhoenixPrefs {
  _BorrowedPrefs(super.db, super.allowList) : super._();

  /// Detaches from the shared database without closing it.
  @override
  Future<void> close() async {
    _closed = true;
  }
}
