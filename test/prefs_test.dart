/// Tests for the `shared_preferences`-style [PhoenixPrefs] facade.
///
/// These run against the real native library, so they exercise the full path:
/// Dart -> isolate -> FFI -> Rust engine -> disk.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  late Directory dir;
  late PhoenixPrefs prefs;

  setUp(() async {
    dir = await Directory.systemTemp.createTemp('phoenix_prefs_test');
    prefs = await PhoenixPrefs.open('${dir.path}/prefs.pdb');
  });

  tearDown(() async {
    if (!prefs.isClosed) await prefs.close();
    if (dir.existsSync()) {
      try {
        dir.deleteSync(recursive: true);
      } on FileSystemException {
        // Windows sometimes holds the file briefly; not a test failure.
      }
    }
  });

  group('round-trips', () {
    test('string', () async {
      await prefs.setString('action', 'Start');
      expect(await prefs.getString('action'), 'Start');
    });

    test('int', () async {
      await prefs.setInt('counter', 10);
      expect(await prefs.getInt('counter'), 10);
    });

    test('negative and extreme ints', () async {
      await prefs.setInt('min', -9223372036854775808);
      await prefs.setInt('max', 9223372036854775807);
      await prefs.setInt('neg', -42);
      expect(await prefs.getInt('min'), -9223372036854775808);
      expect(await prefs.getInt('max'), 9223372036854775807);
      expect(await prefs.getInt('neg'), -42);
    });

    test('double', () async {
      await prefs.setDouble('decimal', 1.5);
      expect(await prefs.getDouble('decimal'), 1.5);
      await prefs.setDouble('tiny', -0.000123);
      expect(await prefs.getDouble('tiny'), -0.000123);
    });

    test('bool', () async {
      await prefs.setBool('repeat', true);
      await prefs.setBool('off', false);
      expect(await prefs.getBool('repeat'), isTrue);
      expect(await prefs.getBool('off'), isFalse);
    });

    test('string list', () async {
      await prefs.setStringList('items', ['Earth', 'Moon', 'Sun']);
      expect(await prefs.getStringList('items'), ['Earth', 'Moon', 'Sun']);
    });

    test('empty string list and empty strings', () async {
      await prefs.setStringList('none', []);
      expect(await prefs.getStringList('none'), isEmpty);
      await prefs.setStringList('blanks', ['', 'a', '']);
      expect(await prefs.getStringList('blanks'), ['', 'a', '']);
    });

    test('unicode survives', () async {
      await prefs.setString('emoji', 'héllo 🌍 日本語');
      expect(await prefs.getString('emoji'), 'héllo 🌍 日本語');
      await prefs.setStringList('list', ['日本', '🌙']);
      expect(await prefs.getStringList('list'), ['日本', '🌙']);
    });

    test('bytes', () async {
      final data = Uint8List.fromList([0, 1, 255, 128, 0]);
      await prefs.setBytes('blob', data);
      expect(await prefs.getBytes('blob'), data);
    });
  });

  group('absent keys', () {
    test('every getter returns null', () async {
      expect(await prefs.getString('nope'), isNull);
      expect(await prefs.getInt('nope'), isNull);
      expect(await prefs.getDouble('nope'), isNull);
      expect(await prefs.getBool('nope'), isNull);
      expect(await prefs.getStringList('nope'), isNull);
      expect(await prefs.getBytes('nope'), isNull);
      expect(await prefs.getValue('nope'), isNull);
      expect(await prefs.typeOf('nope'), isNull);
    });

    test('containsKey reflects presence', () async {
      expect(await prefs.containsKey('k'), isFalse);
      await prefs.setString('k', 'v');
      expect(await prefs.containsKey('k'), isTrue);
    });
  });

  group('type safety', () {
    test('reading an int as a string throws', () async {
      await prefs.setInt('counter', 10);
      expect(
        () => prefs.getString('counter'),
        throwsA(isA<PhoenixTypeMismatch>()),
      );
    });

    test('reading a string as an int throws', () async {
      await prefs.setString('action', 'Start');
      expect(() => prefs.getInt('action'), throwsA(isA<PhoenixTypeMismatch>()));
    });

    test('bool and int are distinct types', () async {
      await prefs.setBool('flag', true);
      expect(() => prefs.getInt('flag'), throwsA(isA<PhoenixTypeMismatch>()));
    });

    test('typeOf reports the stored type', () async {
      await prefs.setString('s', 'x');
      await prefs.setInt('i', 1);
      await prefs.setDouble('d', 1.0);
      await prefs.setBool('b', true);
      await prefs.setStringList('l', ['a']);
      await prefs.setBytes('y', Uint8List.fromList([1]));

      expect(await prefs.typeOf('s'), PrefType.string);
      expect(await prefs.typeOf('i'), PrefType.int64);
      expect(await prefs.typeOf('d'), PrefType.float64);
      expect(await prefs.typeOf('b'), PrefType.boolean);
      expect(await prefs.typeOf('l'), PrefType.stringList);
      expect(await prefs.typeOf('y'), PrefType.bytes);
    });

    test('getValue returns the natural Dart type', () async {
      await prefs.setString('s', 'x');
      await prefs.setInt('i', 7);
      await prefs.setStringList('l', ['a', 'b']);

      expect(await prefs.getValue('s'), isA<String>());
      expect(await prefs.getValue('i'), 7);
      expect(await prefs.getValue('l'), ['a', 'b']);
    });
  });

  group('overwrite and remove', () {
    test('overwrite replaces the value', () async {
      await prefs.setString('k', 'first');
      await prefs.setString('k', 'second');
      expect(await prefs.getString('k'), 'second');
    });

    test('overwrite can change the type', () async {
      await prefs.setString('k', 'text');
      await prefs.setInt('k', 99);
      expect(await prefs.getInt('k'), 99);
      expect(await prefs.typeOf('k'), PrefType.int64);
    });

    test('remove deletes and reports existence', () async {
      await prefs.setString('k', 'v');
      expect(await prefs.remove('k'), isTrue);
      expect(await prefs.getString('k'), isNull);
      // Removing an absent key is a no-op, not an error.
      expect(await prefs.remove('k'), isFalse);
      expect(await prefs.remove('never-existed'), isFalse);
    });
  });

  group('atomic batches', () {
    test('setMany writes every entry', () async {
      await prefs.setMany({
        'name': 'phoenix',
        'count': 3,
        'ratio': 0.5,
        'on': true,
        'tags': ['a', 'b'],
        'blob': Uint8List.fromList([9, 8]),
      });

      expect(await prefs.getString('name'), 'phoenix');
      expect(await prefs.getInt('count'), 3);
      expect(await prefs.getDouble('ratio'), 0.5);
      expect(await prefs.getBool('on'), isTrue);
      expect(await prefs.getStringList('tags'), ['a', 'b']);
      expect(await prefs.getBytes('blob'), Uint8List.fromList([9, 8]));
    });

    test('setMany rejects an unsupported type', () async {
      expect(
        () => prefs.setMany({'bad': DateTime.now()}),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('setMany with no entries is a no-op', () async {
      await prefs.setMany({});
      expect(await prefs.count(), 0);
    });

    test('removeMany deletes a batch', () async {
      await prefs.setMany({'a': 1, 'b': 2, 'c': 3});
      expect(await prefs.count(), 3);
      await prefs.removeMany(['a', 'c', 'absent']);
      expect(await prefs.getInt('a'), isNull);
      expect(await prefs.getInt('b'), 2);
      expect(await prefs.getInt('c'), isNull);
    });
  });

  group('durability', () {
    test('values survive close and reopen', () async {
      await prefs.setString('action', 'Start');
      await prefs.setInt('counter', 42);
      await prefs.setStringList('items', ['x', 'y']);
      final path = '${dir.path}/prefs.pdb';
      await prefs.close();

      prefs = await PhoenixPrefs.open(path);
      expect(await prefs.getString('action'), 'Start');
      expect(await prefs.getInt('counter'), 42);
      expect(await prefs.getStringList('items'), ['x', 'y']);
    });

    test('checkpoint keeps data readable', () async {
      for (var i = 0; i < 50; i++) {
        await prefs.setInt('k$i', i);
      }
      await prefs.checkpoint();
      expect(await prefs.getInt('k0'), 0);
      expect(await prefs.getInt('k49'), 49);
      expect(await prefs.count(), 50);
    });
  });

  group('allow-list', () {
    test('permits listed keys and rejects others', () async {
      final guarded = await PhoenixPrefs.open(
        '${dir.path}/guarded.pdb',
        allowList: {'repeat', 'action'},
      );
      try {
        await guarded.setBool('repeat', true);
        expect(await guarded.getBool('repeat'), isTrue);
        expect(
          () => guarded.setString('other', 'x'),
          throwsA(isA<ArgumentError>()),
        );
        expect(() => guarded.getString('other'), throwsA(isA<ArgumentError>()));
      } finally {
        await guarded.close();
      }
    });
  });

  group('lifecycle', () {
    test('empty key is rejected', () async {
      expect(() => prefs.setString('', 'v'), throwsA(isA<ArgumentError>()));
    });

    test('operations after close throw', () async {
      await prefs.close();
      expect(prefs.isClosed, isTrue);
      expect(() => prefs.setString('k', 'v'), throwsA(isA<PhoenixException>()));
      expect(() => prefs.getString('k'), throwsA(isA<PhoenixException>()));
    });

    test('close is idempotent', () async {
      await prefs.close();
      await prefs.close();
      expect(prefs.isClosed, isTrue);
    });

    test('wrap borrows without closing the database', () async {
      final db = await AsyncPhoenixDB.open('${dir.path}/shared.pdb');
      try {
        final view = PhoenixPrefs.wrap(db);
        await view.setString('k', 'v');
        await view.close();
        expect(view.isClosed, isTrue);
        // The underlying database must still be usable.
        expect(db.isClosed, isFalse);
        expect(await db.count(), 1);
      } finally {
        await db.close();
      }
    });
  });
}
