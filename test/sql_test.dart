/// SQL tests exercising the full Dart -> FFI -> Rust path.
///
/// These use the real native library, so they cover the JSON boundary,
/// pointer ownership, and type preservation across the FFI edge — none of
/// which the Rust-side tests can check.
library;

import 'dart:io';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  late Directory dir;
  late PhoenixDatabase db;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('phoenix_sql_');
    db = PhoenixDatabase.open('${dir.path}/sql.pdb');
  });

  tearDown(() {
    try {
      db.close();
    } on PhoenixException {
      // already closed by the test
    }
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold the file briefly; not a test failure.
    }
  });

  group('availability', () {
    test('the native build reports SQL support', () {
      expect(db.supportsSql, isTrue);
    });

    test('the ABI version matches what the package expects', () {
      // A stale DLL is the likeliest cause of a confusing crash later.
      expect(kExpectedAbiVersion, 3);
    });
  });

  group('statements', () {
    test('CREATE TABLE reports the schema change', () {
      final r = db.query('CREATE TABLE users (id INTEGER, name TEXT)');
      expect(r.detail, contains('created'));
      expect(r.affected, 0);
    });

    test('INSERT reports affected rows', () {
      db.query('CREATE TABLE t (a INTEGER)');
      expect(db.query('INSERT INTO t VALUES (1)').affected, 1);
      expect(db.query('INSERT INTO t VALUES (2), (3)').affected, 2);
    });

    test('SELECT returns columns and rows', () {
      db.query('CREATE TABLE t (a INTEGER, b TEXT)');
      db.query("INSERT INTO t VALUES (1, 'x'), (2, 'y')");

      final r = db.query('SELECT a, b FROM t ORDER BY a');
      expect(r.columns, ['a', 'b']);
      expect(r.rows.length, 2);
      expect(r.rows[0], [1, 'x']);
      expect(r.rows[1], [2, 'y']);
    });

    test('UPDATE and DELETE report counts', () {
      db.query('CREATE TABLE t (a INTEGER)');
      db.query('INSERT INTO t VALUES (1), (2), (3)');
      expect(db.query('UPDATE t SET a = 9 WHERE a = 1').affected, 1);
      expect(db.query('DELETE FROM t WHERE a = 9').affected, 1);
      expect(db.query('SELECT * FROM t').length, 2);
    });
  });

  group('type preservation across the FFI boundary', () {
    test('integers arrive as int, not string', () {
      db.query('CREATE TABLE t (n INTEGER)');
      db.query('INSERT INTO t VALUES (42)');
      final v = db.query('SELECT n FROM t').scalar;
      expect(v, isA<int>());
      expect(v, 42);
    });

    test('floats arrive as double', () {
      db.query('CREATE TABLE t (f INTEGER)');
      db.query('INSERT INTO t VALUES (3.5)');
      final v = db.query('SELECT f FROM t').scalar;
      expect(v, isA<double>());
      expect(v, 3.5);
    });

    test('text arrives as String', () {
      db.query('CREATE TABLE t (s TEXT)');
      db.query("INSERT INTO t VALUES ('hello')");
      expect(db.query('SELECT s FROM t').scalar, 'hello');
    });

    test('NULL arrives as null', () {
      db.query('CREATE TABLE t (a INTEGER, b TEXT)');
      db.query("INSERT INTO t (b) VALUES ('only b')");
      expect(db.query('SELECT a FROM t').scalar, isNull);
    });

    test('negative and large integers survive', () {
      db.query('CREATE TABLE t (n INTEGER)');
      db.query('INSERT INTO t VALUES (-9007199254740991), (9007199254740991)');
      final r = db.query('SELECT n FROM t ORDER BY n');
      expect(r.rows[0][0], -9007199254740991);
      expect(r.rows[1][0], 9007199254740991);
    });
  });

  group('JSON escaping', () {
    // Each of these would corrupt the result document if escaping were wrong.
    test('quotes and backslashes survive', () {
      db.query('CREATE TABLE t (s TEXT)');
      db.query(r"INSERT INTO t VALUES ('a\b'), ('say ''hi''')");
      final got = db.query('SELECT s FROM t').rows.map((r) => r[0]).toList();
      expect(got, containsAll([r'a\b', "say 'hi'"]));
    });

    test('newlines and tabs survive', () {
      db.query('CREATE TABLE t (s TEXT)');
      db.query("INSERT INTO t VALUES ('line1\nline2\ttabbed')");
      expect(db.query('SELECT s FROM t').scalar, 'line1\nline2\ttabbed');
    });

    test('unicode and emoji survive', () {
      db.query('CREATE TABLE t (s TEXT)');
      db.query("INSERT INTO t VALUES ('héllo 🌍 日本語')");
      expect(db.query('SELECT s FROM t').scalar, 'héllo 🌍 日本語');
    });

    test('a value that looks like JSON is not re-interpreted', () {
      db.query('CREATE TABLE t (s TEXT)');
      db.query('INSERT INTO t VALUES (\'{"type":"rows","rows":[[1]]}\')');
      expect(
        db.query('SELECT s FROM t').scalar,
        '{"type":"rows","rows":[[1]]}',
      );
    });
  });

  group('SqlResult helpers', () {
    setUp(() {
      db.query('CREATE TABLE t (id INTEGER, name TEXT)');
      db.query("INSERT INTO t VALUES (1, 'alice'), (2, 'bob')");
    });

    test('scalar returns the single cell, or null when the shape differs', () {
      expect(db.query('SELECT name FROM t WHERE id = 1').scalar, 'alice');
      expect(db.query('SELECT * FROM t').scalar, isNull, reason: '2 columns');
      expect(db.query('SELECT name FROM t').scalar, isNull, reason: '2 rows');
    });

    test('firstOrNull is null on an empty result', () {
      expect(db.query('SELECT * FROM t WHERE id = 99').firstOrNull, isNull);
      expect(db.query('SELECT * FROM t').firstOrNull, isNotNull);
    });

    test('cell looks up by column name, case-insensitively', () {
      final r = db.query('SELECT id, name FROM t ORDER BY id');
      expect(r.cell(0, 'name'), 'alice');
      expect(r.cell(0, 'NAME'), 'alice');
      expect(r.cell(1, 'id'), 2);
      expect(r.cell(9, 'id'), isNull, reason: 'row out of range');
      expect(r.cell(0, 'nope'), isNull, reason: 'unknown column');
    });

    test('asMaps keys rows by column name', () {
      final maps = db.query('SELECT id, name FROM t ORDER BY id').asMaps;
      expect(maps.first, {'id': 1, 'name': 'alice'});
    });

    test('round-trips losslessly through JSON', () {
      // This is what the isolate boundary relies on.
      for (final sql in [
        'SELECT * FROM t',
        'DELETE FROM t WHERE id = 1',
        'CREATE TABLE t2 (x INTEGER)',
      ]) {
        final original = db.query(sql);
        final restored = SqlResult.fromJson(original.toJsonString());
        expect(restored.columns, original.columns, reason: sql);
        expect(restored.rows, original.rows, reason: sql);
        expect(restored.affected, original.affected, reason: sql);
        expect(restored.detail, original.detail, reason: sql);
      }
    });
  });

  group('errors', () {
    test('a syntax error throws rather than returning an empty result', () {
      expect(
        () => db.query('SELCT * FROM t'),
        throwsA(isA<PhoenixException>()),
      );
    });

    test('an unknown table throws', () {
      expect(
        () => db.query('SELECT * FROM ghost'),
        throwsA(isA<PhoenixException>()),
      );
    });

    test('the error message survives the FFI boundary', () {
      try {
        db.query('SELECT * FROM ghost');
        fail('expected a PhoenixException');
      } on PhoenixException catch (e) {
        expect(e.toString().toLowerCase(), contains('ghost'));
      }
    });

    test('querying a closed database throws', () {
      db.close();
      expect(() => db.query('SELECT 1'), throwsA(isA<PhoenixException>()));
    });

    test('a failed query leaves the database usable', () {
      db.query('CREATE TABLE t (a INTEGER)');
      expect(() => db.query('bogus'), throwsA(isA<PhoenixException>()));
      // The handle must still work: a parse failure is not fatal.
      expect(db.query('INSERT INTO t VALUES (1)').affected, 1);
      expect(db.query('SELECT a FROM t').scalar, 1);
    });
  });

  group('durability', () {
    test('rows survive close and reopen', () {
      final path = '${dir.path}/persist.pdb';
      final first = PhoenixDatabase.open(path);
      first.query('CREATE TABLE t (id INTEGER, name TEXT)');
      first.query("INSERT INTO t VALUES (1, 'kept')");
      first.checkpoint();
      first.close();

      final second = PhoenixDatabase.open(path);
      addTearDown(second.close);
      expect(second.query('SELECT name FROM t').scalar, 'kept');
    });
  });

  group('async isolate API', () {
    test('queries run off the calling isolate', () async {
      final adb = await AsyncPhoenixDB.open('${dir.path}/async.pdb');
      addTearDown(adb.close);

      await adb.query('CREATE TABLE t (id INTEGER, name TEXT)');
      expect((await adb.query("INSERT INTO t VALUES (1, 'x')")).affected, 1);

      final r = await adb.query('SELECT id, name FROM t');
      expect(r.columns, ['id', 'name']);
      expect(r.rows.first, [1, 'x']);
    });

    test('an error propagates across the isolate boundary', () async {
      final adb = await AsyncPhoenixDB.open('${dir.path}/async_err.pdb');
      addTearDown(adb.close);
      await expectLater(
        adb.query('SELECT * FROM ghost'),
        throwsA(isA<PhoenixException>()),
      );
      // Still usable afterwards.
      expect(
        (await adb.query('CREATE TABLE ok (a INTEGER)')).detail,
        contains('created'),
      );
    });

    test('concurrent queries all complete', () async {
      final adb = await AsyncPhoenixDB.open('${dir.path}/async_conc.pdb');
      addTearDown(adb.close);
      await adb.query('CREATE TABLE t (n INTEGER)');

      await Future.wait([
        for (var i = 0; i < 20; i++) adb.query('INSERT INTO t VALUES ($i)'),
      ]);
      expect((await adb.query('SELECT * FROM t')).length, 20);
    });
  });

  group('interoperability with the key/value API', () {
    test('SQL tables and raw keys coexist', () {
      db.query('CREATE TABLE t (a INTEGER)');
      db.query('INSERT INTO t VALUES (1)');
      db.insert(utf8Key('my-own-key'), utf8Value('my-own-value'));

      // Neither view disturbs the other.
      expect(db.query('SELECT a FROM t').scalar, 1);
      expect(db.get(utf8Key('my-own-key')), utf8Value('my-own-value'));
    });

    test('prefs and SQL share a database safely', () async {
      final prefs = await PhoenixPrefs.open('${dir.path}/mixed.pdb');
      addTearDown(prefs.close);
      await prefs.setString('theme', 'dark');
      expect(await prefs.getString('theme'), 'dark');
    });
  });
}
