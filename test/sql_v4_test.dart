/// SQL 4.0 tests through the full Dart -> FFI -> Rust path: bound
/// parameters, caller transactions, primary keys, and result typing.
library;

import 'dart:io';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  late Directory dir;
  late PhoenixDatabase db;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('phoenix_sql4_');
    db = PhoenixDatabase.open('${dir.path}/sql.pdb');
    db.query(
      'CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)',
    );
  });

  tearDown(() {
    if (!db.isClosed) db.close();
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold the file briefly.
    }
  });

  test('parameters bind values, never SQL', () {
    const hostile = "x'); DROP TABLE users; --";
    db.query('INSERT INTO users VALUES (?, ?, ?)', params: [1, hostile, null]);
    db.query('INSERT INTO users VALUES (?, ?, ?)', params: [2, 'bob', 7]);
    expect(
      db.query('SELECT name FROM users WHERE id = ?', params: [1]).scalar,
      hostile,
    );
    expect(
      db.query('SELECT COUNT(*) FROM users WHERE score IS NULL').scalar,
      1,
    );
    expect(
      () => db.query('SELECT * FROM users WHERE id = ?'),
      throwsA(isA<PhoenixException>()),
      reason: 'a missing parameter is an error, not NULL',
    );
    expect(
      () => db.query('SELECT 1 FROM users', params: [Object()]),
      throwsArgumentError,
    );
    expect(
      db
          .query('SELECT id FROM users WHERE id IN (?, ?)', params: [2, true])
          .length,
      2,
      reason: 'true binds as 1, so both rows 1 and 2 match',
    );
  });

  test('SQL joins a caller transaction with key/value writes', () {
    final txn = db.beginTransaction();
    db.query(
      'INSERT INTO users VALUES (1, ?, 10)',
      params: ['ann'],
      txnId: txn,
    );
    db.insert(utf8Key('audit:1'), utf8Value('created ann'), txnId: txn);
    expect(
      db.query('SELECT COUNT(*) FROM users', txnId: txn).scalar,
      1,
      reason: 'a transaction sees its own writes',
    );
    expect(db.query('SELECT COUNT(*) FROM users').scalar, 0);
    db.commit(txn);
    expect(db.query('SELECT name FROM users WHERE id = 1').scalar, 'ann');
    expect(db.get(utf8Key('audit:1')), isNotNull);

    final rolled = db.beginTransaction();
    db.query('DELETE FROM users', txnId: rolled);
    db.rollback(rolled);
    expect(db.query('SELECT COUNT(*) FROM users').scalar, 1);
  });

  test('primary keys are enforced', () {
    db.query('INSERT INTO users VALUES (1, ?, 0)', params: ['a']);
    expect(
      () => db.query('INSERT INTO users VALUES (1, ?, 0)', params: ['b']),
      throwsA(
        isA<PhoenixException>().having(
          (e) => e.message,
          'message',
          contains('duplicate PRIMARY KEY'),
        ),
      ),
    );
  });

  test('richer queries: ordering, aggregates, LIKE, IN', () {
    db.query(
      "INSERT INTO users VALUES (1, 'ann', 30), (2, 'bob', NULL), "
      "(3, 'cat', 10), (4, 'Ben', 20)",
    );
    expect(
      db.query('SELECT score FROM users ORDER BY score').rows.map((r) => r[0]),
      [10, 20, 30, null],
    );
    final agg = db.query(
      'SELECT COUNT(*) AS n, COUNT(score) AS scored, AVG(score) AS mean FROM users',
    );
    expect(agg.asMaps.single, {'n': 4, 'scored': 3, 'mean': 20.0});
    expect(
      db.query("SELECT id FROM users WHERE name ILIKE 'b%' ORDER BY id").rows,
      [
        [2],
        [4],
      ],
    );
    expect(
      db
          .query('SELECT id FROM users WHERE id IN (1, 3) AND NOT (score > 20)')
          .scalar,
      3,
    );
  });

  test('floats keep their type across the FFI boundary', () {
    db.query('CREATE TABLE f (x TEXT)');
    db.query('INSERT INTO f VALUES (?)', params: [3.0]);
    final v = db.query('SELECT x FROM f').scalar;
    expect(v, isA<double>());
    expect(v, 3.0);
  });

  test('async client binds parameters and transactions too', () async {
    db.close();
    final a = await AsyncPhoenixDB.open('${dir.path}/sql.pdb');
    try {
      final txn = await a.beginTransaction();
      await a.query(
        'INSERT INTO users VALUES (?, ?, ?)',
        params: [9, 'zed', 1],
        txnId: txn,
      );
      await a.commit(txn);
      final r = await a.query(
        'SELECT name FROM users WHERE id = ?',
        params: [9],
      );
      expect(r.scalar, 'zed');
    } finally {
      await a.close();
    }
    db = PhoenixDatabase.open('${dir.path}/sql.pdb');
  });
}
