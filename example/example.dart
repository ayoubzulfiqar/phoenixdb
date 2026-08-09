/// Demonstrates the synchronous and asynchronous PhoenixDB APIs.
library;

import 'dart:io';

import 'package:phoenixdb/phoenixdb.dart';

Future<void> main() async {
  final dir = Directory.systemTemp.createTempSync('phoenixdb_example');
  final path = '${dir.path}/example.pdb';

  // ---- synchronous API ----------------------------------------------------
  final db = PhoenixDatabase.open(path);
  stdout.writeln('opened $path (ABI v${db.abiVersion})');

  db.insert(utf8Key('language'), utf8Value('Dart'));
  db.insert(utf8Key('engine'), utf8Value('Rust'));

  stdout.writeln(
    'language = ${utf8Decode(db.getOrThrow(utf8Key('language')))}',
  );
  stdout.writeln('keys = ${db.count()}');

  // A transaction that commits.
  db.transaction((txn) {
    db.insert(utf8Key('a'), utf8Value('1'), txnId: txn);
    db.insert(utf8Key('b'), utf8Value('2'), txnId: txn);
  });
  stdout.writeln('after commit, keys = ${db.count()}');

  // A transaction that rolls back.
  final txn = db.beginTransaction();
  db.insert(utf8Key('discarded'), utf8Value('nope'), txnId: txn);
  db.rollback(txn);
  stdout.writeln(
    'rolled back, discarded present: '
    '${db.contains(utf8Key('discarded'))}',
  );

  db.delete(utf8Key('a'));
  db.checkpoint();
  db.verify();
  stdout.writeln('verified; final keys = ${db.count()}');
  db.close();

  // ---- asynchronous API ---------------------------------------------------
  final async = await AsyncPhoenixDB.open(path);
  await async.insert(utf8Key('async'), utf8Value('works'));
  final value = await async.get(utf8Key('async'));
  stdout.writeln('async read = ${value == null ? 'null' : utf8Decode(value)}');
  stdout.writeln('async count = ${await async.count()}');
  await async.close();

  dir.deleteSync(recursive: true);
}
