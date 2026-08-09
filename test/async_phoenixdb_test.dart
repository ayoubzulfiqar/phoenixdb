/// Tests for the isolate-backed asynchronous API.
///
/// Requires the native library; the suite skips itself when it is absent.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  late Directory dir;
  late AsyncPhoenixDB db;
  String? skipReason;

  setUpAll(() {
    try {
      final probe = Directory.systemTemp.createTempSync('phoenix_async_probe');
      final p = PhoenixDatabase.open('${probe.path}/probe.pdb');
      p.close();
      probe.deleteSync(recursive: true);
    } on PhoenixLoadException catch (e) {
      skipReason = 'native library unavailable: ${e.message}';
    }
  });

  setUp(() async {
    if (skipReason != null) return;
    dir = Directory.systemTemp.createTempSync('phoenixdb_async_test');
    db = await AsyncPhoenixDB.open('${dir.path}/async.pdb');
  });

  tearDown(() async {
    if (skipReason != null) return;
    if (!db.isClosed) await db.close();
    if (dir.existsSync()) dir.deleteSync(recursive: true);
  });

  test('insert and get round-trip', () async {
    await db.insert(utf8Key('hello'), utf8Value('world'));
    final value = await db.get(utf8Key('hello'));
    expect(value, isNotNull);
    expect(utf8Decode(value!), 'world');
  }, skip: skipReason);

  test('missing key resolves to null', () async {
    expect(await db.get(utf8Key('absent')), isNull);
  }, skip: skipReason);

  test('delete reports whether the key existed', () async {
    await db.insert(utf8Key('k'), utf8Value('v'));
    expect(await db.delete(utf8Key('k')), isTrue);
    expect(await db.delete(utf8Key('k')), isFalse);
  }, skip: skipReason);

  test('count tracks the number of keys', () async {
    for (var i = 0; i < 20; i++) {
      await db.insert(utf8Key('k$i'), utf8Value('v$i'));
    }
    expect(await db.count(), 20);
  }, skip: skipReason);

  test('transaction helper commits', () async {
    await db.transaction((txn) async {
      await db.insert(utf8Key('a'), utf8Value('1'), txnId: txn);
      await db.insert(utf8Key('b'), utf8Value('2'), txnId: txn);
    });
    expect(await db.count(), 2);
  }, skip: skipReason);

  test('transaction helper rolls back on error', () async {
    await expectLater(
      db.transaction((txn) async {
        await db.insert(utf8Key('a'), utf8Value('1'), txnId: txn);
        throw StateError('abort');
      }),
      throwsA(isA<StateError>()),
    );
    expect(await db.get(utf8Key('a')), isNull);
  }, skip: skipReason);

  test('explicit rollback discards writes', () async {
    final txn = await db.beginTransaction();
    await db.insert(utf8Key('x'), utf8Value('1'), txnId: txn);
    await db.rollback(txn);
    expect(await db.get(utf8Key('x')), isNull);
  }, skip: skipReason);

  test('concurrent futures are all served', () async {
    final writes = <Future<void>>[
      for (var i = 0; i < 40; i++)
        db.insert(utf8Key('c$i'), utf8Value('v$i')),
    ];
    await Future.wait(writes);
    expect(await db.count(), 40);

    final reads = await Future.wait([
      for (var i = 0; i < 40; i++) db.get(utf8Key('c$i')),
    ]);
    for (var i = 0; i < 40; i++) {
      expect(utf8Decode(reads[i]!), 'v$i');
    }
  }, skip: skipReason);

  test('large value survives the isolate hop', () async {
    final value = Uint8List.fromList(List.generate(200000, (i) => i % 249));
    await db.insert(utf8Key('big'), value);
    expect(await db.get(utf8Key('big')), equals(value));
  }, skip: skipReason);

  test('checkpoint and verify succeed', () async {
    for (var i = 0; i < 100; i++) {
      await db.insert(utf8Key('k${i.toString().padLeft(4, '0')}'),
          utf8Value('value$i'));
    }
    await db.checkpoint();
    await db.verify();
    expect(await db.count(), 100);
  }, skip: skipReason);

  test('operations after close fail', () async {
    await db.close();
    expect(db.isClosed, isTrue);
    await expectLater(db.count(), throwsA(isA<PhoenixException>()));
  }, skip: skipReason);

  test('double close is a no-op', () async {
    await db.close();
    await expectLater(db.close(), completes);
  }, skip: skipReason);
}
