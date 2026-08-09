/// Tests for the synchronous PhoenixDB API.
///
/// These require the native library; run `./build.sh` (or `build.ps1`) first.
/// When it is missing the whole suite is skipped rather than failing.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  late Directory dir;
  late PhoenixDatabase db;
  String? skipReason;

  setUpAll(() {
    try {
      final probe = Directory.systemTemp.createTempSync('phoenix_probe');
      final p = PhoenixDatabase.open('${probe.path}/probe.pdb');
      p.close();
      probe.deleteSync(recursive: true);
    } on PhoenixLoadException catch (e) {
      skipReason = 'native library unavailable: ${e.message}';
    }
  });

  setUp(() {
    if (skipReason != null) return;
    dir = Directory.systemTemp.createTempSync('phoenixdb_test');
    db = PhoenixDatabase.open('${dir.path}/test.pdb');
  });

  tearDown(() {
    if (skipReason != null) return;
    if (!db.isClosed) db.close();
    if (dir.existsSync()) dir.deleteSync(recursive: true);
  });

  group('basic operations', () {
    test('insert and get round-trip', () {
      db.insert(utf8Key('hello'), utf8Value('world'));
      expect(utf8Decode(db.getOrThrow(utf8Key('hello'))), 'world');
    }, skip: skipReason);

    test('missing key returns null', () {
      expect(db.get(utf8Key('nope')), isNull);
    }, skip: skipReason);

    test('getOrThrow throws KeyNotFoundException', () {
      expect(
        () => db.getOrThrow(utf8Key('nope')),
        throwsA(isA<KeyNotFoundException>()),
      );
    }, skip: skipReason);

    test('overwrite replaces the value', () {
      db.insert(utf8Key('k'), utf8Value('v1'));
      db.insert(utf8Key('k'), utf8Value('v2'));
      expect(utf8Decode(db.getOrThrow(utf8Key('k'))), 'v2');
      expect(db.count(), 1);
    }, skip: skipReason);

    test('delete removes the key and reports absence', () {
      db.insert(utf8Key('k'), utf8Value('v'));
      expect(db.delete(utf8Key('k')), isTrue);
      expect(db.get(utf8Key('k')), isNull);
      expect(db.delete(utf8Key('k')), isFalse);
    }, skip: skipReason);

    test('binary keys and values survive', () {
      final key = Uint8List.fromList([0, 1, 2, 255, 0]);
      final value = Uint8List.fromList(List.generate(5000, (i) => i % 256));
      db.insert(key, value);
      expect(db.getOrThrow(key), equals(value));
    }, skip: skipReason);

    test('empty value is allowed', () {
      db.insert(utf8Key('empty'), Uint8List(0));
      expect(db.getOrThrow(utf8Key('empty')).length, 0);
    }, skip: skipReason);

    test('count reflects inserts and deletes', () {
      for (var i = 0; i < 25; i++) {
        db.insert(utf8Key('k$i'), utf8Value('v$i'));
      }
      expect(db.count(), 25);
      db.delete(utf8Key('k0'));
      expect(db.count(), 24);
    }, skip: skipReason);
  });

  group('transactions', () {
    test('commit makes writes visible', () {
      final txn = db.beginTransaction();
      db.insert(utf8Key('a'), utf8Value('1'), txnId: txn);
      db.commit(txn);
      expect(utf8Decode(db.getOrThrow(utf8Key('a'))), '1');
    }, skip: skipReason);

    test('rollback discards writes', () {
      final txn = db.beginTransaction();
      db.insert(utf8Key('a'), utf8Value('1'), txnId: txn);
      db.rollback(txn);
      expect(db.get(utf8Key('a')), isNull);
    }, skip: skipReason);

    test('transaction helper commits on success', () {
      db.transaction((txn) {
        db.insert(utf8Key('x'), utf8Value('1'), txnId: txn);
        db.insert(utf8Key('y'), utf8Value('2'), txnId: txn);
      });
      expect(db.count(), 2);
    }, skip: skipReason);

    test('transaction helper rolls back on throw', () {
      expect(
        () => db.transaction((txn) {
          db.insert(utf8Key('x'), utf8Value('1'), txnId: txn);
          throw StateError('abort');
        }),
        throwsA(isA<StateError>()),
      );
      expect(db.get(utf8Key('x')), isNull);
    }, skip: skipReason);

    test('uncommitted writes are visible to their own transaction', () {
      final txn = db.beginTransaction();
      db.insert(utf8Key('own'), utf8Value('read'), txnId: txn);
      expect(utf8Decode(db.getOrThrow(utf8Key('own'), txnId: txn)), 'read');
      db.rollback(txn);
    }, skip: skipReason);
  });

  group('validation', () {
    test('empty key is rejected with -2', () {
      try {
        db.insert(Uint8List(0), utf8Value('v'));
        fail('expected a validation failure');
      } on PhoenixException catch (e) {
        expect(e.status, PhoenixStatus.invalidArgument);
      }
    }, skip: skipReason);

    test('oversized key is rejected with -2', () {
      final key = Uint8List(db.maxKeyLength + 1);
      try {
        db.insert(key, utf8Value('v'));
        fail('expected a validation failure');
      } on PhoenixException catch (e) {
        expect(e.status, PhoenixStatus.invalidArgument);
      }
    }, skip: skipReason);

    test('operations on a closed database throw', () {
      db.close();
      expect(() => db.count(), throwsA(isA<PhoenixException>()));
    }, skip: skipReason);

    test('double close is a no-op', () {
      db.close();
      expect(db.close, returnsNormally);
    }, skip: skipReason);

    test('reported limits match the documented values', () {
      expect(db.maxKeyLength, 1024 * 1024);
      expect(db.maxValueLength, 10 * 1024 * 1024);
      expect(db.abiVersion, kExpectedAbiVersion);
    }, skip: skipReason);
  });

  group('durability', () {
    test('data survives close and reopen', () {
      final path = '${dir.path}/persist.pdb';
      final first = PhoenixDatabase.open(path);
      for (var i = 0; i < 50; i++) {
        first.insert(utf8Key('k$i'), utf8Value('v$i'));
      }
      first.close();

      final second = PhoenixDatabase.open(path);
      expect(second.count(), 50);
      expect(utf8Decode(second.getOrThrow(utf8Key('k7'))), 'v7');
      second.close();
    }, skip: skipReason);

    test('large values use overflow pages transparently', () {
      final value = Uint8List.fromList(List.generate(250000, (i) => i % 251));
      db.insert(utf8Key('big'), value);
      db.checkpoint();
      expect(db.getOrThrow(utf8Key('big')), equals(value));
    }, skip: skipReason);

    test('verify passes after heavy churn', () {
      for (var i = 0; i < 400; i++) {
        db.insert(
          utf8Key('key${i.toString().padLeft(5, '0')}'),
          utf8Value('value$i'),
        );
      }
      for (var i = 0; i < 400; i += 3) {
        db.delete(utf8Key('key${i.toString().padLeft(5, '0')}'));
      }
      db.checkpoint();
      expect(db.verify, returnsNormally);
    }, skip: skipReason);
  });
}
