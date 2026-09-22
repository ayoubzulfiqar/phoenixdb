/// Tests for the 4.0 key/value surface: shared engines, range scans, write
/// batches, backup/restore/compact, stats, check, metrics and tracing — and
/// the fixes to scanIter, metricsReport and the trace-event buffer.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

Uint8List k(String s) => utf8Key(s);
Uint8List v(String s) => utf8Value(s);

void main() {
  late Directory dir;
  late String path;
  late PhoenixDatabase db;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('phoenix_v4_');
    path = '${dir.path}/kv.pdb';
    db = PhoenixDatabase.open(path);
  });

  tearDown(() {
    if (!db.isClosed) db.close();
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold files briefly.
    }
  });

  group('one file, many handles', () {
    test('two handles in one isolate share the engine (no corruption)', () {
      // This exact sequence used to corrupt the file: two independent
      // engines overwrote each other's pages, and count() then walked a
      // cyclic leaf chain until the process ran out of memory.
      final other = PhoenixDatabase.open(path);
      for (var i = 0; i < 200; i++) {
        db.insert(Uint8List.fromList([0, i]), Uint8List(100));
      }
      for (var i = 0; i < 200; i++) {
        other.insert(Uint8List.fromList([1, i]), Uint8List(100));
      }
      db.close();
      other.close();
      final reopened = PhoenixDatabase.open(path);
      try {
        expect(reopened.count(), 400);
        final report = reopened.check();
        expect(report.keys + reopened.stats().pendingKeys, greaterThan(0));
      } finally {
        reopened.close();
      }
      db = PhoenixDatabase.open(path); // for tearDown
    });

    test('a sync handle and an async worker see each other', () async {
      final worker = await AsyncPhoenixDB.open(path);
      try {
        await worker.insert(k('from-worker'), v('1'));
        db.insert(k('from-main'), v('2'));
        expect(db.get(k('from-worker')), v('1'));
        expect(await worker.get(k('from-main')), v('2'));
      } finally {
        await worker.close();
      }
      expect(db.count(), 2, reason: 'closing one handle keeps the engine up');
    });
  });

  group('scans', () {
    setUp(() {
      for (var i = 0; i < 100; i++) {
        db.insert(k('user:${i.toString().padLeft(3, '0')}'), v('u$i'));
      }
      db.insert(k('order:1'), v('o'));
      db.checkpoint(); // half the data in the tree...
      db.insert(k('user:007'), v('updated')); // ...and some only in memory
      db.delete(k('user:050'));
    });

    test('scan honours bounds, inclusivity and limit', () {
      final r = db.scan(start: k('user:005'), end: k('user:010'));
      expect(r.map((e) => e.keyString), [
        'user:005',
        'user:006',
        'user:007',
        'user:008',
        'user:009',
      ]);
      expect(r[2].valueString, 'updated');
      final exclusive = db.scan(
        start: k('user:005'),
        startInclusive: false,
        end: k('user:007'),
        endInclusive: true,
      );
      expect(exclusive.map((e) => e.keyString), ['user:006', 'user:007']);
      expect(db.scan(limit: 3).length, 3);
      expect(db.scan().first.keyString, 'order:1');
    });

    test('scanPrefix sees memory and tree, and skips deletes', () {
      final users = db.scanPrefix(k('user:'));
      expect(users.length, 99);
      expect(users.any((e) => e.keyString == 'user:050'), isFalse);
      for (var i = 1; i < users.length; i++) {
        expect(
          users[i - 1].keyString.compareTo(users[i].keyString),
          lessThan(0),
        );
      }
    });

    test('entries pages lazily and can be abandoned', () {
      final all = db.entries(pageSize: 7).toList();
      expect(all.length, 100);
      final firstFive = db.entries(prefix: k('user:'), pageSize: 2).take(5);
      expect(firstFive.map((e) => e.keyString).last, 'user:004');
      expect(
        () => db.entries(prefix: k('a'), start: k('b')).first,
        throwsArgumentError,
      );
    });

    test('a transactional scan sees its own writes only', () {
      final txn = db.beginTransaction();
      db.insert(k('user:zzz'), v('mine'), txnId: txn);
      expect(db.scanPrefix(k('user:'), txnId: txn).length, 100);
      expect(db.scanPrefix(k('user:')).length, 99);
      db.rollback(txn);
    });

    test('scanWhile stops early and scanIter propagates exceptions', () {
      var seen = 0;
      db.scanWhile((key, value) => ++seen < 10);
      expect(seen, 10);
      expect(
        () => db.scanIter((key, value) => throw StateError('boom')),
        throwsStateError,
      );
      // The database is still fully usable afterwards.
      var total = 0;
      db.scanIter((key, value) => total++);
      expect(total, 100);
    });
  });

  group('write batches', () {
    test('apply atomically', () {
      db.insert(k('old'), v('x'));
      db.writeBatch(
        (b) => b
          ..put(k('a'), v('1'))
          ..put(k('b'), v('2'))
          ..delete(k('old'))
          ..deleteIfExists(k('never')),
      );
      expect(db.get(k('a')), v('1'));
      expect(db.get(k('old')), isNull);

      final failing = WriteBatch()
        ..put(k('c'), v('3'))
        ..delete(k('missing'));
      expect(
        () => db.write(failing),
        throwsA(
          isA<PhoenixException>().having((e) => e.isNotFound, 'nf', true),
        ),
      );
      expect(db.get(k('c')), isNull, reason: 'nothing from a failed batch');
      expect(db.stats().activeTransactions, 0);
      db.write(WriteBatch()); // empty is a no-op
    });
  });

  group('maintenance', () {
    test('backup, restore and compact', () {
      for (var i = 0; i < 500; i++) {
        db.insert(k('k$i'), Uint8List(300));
      }
      db.checkpoint(); // put the data into tree pages so there is space to reclaim
      final snapshot = '${dir.path}/snap.pdb';
      db.backup(snapshot);
      db.writeBatch((b) {
        for (var i = 0; i < 500; i++) {
          b.delete(k('k$i'));
        }
      });
      db.checkpoint();
      final before = db.stats().pageCount;
      db.compact();
      expect(db.stats().pageCount, lessThan(before));
      expect(db.count(), 0);
      db.restore(snapshot);
      expect(db.count(), 500);
      expect(db.check().keys, 500);
      expect(
        () => db.restore('${dir.path}/does-not-exist.pdb'),
        throwsA(isA<PhoenixException>()),
      );
    });

    test('stats and check describe the database', () {
      for (var i = 0; i < 2000; i++) {
        db.insert(k('key${i.toString().padLeft(5, '0')}'), Uint8List(50));
      }
      db.checkpoint();
      final stats = db.stats();
      expect(stats.pageCount, greaterThan(2));
      expect(stats.activeTransactions, 0);
      expect(stats.treeTimestamp, greaterThan(0));
      final report = db.check();
      expect(report.keys, 2000);
      expect(report.depth, greaterThanOrEqualTo(2));
      expect(report.freePages, 0);
    });
  });

  group('observability', () {
    test('metrics report counts operations and has no size limit issues', () {
      db.insert(k('a'), v('1'));
      db.get(k('a'));
      final report = db.metricsReport();
      expect(report, contains('PhoenixDB metrics'));
      expect(report, contains('commits=1'));
      expect(db.metricsPrometheus(), contains('phoenixdb_reads_total 1'));
    });

    test('tracing records engine spans', () {
      expect(db.spans(), isEmpty);
      db.setTracing(true);
      db.insert(k('a'), v('1'));
      db.checkpoint();
      final spans = db.spans();
      final commit = spans.firstWhere((s) => s.name == 'commit');
      expect(commit.attributes['writes'], '1');
      expect(spans.any((s) => s.name == 'checkpoint'), isTrue);
      db.setTracing(false);
    });

    test('trace events are a bounded ring, not a leak', () {
      for (var i = 0; i < 1000; i++) {
        db.insert(k('k$i'), v('v'));
      }
      expect(db.traceEvents.length, PhoenixDatabase.maxTraceEvents);
      expect(db.traceEvents.last, contains('insert'));
    });

    test('options are applied and validated', () {
      final fast = PhoenixDatabase.open(
        '${dir.path}/fast.pdb',
        options: const PhoenixOptions(syncOnCommit: false, tracing: true),
      );
      try {
        fast.insert(k('a'), v('1'));
        expect(fast.spans().any((s) => s.name == 'commit'), isTrue);
        expect(fast.metricsReport(), contains('fsyncs=0'));
      } finally {
        fast.close();
      }
      expect(
        () => PhoenixDatabase.open(
          '${dir.path}/bad.pdb',
          options: const PhoenixOptions(fillFactor: 0.1),
        ),
        throwsA(isA<PhoenixException>()),
      );
    });
  });

  group('async API parity', () {
    test('scan, entries, write, stats, backup and metrics', () async {
      db.close();
      final a = await AsyncPhoenixDB.open(path);
      try {
        await a.writeBatch((b) {
          for (var i = 0; i < 50; i++) {
            b.put(k('item:${i.toString().padLeft(2, '0')}'), v('$i'));
          }
        });
        expect((await a.scanPrefix(k('item:1'))).length, 10);
        expect(await a.entries(pageSize: 8).length, 50);
        final stats = await a.stats();
        expect(stats.pendingKeys + (await a.check()).keys, greaterThan(0));
        await a.backup('${dir.path}/async-snap.pdb');
        expect(await a.metricsReport(), contains('PhoenixDB metrics'));
        await a.setTracing(true);
        await a.insert(k('t'), v('t'));
        expect((await a.spans()).any((s) => s.name == 'commit'), isTrue);
      } finally {
        await a.close();
      }
      expect(a.isClosed, isTrue);
      expect(a.count(), throwsA(isA<PhoenixException>()));
      await a.close(); // idempotent
      db = PhoenixDatabase.open(path);
      expect(db.count(), 51);
    });

    test('a failed open reports the engine error', () async {
      await expectLater(
        AsyncPhoenixDB.open(
          '${dir.path}/bad.pdb',
          options: const PhoenixOptions(fillFactor: 3),
        ),
        throwsA(isA<PhoenixException>()),
      );
    });
  });

  group('preferences', () {
    test('getKeys, getAll and clear respect the allow-list', () async {
      db.close();
      final prefs = await PhoenixPrefs.open(path);
      try {
        await prefs.setInt('count', 3);
        await prefs.setString('name', 'ada');
        await prefs.setStringList('tags', ['a', 'b']);
        expect(await prefs.getKeys(), {'count', 'name', 'tags'});
        expect(await prefs.getAll(), {
          'count': 3,
          'name': 'ada',
          'tags': ['a', 'b'],
        });
        final limited = PhoenixPrefs.wrap(prefs.database, allowList: {'name'});
        expect(await limited.getKeys(), {'name'});
        await limited.clear();
        expect(await prefs.getKeys(), {'count', 'tags'});
        await prefs.clear();
        expect(await prefs.getKeys(), isEmpty);
      } finally {
        await prefs.close();
      }
      db = PhoenixDatabase.open(path);
    });
  });
}
