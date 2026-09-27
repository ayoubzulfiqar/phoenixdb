/// Reactive change notifications from Dart: the synchronous watcher, the
/// async streams, and collection document changes.
library;

import 'dart:io';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  late Directory dir;
  late String path;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('phoenix_watch_');
    path = '${dir.path}/w.pdb';
  });

  tearDown(() {
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // Windows can hold files briefly.
    }
  });

  group('synchronous watcher', () {
    test('delivers committed changes under a prefix', () {
      final db = PhoenixDatabase.open(path);
      final watcher = db.watch(prefix: utf8Key('user:'), values: true);
      try {
        db.insert(utf8Key('user:1'), utf8Value('ada'));
        db.insert(utf8Key('post:1'), utf8Value('ignored'));
        db.delete(utf8Key('user:1'));

        final seen = <Change>[];
        while (seen.length < 2) {
          final batch = watcher.poll(timeout: const Duration(seconds: 5));
          expect(batch, isNotEmpty, reason: 'poll timed out');
          seen.addAll(batch);
        }
        expect(seen.map((c) => c.keyString), ['user:1', 'user:1']);
        expect(seen.map((c) => c.kind), [ChangeKind.put, ChangeKind.delete]);
        expect(seen.first.valueString, 'ada');
        expect(seen.last.value, isNull);
        expect(seen.first.commitTs, lessThan(seen.last.commitTs));
        expect(watcher.dropped, 0);
        expect(watcher.isFinished, isFalse);
      } finally {
        watcher.close();
        watcher.close(); // idempotent
        db.close();
      }
    });

    test('an uncommitted transaction publishes nothing', () {
      final db = PhoenixDatabase.open(path);
      final watcher = db.watch();
      try {
        final txn = db.beginTransaction();
        db.insert(utf8Key('k'), utf8Value('v'), txnId: txn);
        expect(
          watcher.poll(timeout: const Duration(milliseconds: 100)),
          isEmpty,
        );
        db.rollback(txn);
        expect(
          watcher.poll(timeout: const Duration(milliseconds: 100)),
          isEmpty,
        );

        db.writeBatch((b) => b.put(utf8Key('k'), utf8Value('v')));
        expect(watcher.poll(timeout: const Duration(seconds: 5)), hasLength(1));
      } finally {
        watcher.close();
        db.close();
      }
    });

    test('a slow consumer is told what it lost', () {
      final db = PhoenixDatabase.open(path);
      final watcher = db.watch(capacity: 4);
      try {
        for (var i = 0; i < 30; i++) {
          db.insert(utf8Key('k$i'), utf8Value('v'));
        }
        final seen = watcher.poll(timeout: const Duration(seconds: 5));
        expect(seen, hasLength(4));
        expect(seen.last.keyString, 'k29', reason: 'the newest survive');
        expect(watcher.dropped, 26);
      } finally {
        watcher.close();
        db.close();
      }
    });

    test('closing the database finishes the watcher', () {
      final db = PhoenixDatabase.open(path);
      final watcher = db.watch();
      db.close();
      expect(watcher.isFinished, isTrue);
      expect(watcher.poll(timeout: const Duration(seconds: 5)), isEmpty);
      watcher.close();
      expect(() => watcher.poll(), throwsA(isA<PhoenixException>()));
    });

    test('arguments are validated', () {
      final db = PhoenixDatabase.open(path);
      try {
        expect(() => db.watch(capacity: 0), throwsArgumentError);
        expect(() => db.watch(capacity: -1), throwsArgumentError);
      } finally {
        db.close();
      }
      expect(db.watch, throwsA(isA<PhoenixException>()), reason: 'closed');
    });
  });

  group('async streams', () {
    test(
      'AsyncPhoenixDB.changes streams writes from another isolate',
      () async {
        final db = await AsyncPhoenixDB.open(path);
        try {
          final seen = <Change>[];
          final sub = db
              .changes(prefix: utf8Key('user:'), values: true)
              .listen(seen.add);
          // Give the watcher isolate a moment to subscribe before writing.
          await Future<void>.delayed(const Duration(milliseconds: 300));
          await db.insert(utf8Key('user:1'), utf8Value('ada'));
          await db.insert(utf8Key('other'), utf8Value('x'));
          await db.insert(utf8Key('user:2'), utf8Value('grace'));

          final deadline = DateTime.now().add(const Duration(seconds: 10));
          while (seen.length < 2 && DateTime.now().isBefore(deadline)) {
            await Future<void>.delayed(const Duration(milliseconds: 50));
          }
          await sub.cancel();
          expect(seen.map((c) => c.keyString), ['user:1', 'user:2']);
          expect(seen.first.valueString, 'ada');
        } finally {
          await db.close();
        }
      },
    );

    test('cancelling the stream releases the watcher isolate', () async {
      final db = await AsyncPhoenixDB.open(path);
      try {
        final first = db.changes();
        final sub = first.listen((_) {});
        await Future<void>.delayed(const Duration(milliseconds: 200));
        await sub.cancel();
        // A second subscription still works, so nothing was left wedged.
        final seen = <Change>[];
        final second = db.changes().listen(seen.add);
        await Future<void>.delayed(const Duration(milliseconds: 300));
        await db.insert(utf8Key('k'), utf8Value('v'));
        final deadline = DateTime.now().add(const Duration(seconds: 10));
        while (seen.isEmpty && DateTime.now().isBefore(deadline)) {
          await Future<void>.delayed(const Duration(milliseconds: 50));
        }
        await second.cancel();
        expect(seen, hasLength(1));
      } finally {
        await db.close();
      }
    });
  });

  group('collection changes', () {
    test('document upserts and deletes stream by id', () async {
      final kb = await AsyncPhoenixCollection.open(
        '${dir.path}/kb',
        dimensions: 2,
        sync: false,
      );
      try {
        final seen = <CollectionChange>[];
        final sub = kb.changes().listen(seen.add);
        await Future<void>.delayed(const Duration(milliseconds: 300));
        await kb.upsert([
          const Document('a', text: 'first'),
          const Document('b', text: 'second'),
        ]);
        await kb.delete(['a']);

        final deadline = DateTime.now().add(const Duration(seconds: 10));
        while (seen.length < 3 && DateTime.now().isBefore(deadline)) {
          await Future<void>.delayed(const Duration(milliseconds: 50));
        }
        await sub.cancel();
        expect(seen.map((c) => c.id), ['a', 'b', 'a']);
        expect(seen.map((c) => c.kind), [
          ChangeKind.put,
          ChangeKind.put,
          ChangeKind.delete,
        ]);
      } finally {
        await kb.close();
      }
    });

    test('the synchronous collection watcher reports ids too', () {
      final kb = PhoenixCollection.open('${dir.path}/kb2', sync: false);
      final watcher = kb.watch();
      try {
        kb.add(const Document('doc-1', text: 'hello'));
        final seen = watcher.poll(timeout: const Duration(seconds: 5));
        expect(seen.map((c) => c.id), ['doc-1']);
        expect(seen.single.kind, ChangeKind.put);
      } finally {
        watcher.close();
        kb.close();
      }
    });
  });
}
