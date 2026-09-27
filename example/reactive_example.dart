/// Reactive queries: rebuild what you show when the data behind it moves,
/// instead of polling the database.
///
/// Run with:
/// ```sh
/// dart run example/reactive_example.dart
/// ```
///
/// Two streams run at once — one over a key prefix, one over a collection's
/// documents — while writes happen from the main isolate. Each subscription
/// polls on its own isolate, which is why a Flutter UI can listen to one
/// without dropping frames.
library;

import 'dart:async';
import 'dart:io';

import 'package:phoenixdb/phoenixdb.dart';

Future<void> main() async {
  final dir = Directory.systemTemp.createTempSync('phoenix_reactive_');
  final db = await AsyncPhoenixDB.open('${dir.path}/app.pdb');
  final kb = await AsyncPhoenixCollection.open('${dir.path}/kb', sync: false);
  try {
    // --- keys ---------------------------------------------------------------
    final keyEvents = <String>[];
    final keys = db
        .changes(prefix: utf8Key('user:'), values: true)
        .listen(
          (c) => keyEvents.add(
            '${c.kind.name} ${c.keyString}'
            '${c.valueString == null ? '' : ' = ${c.valueString}'}',
          ),
        );

    // --- documents ----------------------------------------------------------
    final docEvents = <String>[];
    final docs = kb.changes().listen(
      (c) => docEvents.add('${c.kind.name} ${c.id}'),
    );

    // Give both subscriptions a moment to attach before writing.
    await Future<void>.delayed(const Duration(milliseconds: 300));

    await db.insert(utf8Key('user:1'), utf8Value('ada'));
    await db.insert(utf8Key('session:1'), utf8Value('ignored'));
    await db.insert(utf8Key('user:2'), utf8Value('grace'));
    await db.delete(utf8Key('user:1'));

    await kb.upsert([
      const Document('note-1', text: 'reactive queries are handy'),
      const Document('note-2', text: 'so is hybrid search'),
    ]);
    await kb.delete(['note-1']);

    await _settle(() => keyEvents.length >= 3 && docEvents.length >= 3);
    await keys.cancel();
    await docs.cancel();

    print('key changes under "user:":');
    for (final e in keyEvents) {
      print('  $e');
    }
    print('(the session: write was filtered out by the prefix)\n');

    print('document changes:');
    for (final e in docEvents) {
      print('  $e');
    }

    // A watcher that falls behind loses the oldest changes, never the newest,
    // and says so — the synchronous API shows it plainly.
    final sync = PhoenixDatabase.open('${dir.path}/busy.pdb');
    final watcher = sync.watch(capacity: 4);
    try {
      for (var i = 0; i < 20; i++) {
        sync.insert(utf8Key('k$i'), utf8Value('v'));
      }
      final seen = watcher.poll(timeout: const Duration(seconds: 2));
      print(
        '\nbounded queue: kept ${seen.length} newest '
        '(${seen.first.keyString}..${seen.last.keyString}), '
        'dropped ${watcher.dropped}',
      );
    } finally {
      watcher.close();
      sync.close();
    }
  } finally {
    await kb.close();
    await db.close();
    dir.deleteSync(recursive: true);
  }
}

/// Waits until [done] or a couple of seconds pass, whichever comes first.
Future<void> _settle(bool Function() done) async {
  final deadline = DateTime.now().add(const Duration(seconds: 5));
  while (!done() && DateTime.now().isBefore(deadline)) {
    await Future<void>.delayed(const Duration(milliseconds: 25));
  }
}
