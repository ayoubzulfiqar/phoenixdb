import 'dart:io';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  test('trace listener records lifecycle events', () {
    final dir = Directory.systemTemp.createTempSync();
    try {
      final path = '${dir.path}/trace.pdb';
      final db = PhoenixDatabase.open(path);
      try {
        final events = <String>[];
        db.setTraceListener(events.add);
        db.insert(Uint8List.fromList('k'.codeUnits), Uint8List.fromList('v'.codeUnits));
        db.get(Uint8List.fromList('k'.codeUnits));
        db.delete(Uint8List.fromList('k'.codeUnits));
        final txn = db.beginTransaction();
        db.commit(txn);
        expect(db.traceEvents.any((e) => e.contains('open(path=$path')), isTrue);
        expect(db.traceEvents.any((e) => e.contains('commit(txn=')), isTrue);
      } finally {
        db.close();
      }
    } finally {
      dir.deleteSync(recursive: true);
    }
  });
}
