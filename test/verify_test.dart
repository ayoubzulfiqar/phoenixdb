import 'dart:io';
import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  group('salvage', _salvage);

  test('verify succeeds after normal writes', () {
    final db = PhoenixDatabase.open('/tmp/phoenix_verify_test.pdb');
    try {
      db.insert(
        Uint8List.fromList('a'.codeUnits),
        Uint8List.fromList('1'.codeUnits),
      );
      db.insert(
        Uint8List.fromList('b'.codeUnits),
        Uint8List.fromList('2'.codeUnits),
      );
      db.insert(
        Uint8List.fromList('c'.codeUnits),
        Uint8List.fromList('3'.codeUnits),
      );
      db.verify();
      expect(true, isTrue);
    } finally {
      db.close();
    }
  });
}

void _salvage() {
  test('salvage recovers a damaged database into a new file', () {
    final dir = Directory.systemTemp.createTempSync('phoenix_salvage_');
    try {
      final source = '${dir.path}/broken.pdb';
      final destination = '${dir.path}/recovered.pdb';
      final db = PhoenixDatabase.open(source);
      db.writeBatch((b) {
        for (var i = 0; i < 300; i++) {
          b.put(
            utf8Key('key${i.toString().padLeft(4, '0')}'),
            utf8Value('v$i'),
          );
        }
      });
      db.insert(utf8Key('big'), utf8Value('x' * 40000));
      db.checkpoint();
      db.close();

      // An intact file salvages completely.
      final clean = PhoenixDatabase.salvage(source, destination);
      expect(clean.isClean, isTrue, reason: '$clean');
      expect(clean.keysRecovered, 301);
      expect(clean.keysSeen, clean.keysRecovered);

      final recovered = PhoenixDatabase.open(destination);
      try {
        expect(recovered.count(), 301);
        expect(utf8Decode(recovered.getOrThrow(utf8Key('key0009'))), 'v9');
        expect(recovered.getOrThrow(utf8Key('big')).length, 40000);
        recovered.verify();
      } finally {
        recovered.close();
      }

      // Scribbling over a page costs that page's keys and nothing else.
      final damaged = '${dir.path}/torn.pdb';
      File(source).copySync(damaged);
      final handle = File(damaged).openSync(mode: FileMode.append);
      try {
        handle.setPositionSync(4096 * 3);
        handle.writeFromSync(List.filled(4096, 0xAA));
      } finally {
        handle.closeSync();
      }
      final partial = PhoenixDatabase.salvage(damaged, '${dir.path}/out2.pdb');
      expect(partial.pagesDamaged, greaterThan(0), reason: '$partial');
      expect(partial.keysRecovered, greaterThan(0));
      expect(partial.keysRecovered, lessThan(301));

      // The damaged original is untouched, and an existing destination is
      // refused rather than overwritten.
      expect(File(damaged).existsSync(), isTrue);
      expect(
        () => PhoenixDatabase.salvage(source, destination),
        throwsA(isA<PhoenixException>()),
      );
    } finally {
      dir.deleteSync(recursive: true);
    }
  });
}
