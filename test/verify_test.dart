import 'dart:typed_data';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:test/test.dart';

void main() {
  test('verify succeeds after normal writes', () {
    final db = PhoenixDatabase.open('/tmp/phoenix_verify_test.pdb');
    try {
      db.insert(Uint8List.fromList('a'.codeUnits), Uint8List.fromList('1'.codeUnits));
      db.insert(Uint8List.fromList('b'.codeUnits), Uint8List.fromList('2'.codeUnits));
      db.insert(Uint8List.fromList('c'.codeUnits), Uint8List.fromList('3'.codeUnits));
      db.verify();
      expect(true, isTrue);
    } finally {
      db.close();
    }
  });
}
