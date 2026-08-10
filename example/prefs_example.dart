/// Runnable proof that the shared_preferences-style API works verbatim.
library;

import 'dart:io';
import 'package:phoenixdb/phoenixdb.dart';

Future<void> main() async {
  final dir = await Directory.systemTemp.createTemp('phoenix_demo');
  final prefs = await PhoenixPrefs.open('${dir.path}/settings.pdb');

  // --- exactly the shared_preferences shape -------------------------------
  await prefs.setInt('counter', 10);
  await prefs.setBool('repeat', true);
  await prefs.setDouble('decimal', 1.5);
  await prefs.setString('action', 'Start');
  await prefs.setStringList('items', <String>['Earth', 'Moon', 'Sun']);

  final int? counter = await prefs.getInt('counter');
  final bool? repeat = await prefs.getBool('repeat');
  final double? decimal = await prefs.getDouble('decimal');
  final String? action = await prefs.getString('action');
  final List<String>? items = await prefs.getStringList('items');

  print('counter = $counter');
  print('repeat  = $repeat');
  print('decimal = $decimal');
  print('action  = $action');
  print('items   = $items');

  await prefs.remove('counter');
  print('after remove, counter = ${await prefs.getInt('counter')}');

  // --- things shared_preferences cannot do --------------------------------
  await prefs.setMany({'a': 1, 'b': 'two', 'c': false});
  print('atomic batch -> count = ${await prefs.count()}');

  try {
    await prefs.getString('b');
    await prefs.setInt('b', 5);
    await prefs.getString('b'); // now an int; must throw
  } on PhoenixTypeMismatch catch (e) {
    print('type safety: $e');
  }

  await prefs.close();
  dir.deleteSync(recursive: true);
  print('OK');
}
