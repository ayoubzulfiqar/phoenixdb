/// Executes the README's Usage examples verbatim to prove they are real.
///
/// Documentation that does not compile is worse than none: it costs a user
/// their time and their trust. This file is the check.
library;

import 'dart:io';

import 'package:phoenixdb/phoenixdb.dart';

Future<void> main() async {
  final dir = Directory.systemTemp.createTempSync('phoenix_readme_');

  // --- README: Preferences -------------------------------------------------
  final prefs = await PhoenixPrefs.open('${dir.path}/settings.pdb');

  await prefs.setString('theme', 'dark');
  await prefs.setInt('launches', 42);
  await prefs.setBool('onboarded', true);

  print(await prefs.getString('theme')); // dark
  print(await prefs.getInt('missing')); // null

  await prefs.close();

  // --- README: SQL (synchronous) -------------------------------------------
  final db = PhoenixDatabase.open('${dir.path}/app.pdb');

  db.query('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)');
  db.query("INSERT INTO users VALUES (1, 'alice'), (2, 'bob')");

  final result = db.query('SELECT name FROM users WHERE id > 1 ORDER BY name');
  print(result.rows); // [[bob]]
  print(result.scalar); // bob   (single cell shorthand)
  print(result.asMaps); // [{name: bob}]

  final affected = db
      .query("UPDATE users SET name = 'carol' WHERE id = 2")
      .affected; // 1
  print(affected);
  db.close();

  // --- README: SQL (asynchronous) ------------------------------------------
  final adb = await AsyncPhoenixDB.open('${dir.path}/app.pdb');
  final r = await adb.query('SELECT name FROM users WHERE id = 1');
  print(r.scalar); // alice
  await adb.close();

  dir.deleteSync(recursive: true);
}
