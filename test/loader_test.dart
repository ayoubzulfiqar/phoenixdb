/// Tests for native-library discovery.
///
/// The regression these guard against shipped as a release blocker: every
/// search path was relative to the *consumer's* working directory, so a normal
/// `dart pub get` consumer could not load the library at all — the binaries
/// live inside the installed package, not in the app's tree.
library;

import 'dart:io';

import 'package:phoenixdb/phoenixdb.dart';
import 'package:phoenixdb/src/bindings.dart';
import 'package:test/test.dart';

void main() {
  group('target triples', () {
    test('the current ABI resolves to at least one triple', () {
      // A missing entry means the loader cannot build per-target paths, which
      // degrades to "system loader only" and usually fails.
      expect(currentTargetTriples, isNotEmpty);
      expect(currentTargetTriple, isNotNull);
      expect(currentTargetTriple, currentTargetTriples.first);
    });

    test('windows offers both msvc and gnu', () {
      // CI ships an MSVC build; a local build.sh on MSYS produces gnu. Both
      // must be searched or one of the two setups breaks.
      if (!Platform.isWindows) return;
      expect(currentTargetTriples, contains('x86_64-pc-windows-msvc'));
      expect(currentTargetTriples, contains('x86_64-pc-windows-gnu'));
    });

    test('every triple looks like a rust target', () {
      for (final t in currentTargetTriples) {
        expect(
          t,
          matches(RegExp(r'^[a-z0-9_]+-[a-z0-9]+-[a-z0-9]+(-[a-z0-9]+)?$')),
          reason: '$t is not a plausible target triple',
        );
      }
    });
  });

  group('library name', () {
    test('matches the host platform convention', () {
      final name = defaultLibraryName;
      if (Platform.isWindows) {
        expect(name, 'phoenixdb.dll');
      } else if (Platform.isMacOS) {
        expect(name, 'libphoenixdb.dylib');
      } else {
        expect(name, 'libphoenixdb.so');
      }
    });
  });

  group('ABI contract', () {
    test('the loaded library reports the expected ABI version', () {
      // If this fails, the native library is stale relative to this package —
      // the exact condition that blocks a release.
      final b = PhoenixBindings.load();
      expect(b.abiVersion(), kExpectedAbiVersion);
    });

    test('the expected ABI version is 3', () {
      // Kept in lockstep with tests/ffi_safety.rs. Bumped 2 -> 3 when the
      // `phoenix_vector_*` k-NN surface was added.
      expect(kExpectedAbiVersion, 3);
    });

    test('limits are positive and ordered', () {
      final b = PhoenixBindings.load();
      expect(b.maxKeyLen(), greaterThan(0));
      expect(b.maxValueLen(), greaterThan(b.maxKeyLen()));
    });
  });

  group('discovery from an arbitrary working directory', () {
    // The regression test proper: a consumer's cwd has no native/ directory,
    // so the loader must find the library inside the installed package.
    test('the library loads with cwd set outside the package', () {
      final original = Directory.current;
      final elsewhere = Directory.systemTemp.createTempSync('phoenix_cwd_');
      addTearDown(() {
        Directory.current = original;
        try {
          elsewhere.deleteSync(recursive: true);
        } on FileSystemException {
          // Windows may hold the directory briefly; not a test failure.
        }
      });

      Directory.current = elsewhere;
      expect(
        Directory('${elsewhere.path}/native').existsSync(),
        isFalse,
        reason: 'the probe directory must not contain a native/ folder',
      );

      // A fresh open from here exercises the full search path.
      final db = PhoenixDatabase.open('${elsewhere.path}/probe.pdb');
      addTearDown(db.close);
      db.insert(utf8Key('k'), utf8Value('v'));
      expect(db.get(utf8Key('k')), utf8Value('v'));
    });
  });

  group('explicit path override', () {
    test('a bogus explicit path fails loudly rather than falling back', () {
      // Silently searching elsewhere would hide a deployment mistake.
      expect(
        () => PhoenixBindings.load(path: '/nonexistent/libphoenixdb.so'),
        throwsA(isA<PhoenixLoadException>()),
      );
    });

    test('the failure message lists what was attempted', () {
      try {
        PhoenixBindings.load(path: '/nonexistent/phoenixdb.dll');
        fail('expected a PhoenixLoadException');
      } on PhoenixLoadException catch (e) {
        expect(e.toString(), contains('nonexistent'));
      }
    });
  });
}
