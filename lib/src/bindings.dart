/// Low-level `dart:ffi` bindings to the PhoenixDB C ABI.
///
/// This file mirrors `native/include/phoenixdb.h` exactly. Nothing here is
/// safe to call without upholding the ownership rules documented on each
/// function; use the high-level API in `package:phoenixdb/phoenixdb.dart`
/// instead.
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

/// Status codes returned by every `phoenix_*` entry point.
///
/// Mirrors `PhoenixStatus` in the Rust `error` module. Zero is success and
/// every failure is negative.
abstract final class PhoenixStatus {
  /// Operation succeeded.
  static const int ok = 0;

  /// Unclassified internal failure.
  static const int error = -1;

  /// A pointer was null or a length exceeded its documented limit.
  static const int invalidArgument = -2;

  /// The key does not exist in the caller's snapshot.
  static const int notFound = -3;

  /// A page failed CRC32 verification.
  static const int corruption = -4;

  /// Underlying I/O failure.
  static const int io = -5;

  /// Write-write conflict; the transaction should be retried.
  static const int conflict = -6;

  /// A Rust panic was caught at the FFI boundary.
  static const int panic = -7;

  /// The transaction id is unknown or already finished.
  static const int txnNotFound = -8;

  /// A structural capacity limit was reached.
  static const int full = -9;
}

/// Opaque database handle. Only ever held as a `Pointer<PhoenixDB>`.
final class PhoenixDB extends Opaque {}

/// An owned byte buffer returned by `phoenix_get`.
///
/// Must be released with `phoenix_buffer_free`; the Dart allocator must never
/// free `ptr`.
final class PhoenixBuffer extends Struct {
  /// Pointer to [len] bytes owned by the native library.
  external Pointer<Uint8> ptr;

  /// Number of valid bytes.
  @Size()
  external int len;

  /// Allocated capacity, needed by the native free routine.
  @Size()
  external int cap;
}

// ---------------------------------------------------------------------------
// Native function typedefs
// ---------------------------------------------------------------------------

/// Native signature for `OpenNative`.
typedef OpenNative =
    Int32 Function(
      Pointer<Utf8> path,
      Size cachePages,
      Pointer<Pointer<PhoenixDB>> outHandle,
    );

/// Dart signature for `OpenDart`.
typedef OpenDart =
    int Function(
      Pointer<Utf8> path,
      int cachePages,
      Pointer<Pointer<PhoenixDB>> outHandle,
    );

/// Native signature for `CloseNative`.
typedef CloseNative = Int32 Function(Pointer<PhoenixDB> handle);

/// Dart signature for `CloseDart`.
typedef CloseDart = int Function(Pointer<PhoenixDB> handle);

/// Native signature for `BeginTxnNative`.
typedef BeginTxnNative =
    Int32 Function(
      Pointer<PhoenixDB> handle,
      Int32 readOnly,
      Pointer<Uint64> outTxn,
    );

/// Dart signature for `BeginTxnDart`.
typedef BeginTxnDart =
    int Function(
      Pointer<PhoenixDB> handle,
      int readOnly,
      Pointer<Uint64> outTxn,
    );

/// Native signature for `TxnOpNative`.
typedef TxnOpNative = Int32 Function(Pointer<PhoenixDB> handle, Uint64 txnId);

/// Dart signature for `TxnOpDart`.
typedef TxnOpDart = int Function(Pointer<PhoenixDB> handle, int txnId);

/// Native signature for `InsertNative`.
typedef InsertNative =
    Int32 Function(
      Pointer<PhoenixDB> handle,
      Uint64 txnId,
      Pointer<Uint8> key,
      Size keyLen,
      Pointer<Uint8> value,
      Size valueLen,
    );

/// Dart signature for `InsertDart`.
typedef InsertDart =
    int Function(
      Pointer<PhoenixDB> handle,
      int txnId,
      Pointer<Uint8> key,
      int keyLen,
      Pointer<Uint8> value,
      int valueLen,
    );

/// Native signature for `PutAutoNative`.
typedef PutAutoNative =
    Int32 Function(
      Pointer<PhoenixDB> handle,
      Pointer<Uint8> key,
      Size keyLen,
      Pointer<Uint8> value,
      Size valueLen,
    );

/// Dart signature for `PutAutoDart`.
typedef PutAutoDart =
    int Function(
      Pointer<PhoenixDB> handle,
      Pointer<Uint8> key,
      int keyLen,
      Pointer<Uint8> value,
      int valueLen,
    );

/// Native signature for `GetNative`.
typedef GetNative =
    Int32 Function(
      Pointer<PhoenixDB> handle,
      Uint64 txnId,
      Pointer<Uint8> key,
      Size keyLen,
      Pointer<PhoenixBuffer> out,
    );

/// Dart signature for `GetDart`.
typedef GetDart =
    int Function(
      Pointer<PhoenixDB> handle,
      int txnId,
      Pointer<Uint8> key,
      int keyLen,
      Pointer<PhoenixBuffer> out,
    );

/// Native signature for `DeleteNative`.
typedef DeleteNative =
    Int32 Function(
      Pointer<PhoenixDB> handle,
      Uint64 txnId,
      Pointer<Uint8> key,
      Size keyLen,
    );

/// Dart signature for `DeleteDart`.
typedef DeleteDart =
    int Function(
      Pointer<PhoenixDB> handle,
      int txnId,
      Pointer<Uint8> key,
      int keyLen,
    );

/// Native signature for `BufferFreeNative`.
typedef BufferFreeNative = Void Function(Pointer<PhoenixBuffer> buf);

/// Dart signature for `BufferFreeDart`.
typedef BufferFreeDart = void Function(Pointer<PhoenixBuffer> buf);

/// Native signature for `StringFreeNative`.
typedef StringFreeNative = Void Function(Pointer<Utf8> s);

/// Dart signature for `StringFreeDart`.
typedef StringFreeDart = void Function(Pointer<Utf8> s);

/// Native signature for `LastErrorNative`.
typedef LastErrorNative = Pointer<Utf8> Function();

/// Dart signature for `LastErrorDart`.
typedef LastErrorDart = Pointer<Utf8> Function();

/// Native signature for `MaintenanceNative`.
typedef MaintenanceNative = Int32 Function(Pointer<PhoenixDB> handle);

/// Dart signature for `MaintenanceDart`.
typedef MaintenanceDart = int Function(Pointer<PhoenixDB> handle);

/// Native signature for `CountNative`.
typedef CountNative =
    Int32 Function(Pointer<PhoenixDB> handle, Pointer<Uint64> outLen);

/// Dart signature for `CountDart`.
typedef CountDart =
    int Function(Pointer<PhoenixDB> handle, Pointer<Uint64> outLen);

/// Native signature for `AbiVersionNative`.
typedef AbiVersionNative = Uint32 Function();

/// Dart signature for `AbiVersionDart`.
typedef AbiVersionDart = int Function();

/// Native signature for `LimitNative`.
typedef LimitNative = Size Function();

/// Dart signature for `LimitDart`.
typedef LimitDart = int Function();

/// Native signature for `phoenix_sql_query`.
typedef SqlQueryNative =
    Int32 Function(
      Pointer<PhoenixDB> handle,
      Pointer<Utf8> sql,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Dart signature for `phoenix_sql_query`.
typedef SqlQueryDart =
    int Function(
      Pointer<PhoenixDB> handle,
      Pointer<Utf8> sql,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Native signature for `phoenix_has_sql`.
typedef HasSqlNative = Int32 Function();

/// Dart signature for `phoenix_has_sql`.
typedef HasSqlDart = int Function();

// ---------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------

/// ABI version this Dart package was written against.
///
/// Bumped to 2 in PhoenixDB 2.0, which adds `phoenix_sql_query` and
/// `phoenix_has_sql`. The change is additive — every v1 entry point keeps its
/// signature — but the version guard is exact so a stale native library is
/// reported at load time rather than as a missing-symbol crash later.
const int kExpectedAbiVersion = 2;

/// Thrown when the native library cannot be located or is incompatible.
class PhoenixLoadException implements Exception {
  /// Describes what went wrong.
  final String message;

  /// Creates a load failure with [message].
  const PhoenixLoadException(this.message);

  @override
  String toString() => 'PhoenixLoadException: $message';
}

/// Platform-specific file name of the native library.
String get defaultLibraryName {
  if (Platform.isWindows) return 'phoenixdb.dll';
  if (Platform.isMacOS) return 'libphoenixdb.dylib';
  return 'libphoenixdb.so';
}

/// Rust target triples matching the current Dart ABI, most-preferred first.
///
/// Used to locate per-target binaries under `native/<triple>/`, which is where
/// `build.sh --all` and the CI release workflow install them.
///
/// Windows yields two candidates: CI builds the shipped library with the MSVC
/// toolchain, while a local `build.sh` on an MSYS/mingw host produces a `-gnu`
/// build. Both are ABI-compatible for a pure C interface, so either will do —
/// searching both is what lets a locally built library work alongside a
/// published one.
List<String> get currentTargetTriples {
  // `Abi.current()` reports the architecture Dart itself was built for, which
  // is the architecture the native library must match.
  final abi = Abi.current().toString();
  const map = <String, List<String>>{
    'linux_x64': ['x86_64-unknown-linux-gnu'],
    'linux_arm64': ['aarch64-unknown-linux-gnu'],
    'macos_x64': ['x86_64-apple-darwin'],
    'macos_arm64': ['aarch64-apple-darwin'],
    'windows_x64': ['x86_64-pc-windows-msvc', 'x86_64-pc-windows-gnu'],
    'windows_arm64': ['aarch64-pc-windows-msvc'],
  };
  return map[abi] ?? const <String>[];
}

/// The single most-preferred triple for the current ABI, or `null`.
///
/// Retained for callers that only need one name; prefer
/// [currentTargetTriples], which also covers the MSVC/gnu split on Windows.
String? get currentTargetTriple =>
    currentTargetTriples.isEmpty ? null : currentTargetTriples.first;

/// Directory of the installed `phoenixdb` package, or `null` when unavailable.
///
/// This is what makes the package work as an ordinary dependency. A consumer
/// running `dart pub get` gets the native library inside the package (in the
/// pub cache, or at a `path:` dependency's location) — never in their own
/// working directory — so resolving the package root is the only reliable way
/// to find it.
///
/// Two strategies, because neither works everywhere:
///
///  1. [Isolate.resolvePackageUriSync] — correct under `dart run` and
///     `dart test`, but returns `null` in the `flutter test` runner, which
///     serves `package:` URIs from its own in-memory resolver.
///  2. Reading `.dart_tool/package_config.json` — covers the Flutter test
///     runner and any other host that leaves the file on disk.
///
/// Returns `null` in Flutter release builds, where the library is bundled with
/// the app and found on the loader path instead.
String? _packageRootDir() => _resolveViaIsolate() ?? _resolveViaPackageConfig();

/// Strategy 1: ask the Dart runtime to resolve our own `package:` URI.
String? _resolveViaIsolate() {
  try {
    final uri = Isolate.resolvePackageUriSync(
      Uri.parse('package:phoenixdb/phoenixdb.dart'),
    );
    if (uri == null || !uri.isScheme('file')) return null;
    // .../<package>/lib/phoenixdb.dart -> .../<package>
    final dir = File(uri.toFilePath()).parent.parent;
    return dir.existsSync() ? dir.path : null;
  } on UnsupportedError {
    return null; // no package resolution in this runtime
  } catch (_) {
    return null;
  }
}

/// Strategy 2: read `package_config.json`, walking up from the current
/// directory.
///
/// The Flutter test runner intercepts `package:` resolution, so strategy 1
/// fails there even though the file is present on disk.
String? _resolveViaPackageConfig() {
  try {
    var dir = Directory.current;
    // A handful of levels is plenty: the file sits at the project root, and
    // tests run from the project root or one directory below it.
    for (var i = 0; i < 6; i++) {
      final file = File('${dir.path}/.dart_tool/package_config.json');
      if (file.existsSync()) {
        final root = _packageRootFromConfig(file);
        if (root != null) return root;
      }
      final parent = dir.parent;
      if (parent.path == dir.path) break; // reached the filesystem root
      dir = parent;
    }
  } catch (_) {
    // Fall through: the remaining search paths still apply.
  }
  return null;
}

/// Extracts phoenixdb's root directory from a `package_config.json` file.
String? _packageRootFromConfig(File file) {
  // Matched by hand rather than parsed: tolerates an unexpected shape without
  // throwing, and the entry is a flat object so a brace-free match is exact.
  final entry = RegExp(
    r'\{[^{}]*"name"\s*:\s*"phoenixdb"[^{}]*\}',
  ).firstMatch(file.readAsStringSync())?.group(0);
  if (entry == null) return null;

  final rootUri = RegExp(
    r'"rootUri"\s*:\s*"([^"]+)"',
  ).firstMatch(entry)?.group(1);
  if (rootUri == null) return null;

  // rootUri is relative to the .dart_tool directory unless it is absolute.
  final base = Uri.file('${file.parent.path}/', windows: Platform.isWindows);
  final resolved = base.resolve(rootUri);
  if (!resolved.isScheme('file')) return null;

  final dir = Directory(resolved.toFilePath());
  return dir.existsSync() ? dir.path : null;
}

/// Candidate paths searched when no explicit path is supplied.
///
/// Ordered cheapest-and-most-specific first: the per-target directory beats the
/// flat one, so a multi-platform checkout picks the right architecture instead
/// of a stale host build.
List<String> _searchPaths(String name) {
  // On mobile the library ships inside the app and there is no useful
  // filesystem to search:
  //   * Android - the .so from jniLibs/<abi>/ is already on the linker path,
  //     so it must be opened by bare name.
  //   * iOS - the Rust code is statically linked into the app executable, so
  //     the process itself provides the symbols; see [PhoenixBindings.load].
  if (Platform.isAndroid) return <String>[name];
  if (Platform.isIOS) return const <String>[];

  final script = Platform.script.toFilePath();
  final root = script.isEmpty
      ? Directory.current.path
      : File(script).parent.path;
  final cwd = Directory.current.path;
  final triples = currentTargetTriples;
  // Where this package is actually installed. For an ordinary consumer this
  // is the only location that holds the shipped binaries.
  final pkg = _packageRootDir();

  return <String>[
    // 1. Inside the installed package: correct for `dart pub get` consumers,
    //    whether the dependency came from the pub cache or a `path:` entry.
    if (pkg != null) ...[
      for (final t in triples) '$pkg/native/$t/$name',
      '$pkg/native/$name',
    ],

    // 2. Relative to the current directory and the running script: covers
    //    development inside this repo and apps that vendor the library.
    for (final t in triples) ...[
      'native/$t/$name',
      '$cwd/native/$t/$name',
      '$root/native/$t/$name',
      '$root/../native/$t/$name',
    ],

    // 3. The system loader path (LD_LIBRARY_PATH, PATH, DYLD_*, rpath).
    name,

    // 4. Flat native/ directory, for single-platform checkouts.
    'native/$name',
    '$cwd/native/$name',
    '$root/native/$name',
    '$root/../native/$name',

    // 5. Cargo output, so a fresh `cargo build` works without an install step.
    if (pkg != null) ...[
      '$pkg/rust/target/release/$name',
      '$pkg/rust/target/debug/$name',
    ],
    'rust/target/release/$name',
    'rust/target/debug/$name',
    '$cwd/rust/target/release/$name',
    '$cwd/rust/target/debug/$name',
    for (final t in triples) ...[
      'rust/target/$t/release/$name',
      '$cwd/rust/target/$t/release/$name',
    ],
  ];
}

/// Resolved, lazily-initialised bindings to the native library.
///
/// Each isolate that touches the database creates its own [PhoenixBindings];
/// `DynamicLibrary.open` is cheap after the first load because the OS caches
/// the mapping.
class PhoenixBindings {
  /// The loaded dynamic library.
  final DynamicLibrary library;

  /// Opens a database file.
  final OpenDart open;

  /// Closes a database handle.
  final CloseDart close;

  /// Begins a transaction.
  final BeginTxnDart beginTxn;

  /// Commits a transaction.
  final TxnOpDart commitTxn;

  /// Rolls a transaction back.
  final TxnOpDart rollbackTxn;

  /// Inserts a key/value pair inside a transaction.
  final InsertDart insert;

  /// Inserts a key/value pair in an implicit transaction.
  final PutAutoDart putAuto;

  /// Reads a key.
  final GetDart get;

  /// Deletes a key.
  final DeleteDart delete;

  /// Releases a value buffer.
  final BufferFreeDart bufferFree;

  /// Releases an error string.
  final StringFreeDart stringFree;

  /// Retrieves the last error message for the calling thread.
  final LastErrorDart lastError;

  /// Merges, flushes and truncates the WAL.
  final MaintenanceDart checkpoint;

  /// Flushes dirty pages.
  final MaintenanceDart flush;

  /// Verifies checksums and tree invariants.
  final MaintenanceDart verify;

  /// Counts visible keys.
  final CountDart count;

  /// Native ABI version.
  final AbiVersionDart abiVersion;

  /// Executes one SQL statement, yielding a JSON result document.
  final SqlQueryDart sqlQuery;

  /// Whether the native build includes the SQL layer.
  final HasSqlDart hasSql;

  /// Maximum key length accepted by the native layer.
  final LimitDart maxKeyLen;

  /// Maximum value length accepted by the native layer.
  final LimitDart maxValueLen;

  /// Pointer to `phoenix_buffer_free`, for use with [NativeFinalizer].
  final Pointer<NativeFunction<BufferFreeNative>> bufferFreePtr;

  /// Pointer to `phoenix_close`, for use with [NativeFinalizer].
  final Pointer<NativeFunction<CloseNative>> closePtr;

  PhoenixBindings._(this.library)
    : open = library.lookupFunction<OpenNative, OpenDart>('phoenix_open'),
      close = library.lookupFunction<CloseNative, CloseDart>('phoenix_close'),
      beginTxn = library.lookupFunction<BeginTxnNative, BeginTxnDart>(
        'phoenix_begin_txn',
      ),
      commitTxn = library.lookupFunction<TxnOpNative, TxnOpDart>(
        'phoenix_commit_txn',
      ),
      rollbackTxn = library.lookupFunction<TxnOpNative, TxnOpDart>(
        'phoenix_rollback_txn',
      ),
      insert = library.lookupFunction<InsertNative, InsertDart>(
        'phoenix_insert',
      ),
      putAuto = library.lookupFunction<PutAutoNative, PutAutoDart>(
        'phoenix_put_auto',
      ),
      get = library.lookupFunction<GetNative, GetDart>('phoenix_get'),
      delete = library.lookupFunction<DeleteNative, DeleteDart>(
        'phoenix_delete',
      ),
      bufferFree = library.lookupFunction<BufferFreeNative, BufferFreeDart>(
        'phoenix_buffer_free',
      ),
      stringFree = library.lookupFunction<StringFreeNative, StringFreeDart>(
        'phoenix_string_free',
      ),
      lastError = library.lookupFunction<LastErrorNative, LastErrorDart>(
        'phoenix_last_error',
      ),
      checkpoint = library.lookupFunction<MaintenanceNative, MaintenanceDart>(
        'phoenix_checkpoint',
      ),
      flush = library.lookupFunction<MaintenanceNative, MaintenanceDart>(
        'phoenix_flush',
      ),
      verify = library.lookupFunction<MaintenanceNative, MaintenanceDart>(
        'phoenix_verify',
      ),
      count = library.lookupFunction<CountNative, CountDart>('phoenix_count'),
      abiVersion = library.lookupFunction<AbiVersionNative, AbiVersionDart>(
        'phoenix_abi_version',
      ),
      sqlQuery = library.lookupFunction<SqlQueryNative, SqlQueryDart>(
        'phoenix_sql_query',
      ),
      hasSql = library.lookupFunction<HasSqlNative, HasSqlDart>(
        'phoenix_has_sql',
      ),
      maxKeyLen = library.lookupFunction<LimitNative, LimitDart>(
        'phoenix_max_key_len',
      ),
      maxValueLen = library.lookupFunction<LimitNative, LimitDart>(
        'phoenix_max_value_len',
      ),
      bufferFreePtr = library.lookup<NativeFunction<BufferFreeNative>>(
        'phoenix_buffer_free',
      ),
      closePtr = library.lookup<NativeFunction<CloseNative>>('phoenix_close');

  /// Loads the native library, searching well-known locations.
  ///
  /// Pass [path] to bypass the search entirely. Throws [PhoenixLoadException]
  /// when nothing loadable is found or when the native ABI version does not
  /// match [kExpectedAbiVersion].
  factory PhoenixBindings.load({String? path}) {
    final name = path ?? defaultLibraryName;

    // iOS links the Rust archive statically into the app executable, so there
    // is no separate library file to open; the symbols live in the process.
    if (path == null && Platform.isIOS) {
      final bindings = PhoenixBindings._(DynamicLibrary.process());
      final version = bindings.abiVersion();
      if (version != kExpectedAbiVersion) {
        throw PhoenixLoadException(
          'ABI mismatch: native reports $version, package expects '
          '$kExpectedAbiVersion. Rebuild the native library.',
        );
      }
      return bindings;
    }

    final attempts = path != null ? <String>[path] : _searchPaths(name);
    final failures = <String>[];

    for (final candidate in attempts) {
      try {
        final lib = DynamicLibrary.open(candidate);
        final bindings = PhoenixBindings._(lib);
        final version = bindings.abiVersion();
        if (version != kExpectedAbiVersion) {
          throw PhoenixLoadException(
            'ABI mismatch in "$candidate": native reports $version, '
            'package expects $kExpectedAbiVersion. Rebuild the native library.',
          );
        }
        return bindings;
      } on PhoenixLoadException {
        rethrow;
      } catch (e) {
        failures.add('  $candidate: $e');
      }
    }
    throw PhoenixLoadException(
      'Could not load $name. Run build.sh (or build.ps1) to compile the '
      'native library. Attempts:\n${failures.join('\n')}',
    );
  }
}
