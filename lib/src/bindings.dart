/// Low-level `dart:ffi` bindings to the PhoenixDB C ABI.
///
/// This file mirrors `native/include/phoenixdb.h` exactly. Nothing here is
/// safe to call without upholding the ownership rules documented on each
/// function; use the high-level API in `package:phoenixdb/phoenixdb.dart`
/// instead.
library;

import 'dart:ffi';
import 'dart:io';

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
typedef OpenNative = Int32 Function(
    Pointer<Utf8> path, Size cachePages, Pointer<Pointer<PhoenixDB>> outHandle);
/// Dart signature for `OpenDart`.
typedef OpenDart = int Function(
    Pointer<Utf8> path, int cachePages, Pointer<Pointer<PhoenixDB>> outHandle);

/// Native signature for `CloseNative`.
typedef CloseNative = Int32 Function(Pointer<PhoenixDB> handle);
/// Dart signature for `CloseDart`.
typedef CloseDart = int Function(Pointer<PhoenixDB> handle);

/// Native signature for `BeginTxnNative`.
typedef BeginTxnNative = Int32 Function(
    Pointer<PhoenixDB> handle, Int32 readOnly, Pointer<Uint64> outTxn);
/// Dart signature for `BeginTxnDart`.
typedef BeginTxnDart = int Function(
    Pointer<PhoenixDB> handle, int readOnly, Pointer<Uint64> outTxn);

/// Native signature for `TxnOpNative`.
typedef TxnOpNative = Int32 Function(Pointer<PhoenixDB> handle, Uint64 txnId);
/// Dart signature for `TxnOpDart`.
typedef TxnOpDart = int Function(Pointer<PhoenixDB> handle, int txnId);

/// Native signature for `InsertNative`.
typedef InsertNative = Int32 Function(Pointer<PhoenixDB> handle, Uint64 txnId,
    Pointer<Uint8> key, Size keyLen, Pointer<Uint8> value, Size valueLen);
/// Dart signature for `InsertDart`.
typedef InsertDart = int Function(Pointer<PhoenixDB> handle, int txnId,
    Pointer<Uint8> key, int keyLen, Pointer<Uint8> value, int valueLen);

/// Native signature for `PutAutoNative`.
typedef PutAutoNative = Int32 Function(Pointer<PhoenixDB> handle,
    Pointer<Uint8> key, Size keyLen, Pointer<Uint8> value, Size valueLen);
/// Dart signature for `PutAutoDart`.
typedef PutAutoDart = int Function(Pointer<PhoenixDB> handle,
    Pointer<Uint8> key, int keyLen, Pointer<Uint8> value, int valueLen);

/// Native signature for `GetNative`.
typedef GetNative = Int32 Function(Pointer<PhoenixDB> handle, Uint64 txnId,
    Pointer<Uint8> key, Size keyLen, Pointer<PhoenixBuffer> out);
/// Dart signature for `GetDart`.
typedef GetDart = int Function(Pointer<PhoenixDB> handle, int txnId,
    Pointer<Uint8> key, int keyLen, Pointer<PhoenixBuffer> out);

/// Native signature for `DeleteNative`.
typedef DeleteNative = Int32 Function(
    Pointer<PhoenixDB> handle, Uint64 txnId, Pointer<Uint8> key, Size keyLen);
/// Dart signature for `DeleteDart`.
typedef DeleteDart = int Function(
    Pointer<PhoenixDB> handle, int txnId, Pointer<Uint8> key, int keyLen);

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
typedef CountNative = Int32 Function(
    Pointer<PhoenixDB> handle, Pointer<Uint64> outLen);
/// Dart signature for `CountDart`.
typedef CountDart = int Function(
    Pointer<PhoenixDB> handle, Pointer<Uint64> outLen);

/// Native signature for `AbiVersionNative`.
typedef AbiVersionNative = Uint32 Function();
/// Dart signature for `AbiVersionDart`.
typedef AbiVersionDart = int Function();

/// Native signature for `LimitNative`.
typedef LimitNative = Size Function();
/// Dart signature for `LimitDart`.
typedef LimitDart = int Function();

// ---------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------

/// ABI version this Dart package was written against.
const int kExpectedAbiVersion = 1;

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

/// Rust target triple matching the current platform and CPU architecture.
///
/// Used to locate per-target binaries under `native/<triple>/`, which is where
/// `build.sh --all` installs cross-compiled libraries.
String? get currentTargetTriple {
  // `Abi.current()` reports the architecture Dart itself was built for, which
  // is the architecture the native library must match.
  final abi = Abi.current().toString();
  const map = <String, String>{
    'linux_x64': 'x86_64-unknown-linux-gnu',
    'linux_arm64': 'aarch64-unknown-linux-gnu',
    'macos_x64': 'x86_64-apple-darwin',
    'macos_arm64': 'aarch64-apple-darwin',
    'windows_x64': 'x86_64-pc-windows-gnu',
  };
  return map[abi];
}

/// Candidate paths searched when no explicit path is supplied.
///
/// Ordered cheapest-and-most-specific first: the per-target directory beats the
/// flat one, so a multi-platform checkout picks the right architecture instead
/// of a stale host build.
List<String> _searchPaths(String name) {
  final script = Platform.script.toFilePath();
  final root = script.isEmpty ? Directory.current.path : File(script).parent.path;
  final cwd = Directory.current.path;
  final triple = currentTargetTriple;

  return <String>[
    if (triple != null) ...[
      'native/$triple/$name',
      '$cwd/native/$triple/$name',
      '$root/native/$triple/$name',
      '$root/../native/$triple/$name',
    ],
    name, // system loader path
    'native/$name',
    '$cwd/native/$name',
    '$root/native/$name',
    '$root/../native/$name',
    'rust/target/release/$name',
    'rust/target/debug/$name',
    '$cwd/rust/target/release/$name',
    '$cwd/rust/target/debug/$name',
    if (triple != null) ...[
      'rust/target/$triple/release/$name',
      '$cwd/rust/target/$triple/release/$name',
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
        beginTxn = library
            .lookupFunction<BeginTxnNative, BeginTxnDart>('phoenix_begin_txn'),
        commitTxn =
            library.lookupFunction<TxnOpNative, TxnOpDart>('phoenix_commit_txn'),
        rollbackTxn = library
            .lookupFunction<TxnOpNative, TxnOpDart>('phoenix_rollback_txn'),
        insert =
            library.lookupFunction<InsertNative, InsertDart>('phoenix_insert'),
        putAuto = library
            .lookupFunction<PutAutoNative, PutAutoDart>('phoenix_put_auto'),
        get = library.lookupFunction<GetNative, GetDart>('phoenix_get'),
        delete =
            library.lookupFunction<DeleteNative, DeleteDart>('phoenix_delete'),
        bufferFree = library.lookupFunction<BufferFreeNative, BufferFreeDart>(
            'phoenix_buffer_free'),
        stringFree = library.lookupFunction<StringFreeNative, StringFreeDart>(
            'phoenix_string_free'),
        lastError = library
            .lookupFunction<LastErrorNative, LastErrorDart>('phoenix_last_error'),
        checkpoint = library
            .lookupFunction<MaintenanceNative, MaintenanceDart>('phoenix_checkpoint'),
        flush = library
            .lookupFunction<MaintenanceNative, MaintenanceDart>('phoenix_flush'),
        verify = library
            .lookupFunction<MaintenanceNative, MaintenanceDart>('phoenix_verify'),
        count = library.lookupFunction<CountNative, CountDart>('phoenix_count'),
        abiVersion = library
            .lookupFunction<AbiVersionNative, AbiVersionDart>('phoenix_abi_version'),
        maxKeyLen =
            library.lookupFunction<LimitNative, LimitDart>('phoenix_max_key_len'),
        maxValueLen =
            library.lookupFunction<LimitNative, LimitDart>('phoenix_max_value_len'),
        bufferFreePtr =
            library.lookup<NativeFunction<BufferFreeNative>>('phoenix_buffer_free'),
        closePtr = library.lookup<NativeFunction<CloseNative>>('phoenix_close');

  /// Loads the native library, searching well-known locations.
  ///
  /// Pass [path] to bypass the search entirely. Throws [PhoenixLoadException]
  /// when nothing loadable is found or when the native ABI version does not
  /// match [kExpectedAbiVersion].
  factory PhoenixBindings.load({String? path}) {
    final name = path ?? defaultLibraryName;
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
