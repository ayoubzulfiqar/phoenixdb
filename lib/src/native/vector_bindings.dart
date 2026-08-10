/// Low-level `dart:ffi` bindings to the PhoenixDB vector-search C ABI.
///
/// This file mirrors the `phoenix_vector_*` surface in
/// `rust/src/ffi/vector_ffi.rs` exactly. Nothing here is safe to call without
/// upholding the ownership rules documented on each function; use
/// [PhoenixVectorDB] from `package:phoenixdb/phoenixdb.dart` instead.
///
/// ## Ownership rules
///
/// * Vector buffers passed *in* are borrowed for the duration of the call. The
///   native side never retains them, so the caller may free them immediately
///   afterwards.
/// * Id strings written to `outIds` by `phoenix_vector_search` are allocated by
///   the native library and must be released with `phoenix_free_string_array`,
///   passing the same count. The array itself belongs to the caller.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../bindings.dart';

/// Similarity metric used to order neighbours.
///
/// The indices are the wire values `phoenix_vector_init` expects and must
/// never be reordered.
enum VectorMetric {
  /// Angular distance, `1 - cos(a, b)`, in `[0, 2]`.
  ///
  /// Scale-invariant, and the right default for text embeddings, which are
  /// almost always trained with a cosine objective.
  cosine(0, 'cosine'),

  /// Straight-line (L2) distance, in `[0, infinity)`.
  euclidean(1, 'euclidean'),

  /// Inner product. Unlike the other two, magnitude matters — use it only when
  /// vector length is meaningful, as in a MIPS recommender.
  dotProduct(2, 'dotProduct');

  /// Creates a metric with its native [code] and human-readable [label].
  const VectorMetric(this.code, this.label);

  /// Value passed to `phoenix_vector_init`.
  final int code;

  /// Human-readable name, used in diagnostics.
  final String label;
}

/// Opaque vector-engine handle. Only ever held as a `Pointer<PhoenixVectorDB>`.
final class PhoenixVectorEngine extends Opaque {}

// ---------------------------------------------------------------------------
// Native function typedefs
// ---------------------------------------------------------------------------

/// Native signature for `phoenix_vector_init`.
typedef VectorInitNative =
    Int32 Function(
      Pointer<Utf8> path,
      Size dim,
      Uint8 metric,
      Size maxElements,
      Pointer<Pointer<PhoenixVectorEngine>> outHandle,
    );

/// Dart signature for `phoenix_vector_init`.
typedef VectorInitDart =
    int Function(
      Pointer<Utf8> path,
      int dim,
      int metric,
      int maxElements,
      Pointer<Pointer<PhoenixVectorEngine>> outHandle,
    );

/// Native signature for `phoenix_vector_free`.
typedef VectorFreeNative = Void Function(Pointer<PhoenixVectorEngine> handle);

/// Dart signature for `phoenix_vector_free`.
typedef VectorFreeDart = void Function(Pointer<PhoenixVectorEngine> handle);

/// Native signature for `phoenix_vector_insert`.
typedef VectorInsertNative =
    Int32 Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Utf8> id,
      Pointer<Float> vecPtr,
      Size vecLen,
    );

/// Dart signature for `phoenix_vector_insert`.
typedef VectorInsertDart =
    int Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Utf8> id,
      Pointer<Float> vecPtr,
      int vecLen,
    );

/// Native signature for `phoenix_vector_search`.
typedef VectorSearchNative =
    Int32 Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Float> queryPtr,
      Size queryLen,
      Size k,
      Size ef,
      Pointer<Pointer<Utf8>> outIds,
      Pointer<Float> outScores,
      Pointer<Size> outCount,
    );

/// Dart signature for `phoenix_vector_search`.
typedef VectorSearchDart =
    int Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Float> queryPtr,
      int queryLen,
      int k,
      int ef,
      Pointer<Pointer<Utf8>> outIds,
      Pointer<Float> outScores,
      Pointer<Size> outCount,
    );

/// Native signature for `phoenix_vector_get`.
typedef VectorGetNative =
    Int32 Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Utf8> id,
      Pointer<Float> outVec,
      Size outLen,
    );

/// Dart signature for `phoenix_vector_get`.
typedef VectorGetDart =
    int Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Utf8> id,
      Pointer<Float> outVec,
      int outLen,
    );

/// Native signature for `phoenix_vector_remove`.
typedef VectorRemoveNative =
    Int32 Function(Pointer<PhoenixVectorEngine> handle, Pointer<Utf8> id);

/// Dart signature for `phoenix_vector_remove`.
typedef VectorRemoveDart =
    int Function(Pointer<PhoenixVectorEngine> handle, Pointer<Utf8> id);

/// Native signature for `phoenix_vector_contains`.
typedef VectorContainsNative =
    Int32 Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Utf8> id,
      Pointer<Int32> outPresent,
    );

/// Dart signature for `phoenix_vector_contains`.
typedef VectorContainsDart =
    int Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Utf8> id,
      Pointer<Int32> outPresent,
    );

/// Native signature for `phoenix_vector_save`.
typedef VectorSaveNative =
    Int32 Function(Pointer<PhoenixVectorEngine> handle, Pointer<Utf8> path);

/// Dart signature for `phoenix_vector_save`.
typedef VectorSaveDart =
    int Function(Pointer<PhoenixVectorEngine> handle, Pointer<Utf8> path);

/// Native signature for `phoenix_vector_flush`.
typedef VectorFlushNative = Int32 Function(Pointer<PhoenixVectorEngine> handle);

/// Dart signature for `phoenix_vector_flush`.
typedef VectorFlushDart = int Function(Pointer<PhoenixVectorEngine> handle);

/// Native signature for `phoenix_vector_compact`.
typedef VectorCompactNative =
    Int32 Function(Pointer<PhoenixVectorEngine> handle, Pointer<Size> outReclaimed);

/// Dart signature for `phoenix_vector_compact`.
typedef VectorCompactDart =
    int Function(Pointer<PhoenixVectorEngine> handle, Pointer<Size> outReclaimed);

/// Native signature for `phoenix_vector_count` and `phoenix_vector_dim`.
typedef VectorSizeQueryNative =
    Int32 Function(Pointer<PhoenixVectorEngine> handle, Pointer<Size> out);

/// Dart signature for `phoenix_vector_count` and `phoenix_vector_dim`.
typedef VectorSizeQueryDart =
    int Function(Pointer<PhoenixVectorEngine> handle, Pointer<Size> out);

/// Native signature for `phoenix_vector_stats`.
typedef VectorStatsNative =
    Int32 Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Size> outLive,
      Pointer<Size> outTotal,
      Pointer<Size> outDeleted,
    );

/// Dart signature for `phoenix_vector_stats`.
typedef VectorStatsDart =
    int Function(
      Pointer<PhoenixVectorEngine> handle,
      Pointer<Size> outLive,
      Pointer<Size> outTotal,
      Pointer<Size> outDeleted,
    );

/// Native signature for `phoenix_free_string_array`.
typedef FreeStringArrayNative =
    Void Function(Pointer<Pointer<Utf8>> ptrs, Size len);

/// Dart signature for `phoenix_free_string_array`.
typedef FreeStringArrayDart =
    void Function(Pointer<Pointer<Utf8>> ptrs, int len);

/// Native signature for `phoenix_vector_last_error`.
typedef VectorLastErrorNative = Pointer<Utf8> Function();

/// Dart signature for `phoenix_vector_last_error`.
typedef VectorLastErrorDart = Pointer<Utf8> Function();

/// Native signature for `phoenix_vector_kernel`.
typedef VectorKernelNative = Pointer<Utf8> Function();

/// Dart signature for `phoenix_vector_kernel`.
typedef VectorKernelDart = Pointer<Utf8> Function();

/// Native signature for the `phoenix_vector_max_*` limit accessors.
typedef VectorLimitNative = Size Function();

/// Dart signature for the `phoenix_vector_max_*` limit accessors.
typedef VectorLimitDart = int Function();

/// Native signature for `phoenix_has_vector`.
typedef HasVectorNative = Int32 Function();

/// Dart signature for `phoenix_has_vector`.
typedef HasVectorDart = int Function();

/// Resolved bindings to the vector-search half of the native library.
///
/// Loaded from the same [DynamicLibrary] as [PhoenixBindings], so a process
/// that opens both a key/value database and a vector index pays for one
/// library load, not two.
class PhoenixVectorBindings {
  /// The loaded dynamic library.
  final DynamicLibrary library;

  /// Creates or opens a vector index.
  final VectorInitDart init;

  /// Saves and destroys an index, freeing the handle.
  final VectorFreeDart free;

  /// Inserts or replaces a vector.
  final VectorInsertDart insert;

  /// Searches for the k nearest neighbours.
  final VectorSearchDart search;

  /// Fetches a stored vector by id.
  final VectorGetDart get;

  /// Removes a vector by id.
  final VectorRemoveDart remove;

  /// Tests whether an id is present and live.
  final VectorContainsDart contains;

  /// Syncs and writes the graph snapshot.
  final VectorSaveDart save;

  /// Syncs without writing a snapshot.
  final VectorFlushDart flush;

  /// Rewrites the index without tombstoned records.
  final VectorCompactDart compact;

  /// Number of live vectors.
  final VectorSizeQueryDart count;

  /// Dimensionality of the index.
  final VectorSizeQueryDart dim;

  /// Live, total and deleted record counts.
  final VectorStatsDart stats;

  /// Releases an array of native id strings.
  final FreeStringArrayDart freeStringArray;

  /// Releases a single native string, such as an error message.
  ///
  /// Shared with the key/value surface: `phoenix_string_free` is one function
  /// in one library, so a message from either side is released the same way.
  final StringFreeDart stringFree;

  /// Last vector error on the calling thread.
  final VectorLastErrorDart lastError;

  /// Name of the SIMD kernel this CPU selected.
  final VectorKernelDart kernel;

  /// Largest dimensionality the native layer accepts.
  final VectorLimitDart maxDim;

  /// Largest `k` a single search may request.
  final VectorLimitDart maxK;

  /// Largest vector id, in bytes.
  final VectorLimitDart maxIdLen;

  /// Whether the native build includes the vector engine.
  final HasVectorDart hasVector;

  /// Pointer to `phoenix_vector_free`, for use with [NativeFinalizer].
  final Pointer<NativeFunction<VectorFreeNative>> freePtr;

  PhoenixVectorBindings._(this.library)
    : init = library.lookupFunction<VectorInitNative, VectorInitDart>(
        'phoenix_vector_init',
      ),
      free = library.lookupFunction<VectorFreeNative, VectorFreeDart>(
        'phoenix_vector_free',
      ),
      insert = library.lookupFunction<VectorInsertNative, VectorInsertDart>(
        'phoenix_vector_insert',
      ),
      search = library.lookupFunction<VectorSearchNative, VectorSearchDart>(
        'phoenix_vector_search',
      ),
      get = library.lookupFunction<VectorGetNative, VectorGetDart>(
        'phoenix_vector_get',
      ),
      remove = library.lookupFunction<VectorRemoveNative, VectorRemoveDart>(
        'phoenix_vector_remove',
      ),
      contains = library
          .lookupFunction<VectorContainsNative, VectorContainsDart>(
            'phoenix_vector_contains',
          ),
      save = library.lookupFunction<VectorSaveNative, VectorSaveDart>(
        'phoenix_vector_save',
      ),
      flush = library.lookupFunction<VectorFlushNative, VectorFlushDart>(
        'phoenix_vector_flush',
      ),
      compact = library.lookupFunction<VectorCompactNative, VectorCompactDart>(
        'phoenix_vector_compact',
      ),
      count = library
          .lookupFunction<VectorSizeQueryNative, VectorSizeQueryDart>(
            'phoenix_vector_count',
          ),
      dim = library.lookupFunction<VectorSizeQueryNative, VectorSizeQueryDart>(
        'phoenix_vector_dim',
      ),
      stats = library.lookupFunction<VectorStatsNative, VectorStatsDart>(
        'phoenix_vector_stats',
      ),
      freeStringArray = library
          .lookupFunction<FreeStringArrayNative, FreeStringArrayDart>(
            'phoenix_free_string_array',
          ),
      stringFree = library.lookupFunction<StringFreeNative, StringFreeDart>(
        'phoenix_string_free',
      ),
      lastError = library
          .lookupFunction<VectorLastErrorNative, VectorLastErrorDart>(
            'phoenix_vector_last_error',
          ),
      kernel = library.lookupFunction<VectorKernelNative, VectorKernelDart>(
        'phoenix_vector_kernel',
      ),
      maxDim = library.lookupFunction<VectorLimitNative, VectorLimitDart>(
        'phoenix_vector_max_dim',
      ),
      maxK = library.lookupFunction<VectorLimitNative, VectorLimitDart>(
        'phoenix_vector_max_k',
      ),
      maxIdLen = library.lookupFunction<VectorLimitNative, VectorLimitDart>(
        'phoenix_vector_max_id_len',
      ),
      hasVector = library.lookupFunction<HasVectorNative, HasVectorDart>(
        'phoenix_has_vector',
      ),
      freePtr = library.lookup<NativeFunction<VectorFreeNative>>(
        'phoenix_vector_free',
      );

  /// Loads the vector bindings, reusing [PhoenixBindings]' library search.
  ///
  /// Pass [path] to bypass discovery entirely. Throws [PhoenixLoadException]
  /// when the library cannot be found, when its ABI version does not match, or
  /// when it was built without the vector engine — the last case is reported
  /// here rather than as a missing-symbol crash at first search.
  factory PhoenixVectorBindings.load({String? path}) {
    // Delegating to PhoenixBindings is deliberate: it already performs the
    // multi-strategy search and the exact ABI-version check, and both surfaces
    // live in one shared object.
    final base = PhoenixBindings.load(path: path);
    final bindings = PhoenixVectorBindings._(base.library);
    if (bindings.hasVector() == 0) {
      throw const PhoenixLoadException(
        'the native library was built without the vector search engine',
      );
    }
    return bindings;
  }

  /// Wraps an already-loaded library, for a caller that holds its own handle.
  factory PhoenixVectorBindings.fromLibrary(DynamicLibrary library) =>
      PhoenixVectorBindings._(library);
}
