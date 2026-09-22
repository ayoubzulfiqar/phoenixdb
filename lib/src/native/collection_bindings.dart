/// Low-level `dart:ffi` bindings to the `phoenix_collection_*` C ABI.
///
/// Mirrors `rust/src/ffi/collection_ffi.rs`. Use [PhoenixCollection] from
/// `package:phoenixdb/phoenixdb.dart` instead of calling these directly.
///
/// Structured data crosses as JSON strings; embeddings cross as raw `float`
/// buffers. Strings written to an `out_json` parameter are allocated by the
/// native library and must be released with `phoenix_string_free`.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../bindings.dart';

/// Opaque collection handle.
final class PhoenixCollectionNative extends Opaque {}

/// Native signature for `phoenix_collection_open`.
typedef CollectionOpenNative =
    Int32 Function(
      Pointer<Utf8> path,
      Pointer<Utf8> optionsJson,
      Pointer<Pointer<PhoenixCollectionNative>> outHandle,
    );

/// Dart signature for `phoenix_collection_open`.
typedef CollectionOpenDart =
    int Function(
      Pointer<Utf8> path,
      Pointer<Utf8> optionsJson,
      Pointer<Pointer<PhoenixCollectionNative>> outHandle,
    );

/// Native signature for `phoenix_collection_close`.
typedef CollectionCloseNative =
    Void Function(Pointer<PhoenixCollectionNative> handle);

/// Dart signature for `phoenix_collection_close`.
typedef CollectionCloseDart =
    void Function(Pointer<PhoenixCollectionNative> handle);

/// Native signature for `phoenix_collection_upsert`.
typedef CollectionUpsertNative =
    Int32 Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> docsJson,
      Pointer<Float> vectors,
      Size vectorsLen,
    );

/// Dart signature for `phoenix_collection_upsert`.
typedef CollectionUpsertDart =
    int Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> docsJson,
      Pointer<Float> vectors,
      int vectorsLen,
    );

/// Native signature for `phoenix_collection_delete` and `_count`.
typedef CollectionJsonCountNative =
    Int32 Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> json,
      Pointer<Uint64> outCount,
    );

/// Dart signature for `phoenix_collection_delete` and `_count`.
typedef CollectionJsonCountDart =
    int Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> json,
      Pointer<Uint64> outCount,
    );

/// Native signature for `phoenix_collection_get`.
typedef CollectionGetNative =
    Int32 Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> id,
      Int32 withVector,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Dart signature for `phoenix_collection_get`.
typedef CollectionGetDart =
    int Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> id,
      int withVector,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Native signature for `phoenix_collection_search`.
typedef CollectionSearchNative =
    Int32 Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> requestJson,
      Pointer<Float> query,
      Size queryLen,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Dart signature for `phoenix_collection_search`.
typedef CollectionSearchDart =
    int Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> requestJson,
      Pointer<Float> query,
      int queryLen,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Native signature for `phoenix_collection_list`.
typedef CollectionListNative =
    Int32 Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> filterJson,
      Uint64 limit,
      Pointer<Utf8> after,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Dart signature for `phoenix_collection_list`.
typedef CollectionListDart =
    int Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Utf8> filterJson,
      int limit,
      Pointer<Utf8> after,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Native signature for `phoenix_collection_stats`.
typedef CollectionStatsNative =
    Int32 Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Dart signature for `phoenix_collection_stats`.
typedef CollectionStatsDart =
    int Function(
      Pointer<PhoenixCollectionNative> handle,
      Pointer<Pointer<Utf8>> outJson,
    );

/// Native signature for `phoenix_collection_flush`.
typedef CollectionFlushNative =
    Int32 Function(Pointer<PhoenixCollectionNative> handle);

/// Dart signature for `phoenix_collection_flush`.
typedef CollectionFlushDart =
    int Function(Pointer<PhoenixCollectionNative> handle);

/// Native signature for `phoenix_collection_open_count`.
typedef CollectionOpenCountNative = Uint32 Function();

/// Dart signature for `phoenix_collection_open_count`.
typedef CollectionOpenCountDart = int Function();

/// Resolved `phoenix_collection_*` functions.
class PhoenixCollectionBindings {
  /// The key/value bindings sharing this library (error channel, string
  /// release).
  final PhoenixBindings base;

  /// `phoenix_collection_open`.
  final CollectionOpenDart open;

  /// `phoenix_collection_close`.
  final CollectionCloseDart close;

  /// `phoenix_collection_upsert`.
  final CollectionUpsertDart upsert;

  /// `phoenix_collection_delete`.
  final CollectionJsonCountDart delete;

  /// `phoenix_collection_get`.
  final CollectionGetDart get;

  /// `phoenix_collection_search`.
  final CollectionSearchDart search;

  /// `phoenix_collection_count`.
  final CollectionJsonCountDart count;

  /// `phoenix_collection_list`.
  final CollectionListDart list;

  /// `phoenix_collection_stats`.
  final CollectionStatsDart stats;

  /// `phoenix_collection_flush`.
  final CollectionFlushDart flush;

  /// `phoenix_collection_open_count`.
  final CollectionOpenCountDart openCount;

  /// Pointer to `phoenix_collection_close`, for a [NativeFinalizer].
  final Pointer<NativeFunction<CollectionCloseNative>> closePtr;

  PhoenixCollectionBindings._(this.base)
    : open = base.library
          .lookupFunction<CollectionOpenNative, CollectionOpenDart>(
            'phoenix_collection_open',
          ),
      close = base.library
          .lookupFunction<CollectionCloseNative, CollectionCloseDart>(
            'phoenix_collection_close',
          ),
      upsert = base.library
          .lookupFunction<CollectionUpsertNative, CollectionUpsertDart>(
            'phoenix_collection_upsert',
          ),
      delete = base.library
          .lookupFunction<CollectionJsonCountNative, CollectionJsonCountDart>(
            'phoenix_collection_delete',
          ),
      get = base.library.lookupFunction<CollectionGetNative, CollectionGetDart>(
        'phoenix_collection_get',
      ),
      search = base.library
          .lookupFunction<CollectionSearchNative, CollectionSearchDart>(
            'phoenix_collection_search',
          ),
      count = base.library
          .lookupFunction<CollectionJsonCountNative, CollectionJsonCountDart>(
            'phoenix_collection_count',
          ),
      list = base.library
          .lookupFunction<CollectionListNative, CollectionListDart>(
            'phoenix_collection_list',
          ),
      stats = base.library
          .lookupFunction<CollectionStatsNative, CollectionStatsDart>(
            'phoenix_collection_stats',
          ),
      flush = base.library
          .lookupFunction<CollectionFlushNative, CollectionFlushDart>(
            'phoenix_collection_flush',
          ),
      openCount = base.library
          .lookupFunction<CollectionOpenCountNative, CollectionOpenCountDart>(
            'phoenix_collection_open_count',
          ),
      closePtr = base.library.lookup<NativeFunction<CollectionCloseNative>>(
        'phoenix_collection_close',
      );

  /// Loads the bindings through [PhoenixBindings.load]'s library search and
  /// ABI check.
  factory PhoenixCollectionBindings.load({String? path}) =>
      PhoenixCollectionBindings._(PhoenixBindings.load(path: path));
}
