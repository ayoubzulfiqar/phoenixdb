/// Value types shared by the synchronous and asynchronous key/value APIs.
///
/// Everything here is plain data, so instances cross isolate boundaries
/// unchanged (the async API sends them to and from its worker isolate).
library;

import 'dart:convert';
import 'dart:typed_data';

/// Engine options applied when a database is opened.
///
/// Every field has an engine default. Options only take effect for the first
/// handle that opens a file in a process: later opens of the same path join
/// the running engine and share its options.
class PhoenixOptions {
  /// `fsync` the write-ahead log on every commit (the default).
  ///
  /// When `false`, a commit is handed to the operating system before it
  /// returns: it survives the app crashing, but not a power loss. Commits
  /// become much cheaper, which suits caches and derived data.
  final bool syncOnCommit;

  /// Page cache capacity in 4 KiB pages; `0` selects the engine default.
  final int cachePages;

  /// Write-ahead-log size, in bytes, that triggers an automatic checkpoint;
  /// `0` selects the engine default (4 MiB).
  final int checkpointBytes;

  /// Record engine spans, readable with `spans()`.
  final bool tracing;

  /// How full a B+Tree leaf may get before it splits, in `(0.5, 1.0]`.
  /// Lower values leave room for in-place growth at the cost of space.
  final double? fillFactor;

  /// Creates engine options; every argument is optional.
  const PhoenixOptions({
    this.syncOnCommit = true,
    this.cachePages = 0,
    this.checkpointBytes = 0,
    this.tracing = false,
    this.fillFactor,
  });

  /// Options tuned for throughput over power-loss durability.
  static const PhoenixOptions fast = PhoenixOptions(syncOnCommit: false);

  @override
  String toString() =>
      'PhoenixOptions(syncOnCommit: $syncOnCommit, cachePages: $cachePages, '
      'checkpointBytes: $checkpointBytes, tracing: $tracing, '
      'fillFactor: $fillFactor)';
}

/// One key/value pair returned by a scan.
class PhoenixEntry {
  /// Key bytes.
  final Uint8List key;

  /// Value bytes.
  final Uint8List value;

  /// Creates an entry.
  const PhoenixEntry(this.key, this.value);

  /// The key decoded as UTF-8.
  String get keyString => utf8.decode(key);

  /// The value decoded as UTF-8.
  String get valueString => utf8.decode(value);

  @override
  String toString() {
    String show(Uint8List b) {
      try {
        return '"${utf8.decode(b)}"';
      } on FormatException {
        return '${b.length} bytes';
      }
    }

    return 'PhoenixEntry(${show(key)}: ${show(value)})';
  }
}

/// A set of writes applied atomically by `write`.
///
/// ```dart
/// final batch = WriteBatch()
///   ..put(utf8Key('user:1'), utf8Value('ada'))
///   ..put(utf8Key('user:2'), utf8Value('grace'))
///   ..deleteIfExists(utf8Key('user:0'));
/// db.write(batch);
/// ```
///
/// Either every write lands or none does. A strict [delete] of a missing key
/// fails the whole batch with a not-found error.
class WriteBatch {
  final BytesBuilder _ops = BytesBuilder(copy: false);
  int _length = 0;

  /// Creates an empty batch.
  WriteBatch();

  static const int _put = 1;
  static const int _delete = 2;
  static const int _deleteIfExists = 3;

  /// Number of staged writes.
  int get length => _length;

  /// True when nothing is staged.
  bool get isEmpty => _length == 0;

  static Uint8List _u32(int v) =>
      Uint8List(4)..buffer.asByteData().setUint32(0, v, Endian.little);

  /// Inserts or replaces [key].
  void put(Uint8List key, Uint8List value) {
    _ops
      ..addByte(_put)
      ..add(_u32(key.length))
      ..add(Uint8List.fromList(key))
      ..add(_u32(value.length))
      ..add(Uint8List.fromList(value));
    _length++;
  }

  /// Deletes [key]; the whole batch fails if the key does not exist.
  void delete(Uint8List key) => _del(_delete, key);

  /// Deletes [key] when present; a missing key is not an error.
  void deleteIfExists(Uint8List key) => _del(_deleteIfExists, key);

  void _del(int op, Uint8List key) {
    _ops
      ..addByte(op)
      ..add(_u32(key.length))
      ..add(Uint8List.fromList(key));
    _length++;
  }

  /// The wire encoding consumed by `phoenix_write_batch`.
  Uint8List toBytes() => _ops.toBytes();
}

/// Runtime statistics of an open database.
class PhoenixStats {
  /// Pages allocated in the file.
  final int pageCount;

  /// Transactions currently open.
  final int activeTransactions;

  /// Keys with committed versions not yet merged into the B+Tree.
  final int pendingKeys;

  /// Current write-ahead-log size in bytes.
  final int walBytes;

  /// Latest commit timestamp.
  final int commitTimestamp;

  /// Every version at or below this timestamp is in the durable tree.
  final int treeTimestamp;

  /// Page reads served from memory.
  final int cacheHits;

  /// Page reads that had to decode a page from the file.
  final int cacheMisses;

  /// Creates a statistics snapshot.
  const PhoenixStats({
    required this.pageCount,
    required this.activeTransactions,
    required this.pendingKeys,
    required this.walBytes,
    required this.commitTimestamp,
    required this.treeTimestamp,
    required this.cacheHits,
    required this.cacheMisses,
  });

  /// Size of the database file's allocated pages, in bytes.
  int get fileBytes => pageCount * 4096;

  /// Fraction of page reads served from memory, `0` when nothing was read.
  double get cacheHitRatio {
    final total = cacheHits + cacheMisses;
    return total == 0 ? 0 : cacheHits / total;
  }

  @override
  String toString() =>
      'PhoenixStats(pages: $pageCount, activeTxns: $activeTransactions, '
      'pendingKeys: $pendingKeys, walBytes: $walBytes, '
      'commitTs: $commitTimestamp, treeTs: $treeTimestamp, '
      'cacheHitRatio: ${cacheHitRatio.toStringAsFixed(3)})';
}

/// Result of a full structural check of the B+Tree.
class PhoenixTreeReport {
  /// Levels from the root to the leaves.
  final int depth;

  /// Keys stored in the tree (excluding versions still only in memory).
  final int keys;

  /// Leaf pages.
  final int leafPages;

  /// Internal pages.
  final int internalPages;

  /// Pages holding large values.
  final int overflowPages;

  /// Pages on the free list, reusable without growing the file.
  final int freePages;

  /// Allocated pages that are neither in use nor free — space leaked by a
  /// crash. Harmless; `compact()` reclaims it.
  final int unreachablePages;

  /// Leaves below the minimum fill factor.
  final int underfullLeaves;

  /// Creates a report.
  const PhoenixTreeReport({
    required this.depth,
    required this.keys,
    required this.leafPages,
    required this.internalPages,
    required this.overflowPages,
    required this.freePages,
    required this.unreachablePages,
    required this.underfullLeaves,
  });

  @override
  String toString() =>
      'PhoenixTreeReport(depth: $depth, keys: $keys, leaves: $leafPages, '
      'internal: $internalPages, overflow: $overflowPages, free: $freePages, '
      'unreachable: $unreachablePages, underfull: $underfullLeaves)';
}

/// One engine span recorded while tracing was enabled.
class TraceSpan {
  /// Operation name, e.g. `commit` or `checkpoint`.
  final String name;

  /// Groups spans of one logical operation.
  final int traceId;

  /// This span's id.
  final int spanId;

  /// Enclosing span, if any.
  final int? parentId;

  /// Wall-clock duration.
  final Duration duration;

  /// Whether the operation failed.
  final bool error;

  /// Key/value annotations (txn id, write count, ...).
  final Map<String, String> attributes;

  /// Creates a span.
  const TraceSpan({
    required this.name,
    required this.traceId,
    required this.spanId,
    required this.parentId,
    required this.duration,
    required this.error,
    required this.attributes,
  });

  /// Parses one element of `phoenix_spans_json`.
  factory TraceSpan.fromJson(Map<String, Object?> json) => TraceSpan(
    name: json['name'] as String,
    traceId: json['trace_id'] as int,
    spanId: json['span_id'] as int,
    parentId: json['parent_id'] as int?,
    duration: Duration(microseconds: json['duration_us'] as int),
    error: json['error'] as bool,
    attributes: (json['attributes'] as Map<String, Object?>).map(
      (k, v) => MapEntry(k, v as String),
    ),
  );

  @override
  String toString() =>
      'TraceSpan($name, ${duration.inMicroseconds}us'
      '${error ? ', error' : ''}, $attributes)';
}

/// Smallest key greater than every key starting with [prefix], or `null`
/// when there is none (an empty prefix, or one made only of `0xFF` bytes).
Uint8List? prefixSuccessor(Uint8List prefix) {
  final end = Uint8List.fromList(prefix);
  for (var i = end.length - 1; i >= 0; i--) {
    if (end[i] < 0xFF) {
      end[i]++;
      return Uint8List.sublistView(end, 0, i + 1);
    }
  }
  return null;
}

/// Decodes a `[u32 key_len][key][u32 value_len][value]…` scan page.
List<PhoenixEntry> decodeEntries(Uint8List bytes) {
  final out = <PhoenixEntry>[];
  final view = ByteData.sublistView(bytes);
  var at = 0;
  Uint8List next() {
    if (at + 4 > bytes.length) {
      throw const FormatException('scan page truncated in a length prefix');
    }
    final n = view.getUint32(at, Endian.little);
    at += 4;
    if (at + n > bytes.length) {
      throw const FormatException('scan page truncated in a key or value');
    }
    final item = Uint8List.fromList(Uint8List.sublistView(bytes, at, at + n));
    at += n;
    return item;
  }

  while (at < bytes.length) {
    final key = next();
    out.add(PhoenixEntry(key, next()));
  }
  return out;
}
