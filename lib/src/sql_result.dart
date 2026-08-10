/// Typed results for SQL statements executed through [PhoenixDatabase.query].
library;

import 'dart:convert';

/// The outcome of one SQL statement.
///
/// Three shapes, matching the JSON the native layer returns:
///
/// * a `SELECT` yields [columns] and [rows];
/// * a mutation yields [affected];
/// * a schema change yields [detail].
///
/// Reading the wrong one is safe — [rows] is empty for a mutation, [affected]
/// is zero for a `SELECT` — so callers can branch on [isRows] or just read the
/// field they expect.
class SqlResult {
  /// Column names, in projection order. Empty unless [isRows].
  final List<String> columns;

  /// Result rows, each parallel to [columns]. Empty unless [isRows].
  ///
  /// Cells are `String`, `int`, `double`, or `null`, preserving the type the
  /// value was stored with.
  final List<List<Object?>> rows;

  /// Rows inserted, updated, or deleted. Zero for a `SELECT`.
  final int affected;

  /// Human-readable description of a schema change, or `null`.
  final String? detail;

  const SqlResult._({
    required this.columns,
    required this.rows,
    required this.affected,
    required this.detail,
  });

  /// Parses the JSON document produced by `phoenix_sql_query`.
  ///
  /// Throws [FormatException] when the document is not one of the three known
  /// shapes — a malformed result means the native and Dart sides disagree, and
  /// silently returning an empty result would hide that.
  factory SqlResult.fromJson(String source) {
    final Object? decoded = jsonDecode(source);
    if (decoded is! Map<String, Object?>) {
      throw FormatException('expected a JSON object', source);
    }
    switch (decoded['type']) {
      case 'rows':
        final rawColumns = decoded['columns'];
        final rawRows = decoded['rows'];
        if (rawColumns is! List || rawRows is! List) {
          throw FormatException('malformed rows result', source);
        }
        return SqlResult._(
          columns: rawColumns.map((c) => '$c').toList(growable: false),
          rows: rawRows
              .map((r) => (r as List).cast<Object?>().toList(growable: false))
              .toList(growable: false),
          affected: 0,
          detail: null,
        );
      case 'affected':
        final count = decoded['count'];
        if (count is! int) {
          throw FormatException('malformed affected result', source);
        }
        return SqlResult._(
          columns: const [],
          rows: const [],
          affected: count,
          detail: null,
        );
      case 'schema':
        return SqlResult._(
          columns: const [],
          rows: const [],
          affected: 0,
          detail: '${decoded['detail']}',
        );
      default:
        throw FormatException(
          'unknown result type: ${decoded['type']}',
          source,
        );
    }
  }

  /// True when this is a `SELECT` result.
  bool get isRows => columns.isNotEmpty || rows.isNotEmpty;

  /// True when the result set has no rows.
  bool get isEmpty => rows.isEmpty;

  /// Number of rows returned.
  int get length => rows.length;

  /// The first row, or `null` when the result is empty.
  ///
  /// Convenience for the common `SELECT ... WHERE id = ?` shape.
  List<Object?>? get firstOrNull => rows.isEmpty ? null : rows.first;

  /// The single cell of a one-row, one-column result, or `null`.
  ///
  /// Useful for scalar queries; returns `null` rather than throwing when the
  /// shape does not match, so a missing row and a null value read alike.
  Object? get scalar {
    if (rows.length != 1 || rows.first.length != 1) return null;
    return rows.first.first;
  }

  /// Looks a cell up by column name, matched case-insensitively.
  ///
  /// Returns `null` when the column or row does not exist.
  Object? cell(int row, String column) {
    if (row < 0 || row >= rows.length) return null;
    final idx = columns.indexWhere(
      (c) => c.toLowerCase() == column.toLowerCase(),
    );
    if (idx < 0 || idx >= rows[row].length) return null;
    return rows[row][idx];
  }

  /// The rows as maps keyed by column name.
  ///
  /// Costs an allocation per row, so prefer [rows] on hot paths.
  List<Map<String, Object?>> get asMaps => [
    for (final row in rows)
      {
        for (var i = 0; i < columns.length && i < row.length; i++)
          columns[i]: row[i],
      },
  ];

  /// Serialises back to the same JSON shape [SqlResult.fromJson] accepts.
  ///
  /// Used to carry a result across an isolate boundary, where only primitives
  /// and a few built-in types may be sent. Round-tripping through this and
  /// [SqlResult.fromJson] must be lossless.
  String toJsonString() {
    if (detail != null) {
      return jsonEncode({'type': 'schema', 'detail': detail});
    }
    if (!isRows) {
      return jsonEncode({'type': 'affected', 'count': affected});
    }
    return jsonEncode({'type': 'rows', 'columns': columns, 'rows': rows});
  }

  @override
  String toString() {
    if (detail != null) return 'SqlResult(schema: $detail)';
    if (!isRows) return 'SqlResult(affected: $affected)';
    return 'SqlResult(columns: $columns, rows: ${rows.length})';
  }
}
