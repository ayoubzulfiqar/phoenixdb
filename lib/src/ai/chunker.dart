/// Splitting long documents into retrieval-sized chunks.
library;

/// A slice of a document.
class TextChunk {
  /// Position of this chunk in the document.
  final int index;

  /// The chunk text (an exact substring of the source, trimmed).
  final String text;

  /// Start offset in the source, in UTF-16 code units.
  final int start;

  /// End offset (exclusive) in the source.
  final int end;

  /// Creates a chunk.
  const TextChunk(this.index, this.text, this.start, this.end);

  @override
  String toString() => 'TextChunk($index, $start..$end, ${text.length} chars)';
}

/// Splits text into chunks of at most [chunkSize] characters, preferring
/// natural boundaries: paragraphs, then lines, then sentences, then words.
///
/// Each chunk ends at the coarsest boundary found in the back half of its
/// window, and the next chunk starts about [overlap] characters earlier
/// (snapped to the start of a word), so a sentence cut by a boundary still
/// appears intact in one of the two. Offsets always refer to the original
/// text, so a hit can be highlighted in place.
class TextChunker {
  /// Longest chunk, in characters.
  final int chunkSize;

  /// Characters of context repeated between consecutive chunks.
  final int overlap;

  /// Boundaries to cut at, coarsest first. When none fits, a chunk is cut
  /// at [chunkSize] (never inside a surrogate pair).
  final List<String> separators;

  /// Creates a chunker.
  TextChunker({
    this.chunkSize = 1000,
    this.overlap = 150,
    this.separators = const [
      '\n\n',
      '\n',
      '. ',
      '? ',
      '! ',
      '。',
      '？',
      '！',
      '; ',
      ', ',
      ' ',
    ],
  }) {
    if (chunkSize <= 0) {
      throw ArgumentError.value(chunkSize, 'chunkSize', 'must be positive');
    }
    if (overlap < 0 || overlap >= chunkSize) {
      throw ArgumentError.value(
        overlap,
        'overlap',
        'must be in [0, chunkSize)',
      );
    }
  }

  /// Splits [text]; whitespace-only input gives no chunks.
  List<TextChunk> split(String text) {
    final n = text.length;
    final chunks = <TextChunk>[];
    var start = _skipSpace(text, 0);
    while (start < n) {
      final end = start + chunkSize >= n ? n : _cut(text, start);
      var e = end;
      while (e > start && _isSpace(text.codeUnitAt(e - 1))) {
        e--;
      }
      if (e > start) {
        chunks.add(
          TextChunk(chunks.length, text.substring(start, e), start, e),
        );
      }
      if (end >= n) break;
      var next = end - overlap;
      next = next <= start ? end : _wordStart(text, next, end);
      start = _skipSpace(text, next);
    }
    return chunks;
  }

  /// Where to end the chunk starting at [start]: just after the coarsest
  /// separator that ends in the back half of the window.
  int _cut(String text, int start) {
    final limit = start + chunkSize;
    final floor = start + chunkSize ~/ 2;
    for (final sep in separators) {
      if (sep.isEmpty || sep.length > chunkSize) continue;
      final i = text.lastIndexOf(sep, limit - sep.length);
      if (i >= start && i + sep.length > floor) return i + sep.length;
    }
    var e = limit;
    // Never cut a surrogate pair in half.
    if (e - 1 > start && _isHighSurrogate(text.codeUnitAt(e - 1))) e--;
    return e;
  }

  /// The first word start at or after [from] and before [end], or [from]
  /// itself (adjusted off a surrogate pair) when the span has no spaces.
  static int _wordStart(String text, int from, int end) {
    for (var i = from; i < end; i++) {
      if (_isSpace(text.codeUnitAt(i))) return i + 1;
    }
    if (from > 0 && _isHighSurrogate(text.codeUnitAt(from - 1))) {
      return from + 1;
    }
    return from;
  }

  static int _skipSpace(String text, int i) {
    while (i < text.length && _isSpace(text.codeUnitAt(i))) {
      i++;
    }
    return i;
  }

  static bool _isHighSurrogate(int unit) => unit >= 0xd800 && unit <= 0xdbff;

  static bool _isSpace(int unit) =>
      unit == 0x20 ||
      unit == 0x0a ||
      unit == 0x0d ||
      unit == 0x09 ||
      unit == 0x0c ||
      unit == 0xa0 ||
      unit == 0x3000;
}
