/// Sanitize transcription output: remove null bytes, collapse whitespace,
/// strip leading/trailing punctuation artifacts from model output.
pub fn sanitize_utf8(text: &str) -> String {
    let mut result = text.replace('\0', "");
    // Collapse multiple spaces into one
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }
    // Remove leading punctuation (common model artifact)
    result = result.trim_start_matches(|c: char| c == ',' || c == '.' || c == ' ').to_string();
    result = result.trim().to_string();
    result
}

/// Find the best split point within `text[..max_len]` using priority:
/// 1. Last newline
/// 2. Last sentence boundary (`. `, `! `, `? `) — split after punctuation
/// 3. Last space
/// 4. Hard cut at max_len
fn find_split_point(text: &str, max_len: usize) -> usize {
    let boundary = floor_char_boundary(text, max_len);
    let window = &text[..boundary];

    if let Some(pos) = window.rfind('\n') {
        return pos;
    }

    for pat in &[". ", "! ", "? "] {
        if let Some(pos) = window.rfind(pat) {
            return pos + pat.len() - 1; // include punctuation, split before space
        }
    }

    if let Some(pos) = window.rfind(' ') {
        return pos;
    }

    boundary
}

/// Split text into chunks of at most `max_len` characters.
/// Split priority: newline > sentence boundary > word boundary > hard cut.
pub fn split_text(text: &str, max_len: usize) -> Vec<String> {
    if max_len == 0 || text.len() <= max_len {
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() {
            return Vec::new();
        }
        return vec![trimmed];
    }

    let mut chunks = Vec::new();
    let mut remaining = text;

    while remaining.len() > max_len {
        let split_at = find_split_point(remaining, max_len);
        let chunk = remaining[..split_at].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        remaining = &remaining[split_at..];
        // Skip leading whitespace/newline after split
        remaining = remaining.trim_start();
    }

    let tail = remaining.trim();
    if !tail.is_empty() {
        chunks.push(tail.to_string());
    }

    chunks
}

/// Safe char-boundary floor (stable alternative to nightly `floor_char_boundary`).
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_short_text() {
        assert_eq!(split_text("hello", 100), vec!["hello"]);
    }

    #[test]
    fn test_split_exact_fit() {
        let text = "exact";
        assert_eq!(split_text(text, text.len()), vec!["exact"]);
    }

    #[test]
    fn test_split_empty() {
        let result = split_text("", 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_zero_max_len() {
        assert_eq!(split_text("hello world", 0), vec!["hello world"]);
    }

    #[test]
    fn test_split_at_newline() {
        let result = split_text("line one\nline two", 12);
        assert_eq!(result, vec!["line one", "line two"]);
    }

    #[test]
    fn test_split_at_sentence() {
        let result = split_text("Hello world. Next sentence.", 15);
        assert_eq!(result, vec!["Hello world.", "Next sentence."]);
    }

    #[test]
    fn test_split_at_word() {
        let result = split_text("word1 word2 word3", 12);
        assert_eq!(result, vec!["word1 word2", "word3"]);
    }

    #[test]
    fn test_split_hard_cut() {
        let result = split_text("abcdefghij", 5);
        assert_eq!(result, vec!["abcde", "fghij"]);
    }

    #[test]
    fn test_sanitize_utf8_valid() {
        assert_eq!(sanitize_utf8("hello"), "hello");
    }

    #[test]
    fn test_sanitize_utf8_null_bytes() {
        assert_eq!(sanitize_utf8("hel\0lo"), "hello");
    }

    #[test]
    fn test_sanitize_utf8_cyrillic() {
        assert_eq!(sanitize_utf8("Привет мир"), "Привет мир");
    }

    #[test]
    fn test_sanitize_utf8_emoji() {
        assert_eq!(sanitize_utf8("Hello 🌍!"), "Hello 🌍!");
    }

    #[test]
    fn test_split_cyrillic() {
        // "Привет"=12b, " "=1b, "мир"=6b, " "=1b, "друг"=8b = 28 bytes
        // max_len=14: window="Привет м", splits at space (pos 12)
        // Must not panic on char boundaries
        let text = "Привет мир друг";
        let result = split_text(text, 14);
        assert_eq!(result, vec!["Привет", "мир", "друг"]);
        for chunk in &result {
            assert!(!chunk.is_empty());
        }
    }
}
