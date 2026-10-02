/// Sanitize transcription output: remove null bytes, collapse whitespace,
/// strip leading/trailing punctuation artifacts from model output.
pub fn sanitize_utf8(text: &str) -> String {
    let mut result = text.replace('\0', "");
    // Collapse multiple spaces into one
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }
    // Remove leading punctuation (common model artifact)
    result = result
        .trim_start_matches(|c: char| c == ',' || c == '.' || c == ' ')
        .to_string();
    result = result.trim().to_string();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
