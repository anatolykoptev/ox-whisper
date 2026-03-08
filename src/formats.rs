/// Subtitle format generators: SRT and WebVTT from word timestamps.

use crate::words::WordTimestamp;

const DEFAULT_MAX_WORDS: usize = 8;

pub(crate) fn format_srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

pub(crate) fn format_vtt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// Group words into subtitle segments of at most `max_words` words each.
/// Returns (start, end, text) tuples.
pub(crate) fn group_words(
    words: &[WordTimestamp],
    max_words: usize,
) -> Vec<(f32, f32, String)> {
    let limit = if max_words == 0 { DEFAULT_MAX_WORDS } else { max_words };
    let mut groups: Vec<(f32, f32, String)> = Vec::new();

    for chunk in words.chunks(limit) {
        if chunk.is_empty() {
            continue;
        }
        let start = chunk.first().unwrap().start;
        let end = chunk.last().unwrap().end;
        let text: Vec<&str> = chunk.iter().map(|w| w.word.as_str()).collect();
        groups.push((start, end, text.join(" ")));
    }

    groups
}

/// Generate an SRT subtitle string from word timestamps.
pub fn to_srt(words: &[WordTimestamp]) -> String {
    let groups = group_words(words, DEFAULT_MAX_WORDS);
    let mut out = String::new();

    for (i, (start, end, text)) in groups.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n",
            i + 1,
            format_srt_time(*start),
            format_srt_time(*end),
            text,
        ));
    }

    out
}

/// Generate a WebVTT subtitle string from word timestamps.
pub fn to_vtt(words: &[WordTimestamp]) -> String {
    let groups = group_words(words, DEFAULT_MAX_WORDS);
    let mut out = String::from("WEBVTT\n\n");

    for (i, (start, end, text)) in groups.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!(
            "{} --> {}\n{}\n",
            format_vtt_time(*start),
            format_vtt_time(*end),
            text,
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_word(word: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp { word: word.to_string(), start, end, confidence: None, speaker: None }
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(61.234), "00:01:01,234");
        assert_eq!(format_srt_time(3661.1), "01:01:01,100");
    }

    #[test]
    fn test_format_vtt_time() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
        assert_eq!(format_vtt_time(1.5), "00:00:01.500");
        assert_eq!(format_vtt_time(61.234), "00:01:01.234");
        assert_eq!(format_vtt_time(3661.1), "01:01:01.100");
    }

    #[test]
    fn test_group_words_basic() {
        let words: Vec<WordTimestamp> = (0..10)
            .map(|i| make_word(&format!("w{i}"), i as f32, i as f32 + 0.5))
            .collect();
        let groups = group_words(&words, 4);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].2, "w0 w1 w2 w3");
        assert_eq!(groups[1].2, "w4 w5 w6 w7");
        assert_eq!(groups[2].2, "w8 w9");
        assert!((groups[0].0 - 0.0).abs() < f32::EPSILON);
        assert!((groups[0].1 - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_group_words_default() {
        let words: Vec<WordTimestamp> = (0..9)
            .map(|i| make_word(&format!("w{i}"), i as f32, i as f32 + 0.5))
            .collect();
        let groups = group_words(&words, 0);
        assert_eq!(groups.len(), 2); // 8 + 1
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(to_srt(&[]), "");
        assert_eq!(to_vtt(&[]), "WEBVTT\n\n");
        assert!(group_words(&[], 5).is_empty());
    }

    #[test]
    fn test_to_srt() {
        let words = vec![
            make_word("Hello", 0.0, 0.5),
            make_word("world", 0.6, 1.0),
        ];
        let srt = to_srt(&words);
        assert!(srt.starts_with("1\n"));
        assert!(srt.contains("00:00:00,000 --> 00:00:01,000"));
        assert!(srt.contains("Hello world"));
    }

    #[test]
    fn test_to_vtt() {
        let words = vec![
            make_word("Hello", 0.0, 0.5),
            make_word("world", 0.6, 1.0),
        ];
        let vtt = to_vtt(&words);
        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:01.000"));
        assert!(vtt.contains("Hello world"));
        // VTT should NOT contain sequence numbers
        assert!(!vtt.contains("1\n00:00"));
    }

    #[test]
    fn test_srt_multiple_groups() {
        let words: Vec<WordTimestamp> = (0..10)
            .map(|i| make_word(&format!("w{i}"), i as f32, i as f32 + 0.5))
            .collect();
        let srt = to_srt(&words);
        assert!(srt.contains("1\n"));
        assert!(srt.contains("2\n"));
    }
}
