/// Auto-split transcription into paragraphs based on pauses between words.

use crate::words::WordTimestamp;

const DEFAULT_PAUSE_THRESHOLD_S: f32 = 1.5;

/// Split text into paragraphs based on word timing gaps.
/// If gap between consecutive words >= threshold, insert double newline.
pub fn split_paragraphs(text: &str, words: &[WordTimestamp], threshold: f32) -> String {
    if words.len() < 2 {
        return text.to_string();
    }

    let mut result = String::new();
    let mut word_idx = 0;

    for token in text.split_whitespace() {
        if word_idx > 0 && word_idx < words.len() {
            let gap = words[word_idx].start - words[word_idx - 1].end;
            if gap >= threshold {
                result.push_str("\n\n");
            } else {
                result.push(' ');
            }
        } else if word_idx > 0 {
            result.push(' ');
        }
        result.push_str(token);
        word_idx += 1;
    }

    result
}

/// Get the default pause threshold.
pub fn default_threshold() -> f32 {
    DEFAULT_PAUSE_THRESHOLD_S
}

#[cfg(test)]
#[path = "paragraphs_tests.rs"]
mod tests;
