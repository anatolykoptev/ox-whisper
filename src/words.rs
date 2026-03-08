/// Word-level timestamp extraction from recognizer token output.

#[derive(Debug, Clone, serde::Serialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f32,
    pub end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<i32>,
}

/// Compute the time offset (in seconds) of each audio chunk.
pub fn compute_chunk_offsets(chunks: &[Vec<f32>], sample_rate: u32) -> Vec<f64> {
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut offset = 0.0;
    for chunk in chunks {
        offsets.push(offset);
        offset += chunk.len() as f64 / sample_rate as f64;
    }
    offsets
}

/// Group subword tokens into words using timestamp alignment.
///
/// Tokens starting with '▁' or space indicate a new word boundary.
/// Special tokens (sos, eos, eot) are skipped. Each word gets
/// the start time of its first token and end time of its last token,
/// shifted by the chunk offset.
pub fn extract_words_with_confidence(
    tokens: &[String],
    timestamps: &[f32],
    log_probs: &[f32],
    offset: f32,
    out: &mut Vec<WordTimestamp>,
) {
    if tokens.is_empty() || timestamps.is_empty() {
        return;
    }

    let mut current_word = String::new();
    let mut word_start: f32 = 0.0;
    let mut word_end: f32 = 0.0;
    let mut word_log_probs: Vec<f32> = Vec::new();

    for (i, token) in tokens.iter().enumerate() {
        let t = timestamps.get(i).copied().unwrap_or(0.0);
        let clean = token.replace('▁', " ");
        let clean = clean.trim_matches(|c: char| c == '<' || c == '>');

        if clean.is_empty() || clean == "sos" || clean == "eos" || clean == "eot" {
            continue;
        }

        let starts_new_word = token.starts_with('▁')
            || token.starts_with(' ')
            || current_word.is_empty();

        if starts_new_word && !current_word.is_empty() {
            let w = current_word.trim().to_string();
            if !w.is_empty() {
                out.push(WordTimestamp {
                    word: w,
                    start: word_start + offset,
                    end: word_end + offset,
                    confidence: avg_confidence(&word_log_probs),
                    speaker: None,
                });
            }
            current_word.clear();
            word_log_probs.clear();
            word_start = t;
        }
        if current_word.is_empty() {
            word_start = t;
        }
        current_word.push_str(clean.trim());
        word_end = t;
        if let Some(&lp) = log_probs.get(i) {
            word_log_probs.push(lp);
        }
    }

    if !current_word.is_empty() {
        let w = current_word.trim().to_string();
        if !w.is_empty() {
            out.push(WordTimestamp {
                word: w,
                start: word_start + offset,
                end: word_end + offset,
                confidence: avg_confidence(&word_log_probs),
                speaker: None,
            });
        }
    }
}

/// Estimate word timestamps proportionally from text when the model
/// doesn't provide per-token timestamps (e.g. Moonshine v2).
pub fn estimate_words_from_text(
    text: &str, audio_duration_s: f32, offset: f32,
) -> Vec<WordTimestamp> {
    let text = text.trim();
    if text.is_empty() || audio_duration_s <= 0.0 {
        return Vec::new();
    }
    let total_chars = text.chars().count();
    if total_chars == 0 {
        return Vec::new();
    }

    let mut words = Vec::new();
    let mut char_pos = 0usize;

    for word in text.split_whitespace() {
        let word_chars = word.chars().count();
        let start = audio_duration_s * (char_pos as f32 / total_chars as f32) + offset;
        let end = audio_duration_s * ((char_pos + word_chars) as f32 / total_chars as f32) + offset;
        words.push(WordTimestamp {
            word: word.to_string(),
            start,
            end,
            confidence: None,
            speaker: None,
        });
        char_pos += word_chars + 1; // +1 for the space
    }
    words
}

fn avg_confidence(log_probs: &[f32]) -> Option<f32> {
    if log_probs.is_empty() {
        return None;
    }
    let avg = log_probs.iter().sum::<f32>() / log_probs.len() as f32;
    Some(avg.exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_basic() {
        let words = estimate_words_from_text("hello world", 10.0, 0.0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[1].word, "world");
        assert!(words[0].start < words[0].end);
        assert!(words[0].end <= words[1].start || (words[0].end - words[1].start).abs() < 0.01);
        assert!(words[1].end <= 10.0 + 0.01);
        assert!(words[0].confidence.is_none());
        assert!(words[0].speaker.is_none());
    }

    #[test]
    fn estimate_with_offset() {
        let words = estimate_words_from_text("hi there", 4.0, 5.0);
        assert_eq!(words.len(), 2);
        assert!(words[0].start >= 5.0);
        assert!(words[1].end <= 9.0 + 0.01);
    }

    #[test]
    fn estimate_empty_text() {
        assert!(estimate_words_from_text("", 10.0, 0.0).is_empty());
        assert!(estimate_words_from_text("   ", 10.0, 0.0).is_empty());
    }

    #[test]
    fn estimate_zero_duration() {
        assert!(estimate_words_from_text("hello", 0.0, 0.0).is_empty());
        assert!(estimate_words_from_text("hello", -1.0, 0.0).is_empty());
    }

    #[test]
    fn estimate_single_word() {
        let words = estimate_words_from_text("hello", 5.0, 0.0);
        assert_eq!(words.len(), 1);
        assert!((words[0].start - 0.0).abs() < 0.001);
        assert!((words[0].end - 5.0).abs() < 0.001);
    }

    #[test]
    fn estimate_cyrillic() {
        let words = estimate_words_from_text("привет мир", 6.0, 0.0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "привет");
        assert_eq!(words[1].word, "мир");
        assert!(words[1].end <= 6.0 + 0.01);
    }
}
