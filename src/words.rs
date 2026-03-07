/// Word-level timestamp extraction from recognizer token output.

#[derive(Debug, Clone, serde::Serialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f32,
    pub end: f32,
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
pub fn extract_words(
    tokens: &[String],
    timestamps: &[f32],
    offset: f32,
    out: &mut Vec<WordTimestamp>,
) {
    if tokens.is_empty() || timestamps.is_empty() {
        return;
    }

    let mut current_word = String::new();
    let mut word_start: f32 = 0.0;
    let mut word_end: f32 = 0.0;

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
                });
            }
            current_word.clear();
            word_start = t;
        }
        if current_word.is_empty() {
            word_start = t;
        }
        current_word.push_str(clean.trim());
        word_end = t;
    }

    if !current_word.is_empty() {
        let w = current_word.trim().to_string();
        if !w.is_empty() {
            out.push(WordTimestamp {
                word: w,
                start: word_start + offset,
                end: word_end + offset,
            });
        }
    }
}
