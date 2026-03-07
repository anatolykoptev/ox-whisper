use std::io::Write;
use std::path::Path;
use std::time::Instant;

use flate2::Compression;
use flate2::write::ZlibEncoder;

use crate::audio::{AudioError, ensure_wav, load_wav};
use crate::chunking::{sanitize_utf8, split_text};
use crate::config::Config;
use crate::models::Models;
use crate::punctuate::add_punctuation;
use crate::vad::apply_vad;

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error("audio error: {0}")]
    Audio(#[from] AudioError),
    #[error("language '{0}' not supported or model not loaded")]
    LanguageNotAvailable(String),
    #[error("audio too long: {0:.1}s exceeds max {1:.1}s")]
    TooLong(f64, f64),
    #[error("no recognizer available")]
    NoRecognizer,
}

pub struct TranscribeResult {
    pub text: String,
    pub chunks: Vec<String>,
    pub duration_ms: f64,
    pub speech_ms: f64,
    pub words: Vec<WordTimestamp>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

/// Core transcription pipeline: WAV conversion -> VAD -> recognition -> punctuation -> chunking.
pub fn transcribe(
    models: &Models,
    config: &Config,
    audio_path: &Path,
    language: &str,
    vad_override: Option<bool>,
    punctuate_override: Option<bool>,
    max_chunk_len: usize,
) -> Result<TranscribeResult, TranscribeError> {
    let start = Instant::now();

    // 1. Convert to WAV if needed
    let (wav_path, needs_cleanup) = ensure_wav(audio_path)?;
    let result = do_transcribe(models, config, &wav_path, language, vad_override, punctuate_override, max_chunk_len);

    // Cleanup temp WAV even on error
    if needs_cleanup {
        let _ = std::fs::remove_file(&wav_path);
    }

    let mut res = result?;
    res.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(res)
}

fn do_transcribe(
    models: &Models,
    config: &Config,
    wav_path: &Path,
    language: &str,
    vad_override: Option<bool>,
    punctuate_override: Option<bool>,
    max_chunk_len: usize,
) -> Result<TranscribeResult, TranscribeError> {
    // 2. Load WAV
    let (samples, duration) = load_wav(wav_path)?;

    // 3. Check duration limit
    if duration > config.max_audio_duration_s {
        return Err(TranscribeError::TooLong(duration, config.max_audio_duration_s));
    }

    // 4. Decide VAD
    let use_vad = vad_override
        .unwrap_or(duration >= config.vad_min_duration_s && models.vad.is_some());

    // 5. Build audio chunks
    let (audio_chunks, speech_ms) = if use_vad {
        if let Some(ref vad_mutex) = models.vad {
            let mut vad = vad_mutex.lock().map_err(|_| TranscribeError::NoRecognizer)?;
            let vad_result = apply_vad(&mut vad, &samples, 16000, config.vad_speech_pad_s);
            let total_ms = duration * 1000.0;
            let pct = if total_ms > 0.0 { 100.0 * vad_result.speech_ms / total_ms } else { 0.0 };
            tracing::info!("VAD: {:.0}ms speech / {:.0}ms total ({:.0}%), {} chunk(s)",
                vad_result.speech_ms, total_ms, pct, vad_result.chunks.len());
            (vad_result.chunks, vad_result.speech_ms)
        } else {
            (split_audio_chunks(samples, 16000), 0.0)
        }
    } else {
        (split_audio_chunks(samples, 16000), 0.0)
    };

    // 6. Select recognizer and transcribe chunks
    let chunk_offsets: Vec<f64> = compute_chunk_offsets(&audio_chunks, 16000);
    let (texts, words) = match language {
        "ru" => (transcribe_ru(models, &audio_chunks)?, Vec::new()),
        _ => transcribe_en(models, &audio_chunks, language, &chunk_offsets)?,
    };

    // 9. Join texts
    let joined = texts.join(" ");
    let text = sanitize_utf8(joined.trim());

    // 10. Punctuate
    let text = maybe_punctuate(models, &text, language, punctuate_override);

    // 11. Split into chunks
    let chunks = if max_chunk_len > 0 {
        split_text(&text, max_chunk_len)
    } else {
        Vec::new()
    };

    Ok(TranscribeResult { text, chunks, duration_ms: 0.0, speech_ms, words })
}

fn transcribe_en(
    models: &Models,
    chunks: &[Vec<f32>],
    language: &str,
    chunk_offsets: &[f64],
) -> Result<(Vec<String>, Vec<WordTimestamp>), TranscribeError> {
    let pool = models.en.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = pool.acquire().ok_or(TranscribeError::NoRecognizer)?;
    let mut texts = Vec::new();
    let mut words = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let result = rec.transcribe(16000, chunk);
        let text = result.text.trim().to_string();
        let ratio = compression_ratio(&text);
        if text.is_empty() {
            tracing::debug!("EN chunk {}: empty result ({} samples)", i, chunk.len());
        } else if ratio > 2.4 {
            // Retry with trimmed audio (drop first/last 5% to shift alignment)
            let trim = chunk.len() / 20;
            if trim > 0 && chunk.len() > trim * 2 + 1600 {
                let trimmed = &chunk[trim..chunk.len() - trim];
                let retry = rec.transcribe(16000, trimmed);
                let retry_text = retry.text.trim().to_string();
                let retry_ratio = compression_ratio(&retry_text);
                if !retry_text.is_empty() && retry_ratio <= 2.4 {
                    tracing::info!("EN chunk {}: retry succeeded (ratio {:.2} -> {:.2})", i, ratio, retry_ratio);
                    let offset = chunk_offsets.get(i).copied().unwrap_or(0.0) as f32;
                    extract_words(&retry.tokens, &retry.timestamps, offset, &mut words);
                    texts.push(retry_text);
                    continue;
                }
            }
            tracing::warn!("EN chunk {}: compression ratio {:.2} > 2.4, skipping hallucination: {:?}",
                i, ratio, &text[..text.len().min(80)]);
        } else {
            let offset = chunk_offsets.get(i).copied().unwrap_or(0.0) as f32;
            extract_words(&result.tokens, &result.timestamps, offset, &mut words);
            texts.push(text);
        }
    }
    Ok((texts, words))
}

fn transcribe_ru(
    models: &Models,
    chunks: &[Vec<f32>],
) -> Result<Vec<String>, TranscribeError> {
    let pool = models.ru.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable("ru".to_string()))?;
    let mut rec = pool.acquire().ok_or(TranscribeError::NoRecognizer)?;
    let mut texts = Vec::new();
    for chunk in chunks {
        let text = rec.transcribe(16000, chunk).trim().to_string();
        if !text.is_empty() && compression_ratio(&text) <= 2.4 {
            texts.push(text);
        }
    }
    Ok(texts)
}

fn compute_chunk_offsets(chunks: &[Vec<f32>], sample_rate: u32) -> Vec<f64> {
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut offset = 0.0;
    for chunk in chunks {
        offsets.push(offset);
        offset += chunk.len() as f64 / sample_rate as f64;
    }
    offsets
}

fn extract_words(tokens: &[String], timestamps: &[f32], offset: f32, out: &mut Vec<WordTimestamp>) {
    if tokens.is_empty() || timestamps.is_empty() {
        return;
    }
    // Group subword tokens into words (tokens starting with space = new word)
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

        let starts_new_word = token.starts_with('▁') || token.starts_with(' ') || current_word.is_empty();
        if starts_new_word && !current_word.is_empty() {
            let w = current_word.trim().to_string();
            if !w.is_empty() {
                out.push(WordTimestamp { word: w, start: word_start + offset, end: word_end + offset });
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
            out.push(WordTimestamp { word: w, start: word_start + offset, end: word_end + offset });
        }
    }
}

pub(crate) fn maybe_punctuate(
    models: &Models,
    text: &str,
    language: &str,
    punctuate_override: Option<bool>,
) -> String {
    let should_punctuate = match punctuate_override {
        Some(v) => v,
        None => language == "en" && models.punct.is_some(),
    };
    if should_punctuate {
        if let Some(ref punct_mutex) = models.punct {
            if let Ok(punct) = punct_mutex.lock() {
                return add_punctuation(&punct, text);
            }
        }
    }
    text.to_string()
}

const MAX_CHUNK_SAMPLES: usize = 5 * 16000; // 5s at 16kHz

pub(crate) fn split_audio_chunks(samples: Vec<f32>, _sample_rate: u32) -> Vec<Vec<f32>> {
    if samples.len() <= MAX_CHUNK_SAMPLES {
        return vec![samples];
    }
    samples.chunks(MAX_CHUNK_SAMPLES).map(|c| c.to_vec()).collect()
}

pub(crate) fn compression_ratio(text: &str) -> f64 {
    if text.len() < 10 {
        return 0.0;
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(text.as_bytes()).ok();
    let compressed = encoder.finish().unwrap_or_default();
    text.len() as f64 / compressed.len() as f64
}
