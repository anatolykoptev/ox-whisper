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
            let vad_result = apply_vad(&mut vad, &samples, 16000);
            (vad_result.chunks, vad_result.speech_ms)
        } else {
            (split_audio_chunks(samples, 16000), 0.0)
        }
    } else {
        (split_audio_chunks(samples, 16000), 0.0)
    };

    // 6. Select recognizer and transcribe chunks
    let texts = match language {
        "ru" => transcribe_ru(models, &audio_chunks)?,
        _ => transcribe_en(models, &audio_chunks, language)?,
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

    Ok(TranscribeResult { text, chunks, duration_ms: 0.0, speech_ms })
}

fn transcribe_en(
    models: &Models,
    chunks: &[Vec<f32>],
    language: &str,
) -> Result<Vec<String>, TranscribeError> {
    let en_mutex = models.en.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = en_mutex.lock().map_err(|_| TranscribeError::NoRecognizer)?;
    let mut texts = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let result = rec.transcribe(16000, chunk);
        let text = result.text.trim().to_string();
        let ratio = compression_ratio(&text);
        if text.is_empty() {
            tracing::debug!("EN chunk {}: empty result ({} samples)", i, chunk.len());
        } else if ratio > 2.4 {
            tracing::warn!("EN chunk {}: compression ratio {:.2} > 2.4, skipping hallucination: {:?}",
                i, ratio, &text[..text.len().min(80)]);
        } else {
            texts.push(text);
        }
    }
    Ok(texts)
}

fn transcribe_ru(
    models: &Models,
    chunks: &[Vec<f32>],
) -> Result<Vec<String>, TranscribeError> {
    let ru_mutex = models.ru.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable("ru".to_string()))?;
    let mut rec = ru_mutex.lock().map_err(|_| TranscribeError::NoRecognizer)?;
    let mut texts = Vec::new();
    for chunk in chunks {
        let text = rec.transcribe(16000, chunk).trim().to_string();
        if !text.is_empty() && compression_ratio(&text) <= 2.4 {
            texts.push(text);
        }
    }
    Ok(texts)
}

fn maybe_punctuate(
    models: &Models,
    text: &str,
    language: &str,
    punctuate_override: Option<bool>,
) -> String {
    let should_punctuate = match punctuate_override {
        Some(v) => v,
        None => language != "ru" && models.punct.is_some(),
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

fn split_audio_chunks(samples: Vec<f32>, _sample_rate: u32) -> Vec<Vec<f32>> {
    if samples.len() <= MAX_CHUNK_SAMPLES {
        return vec![samples];
    }
    samples.chunks(MAX_CHUNK_SAMPLES).map(|c| c.to_vec()).collect()
}

fn compression_ratio(text: &str) -> f64 {
    if text.len() < 10 {
        return 0.0;
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(text.as_bytes()).ok();
    let compressed = encoder.finish().unwrap_or_default();
    text.len() as f64 / compressed.len() as f64
}
