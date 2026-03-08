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
use crate::words::{WordTimestamp, compute_chunk_offsets, extract_words, extract_words_with_confidence};

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

pub fn transcribe(
    models: &Models, config: &Config, audio_path: &Path, language: &str,
    vad_override: Option<bool>, punctuate_override: Option<bool>, max_chunk_len: usize,
) -> Result<TranscribeResult, TranscribeError> {
    let start = Instant::now();
    let (wav_path, needs_cleanup) = ensure_wav(audio_path)?;
    let result = do_transcribe(models, config, &wav_path, language, vad_override, punctuate_override, max_chunk_len);
    if needs_cleanup {
        let _ = std::fs::remove_file(&wav_path);
    }
    let mut res = result?;
    res.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(res)
}

fn do_transcribe(
    models: &Models, config: &Config, wav_path: &Path, language: &str,
    vad_override: Option<bool>, punctuate_override: Option<bool>, max_chunk_len: usize,
) -> Result<TranscribeResult, TranscribeError> {
    let (samples, duration) = load_wav(wav_path)?;
    if config.max_audio_duration_s > 0.0 && duration > config.max_audio_duration_s {
        return Err(TranscribeError::TooLong(duration, config.max_audio_duration_s));
    }

    let use_vad = vad_override
        .unwrap_or(duration >= config.vad_min_duration_s && models.vad.is_some());

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

    let chunk_offsets = compute_chunk_offsets(&audio_chunks, 16000);
    let (texts, words) = match language {
        "ru" => transcribe_ru(models, &audio_chunks, &chunk_offsets)?,
        _ => transcribe_en(models, &audio_chunks, language, &chunk_offsets)?,
    };

    let joined = texts.join(" ");
    let text = sanitize_utf8(joined.trim());

    // Skip external punctuation if RU model has built-in punct (GigaAM v3 transducer)
    let has_builtin = language == "ru" && models.ru.as_ref()
        .and_then(|p| p.acquire())
        .map(|r| r.has_builtin_punct())
        .unwrap_or(false);
    let text = if has_builtin {
        text
    } else {
        maybe_punctuate(models, &text, language, punctuate_override)
    };
    let chunks = if max_chunk_len > 0 { split_text(&text, max_chunk_len) } else { Vec::new() };

    Ok(TranscribeResult { text, chunks, duration_ms: 0.0, speech_ms, words })
}

fn transcribe_en(
    models: &Models, chunks: &[Vec<f32>], language: &str, chunk_offsets: &[f64],
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
        let offset = chunk_offsets.get(i).copied().unwrap_or(0.0) as f32;
        if text.is_empty() {
            tracing::debug!("EN chunk {}: empty ({} samples)", i, chunk.len());
        } else if ratio > 2.4 {
            // Retry with trimmed audio (drop 5% from edges to shift alignment)
            let trim = chunk.len() / 20;
            if trim > 0 && chunk.len() > trim * 2 + 1600 {
                let retry = rec.transcribe(16000, &chunk[trim..chunk.len() - trim]);
                let rt = retry.text.trim().to_string();
                if !rt.is_empty() && compression_ratio(&rt) <= 2.4 {
                    tracing::info!("EN chunk {}: retry ok (ratio {:.2} -> ok)", i, ratio);
                    extract_words_with_confidence(&retry.tokens, &retry.timestamps, &retry.log_probs, offset, &mut words);
                    texts.push(rt);
                    continue;
                }
            }
            tracing::warn!("EN chunk {}: ratio {:.2}, skip: {:?}", i, ratio, &text[..text.len().min(80)]);
        } else {
            extract_words_with_confidence(&result.tokens, &result.timestamps, &result.log_probs, offset, &mut words);
            texts.push(text);
        }
    }
    Ok((texts, words))
}

fn transcribe_ru(
    models: &Models, chunks: &[Vec<f32>], chunk_offsets: &[f64],
) -> Result<(Vec<String>, Vec<WordTimestamp>), TranscribeError> {
    let pool = models.ru.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable("ru".to_string()))?;
    let mut rec = pool.acquire().ok_or(TranscribeError::NoRecognizer)?;
    let chunk_refs: Vec<&[f32]> = chunks.iter().map(|c| c.as_slice()).collect();
    let results = rec.transcribe_batch(16000, &chunk_refs);

    let mut texts = Vec::new();
    let mut words = Vec::new();
    for (i, r) in results.into_iter().enumerate() {
        let t = r.text.trim().to_string();
        if t.is_empty() || compression_ratio(&t) > 2.4 { continue; }
        let offset = chunk_offsets.get(i).copied().unwrap_or(0.0) as f32;

        if !r.timestamps.is_empty() {
            extract_words_with_confidence(&r.tokens, &r.timestamps, &r.log_probs, offset, &mut words);
        }

        texts.push(t);
    }
    Ok((texts, words))
}

pub(crate) fn maybe_punctuate(
    models: &Models, text: &str, language: &str, punctuate_override: Option<bool>,
) -> String {
    let should = match punctuate_override {
        Some(v) => v,
        None => (language == "en" || language == "ru") && models.punct.is_some(),
    };
    if should {
        if let Some(ref m) = models.punct {
            if let Ok(p) = m.lock() { return add_punctuation(&p, text); }
        }
    }
    text.to_string()
}

const MAX_CHUNK_SECONDS: usize = 20;
const MAX_CHUNK_SAMPLES: usize = MAX_CHUNK_SECONDS * 16000;

pub(crate) fn split_audio_chunks(samples: Vec<f32>, _sample_rate: u32) -> Vec<Vec<f32>> {
    if samples.len() <= MAX_CHUNK_SAMPLES {
        return vec![samples];
    }
    samples.chunks(MAX_CHUNK_SAMPLES).map(|c| c.to_vec()).collect()
}

pub(crate) fn compression_ratio(text: &str) -> f64 {
    if text.len() < 10 { return 0.0; }
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(text.as_bytes()).ok();
    let compressed = enc.finish().unwrap_or_default();
    text.len() as f64 / compressed.len() as f64
}
