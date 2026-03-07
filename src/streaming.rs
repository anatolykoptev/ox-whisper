use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::audio::{ensure_wav, load_wav};
use crate::chunking::sanitize_utf8;
use crate::config::Config;
use crate::models::Models;
use crate::transcribe::{
    TranscribeError, TranscribeResult, compression_ratio, maybe_punctuate, split_audio_chunks,
};
use crate::vad::apply_vad;

#[derive(Serialize, Clone)]
pub struct StreamEvent {
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub text: String,
}

/// Streaming transcription: sends per-chunk results via `tx`, then returns final result.
/// Runs synchronously (call from `spawn_blocking`).
pub fn transcribe_streaming(
    models: &Models,
    config: &Config,
    audio_path: &Path,
    language: &str,
    vad_override: Option<bool>,
    tx: tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<TranscribeResult, TranscribeError> {
    let start = Instant::now();
    let (wav_path, needs_cleanup) = ensure_wav(audio_path)?;

    let result = do_transcribe_streaming(models, config, &wav_path, language, vad_override, &tx);

    if needs_cleanup {
        let _ = std::fs::remove_file(&wav_path);
    }

    let mut res = result?;
    res.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(res)
}

fn do_transcribe_streaming(
    models: &Models,
    config: &Config,
    wav_path: &Path,
    language: &str,
    vad_override: Option<bool>,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<TranscribeResult, TranscribeError> {
    let (samples, duration) = load_wav(wav_path)?;

    if duration > config.max_audio_duration_s {
        return Err(TranscribeError::TooLong(duration, config.max_audio_duration_s));
    }

    let use_vad = vad_override
        .unwrap_or(duration >= config.vad_min_duration_s && models.vad.is_some());

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

    let total = audio_chunks.len();

    let texts = match language {
        "ru" => transcribe_ru_streaming(models, &audio_chunks, total, tx)?,
        _ => transcribe_en_streaming(models, &audio_chunks, language, total, tx)?,
    };

    let joined = texts.join(" ");
    let text = sanitize_utf8(joined.trim());
    let text = maybe_punctuate(models, &text, language, None);

    Ok(TranscribeResult { text, chunks: Vec::new(), duration_ms: 0.0, speech_ms })
}

fn transcribe_ru_streaming(
    models: &Models,
    chunks: &[Vec<f32>],
    total: usize,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<Vec<String>, TranscribeError> {
    let pool = models.ru.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable("ru".to_string()))?;
    let mut rec = pool.acquire().ok_or(TranscribeError::NoRecognizer)?;
    let mut texts = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let text = rec.transcribe(16000, chunk).trim().to_string();
        if !text.is_empty() && compression_ratio(&text) <= 2.4 {
            let _ = tx.blocking_send(StreamEvent {
                chunk_index: i, total_chunks: total, text: text.clone(),
            });
            texts.push(text);
        }
    }
    Ok(texts)
}

fn transcribe_en_streaming(
    models: &Models,
    chunks: &[Vec<f32>],
    language: &str,
    total: usize,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<Vec<String>, TranscribeError> {
    let pool = models.en.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = pool.acquire().ok_or(TranscribeError::NoRecognizer)?;
    let mut texts = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let result = rec.transcribe(16000, chunk);
        let text = result.text.trim().to_string();
        if !text.is_empty() && compression_ratio(&text) <= 2.4 {
            let _ = tx.blocking_send(StreamEvent {
                chunk_index: i, total_chunks: total, text: text.clone(),
            });
            texts.push(text);
        }
    }
    Ok(texts)
}
