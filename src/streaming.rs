use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::audio::{ensure_wav, load_wav};
use crate::chunking::sanitize_utf8;
use crate::config::Config;
use crate::models::Models;
use crate::recognizer::RuRecognizer;
use crate::routing::Engine;
use crate::transcribe::{
    TranscribeError, TranscribeResult, compression_ratio, maybe_punctuate, split_audio_chunks,
};
use crate::vad::{apply_vad, lock_vad};

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
    let wav = ensure_wav(audio_path, &config.upload_dir)?;

    let result = do_transcribe_streaming(models, config, wav.path(), language, vad_override, &tx);

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

    if config.max_audio_duration_s > 0.0 && duration > config.max_audio_duration_s {
        return Err(TranscribeError::TooLong(
            duration,
            config.max_audio_duration_s,
        ));
    }

    let use_vad =
        vad_override.unwrap_or(duration >= config.vad_min_duration_s && models.vad.is_some());

    let max_chunk_samples = config.max_chunk_s * 16000;
    let (audio_chunks, speech_ms) = if use_vad {
        if let Some(ref vad_mutex) = models.vad {
            let mut vad = lock_vad(vad_mutex);
            let vad_result = apply_vad(
                &mut vad,
                &samples,
                16000,
                config.vad_speech_pad_s,
                config.vad_max_chunk_s,
                "sse",
            );
            (vad_result.chunks, vad_result.speech_ms)
        } else {
            (split_audio_chunks(samples, max_chunk_samples), 0.0)
        }
    } else {
        (split_audio_chunks(samples, max_chunk_samples), 0.0)
    };

    let total = audio_chunks.len();

    let threshold = config.hallucination_threshold;
    let engine = models.route(language, &config.parakeet_langs);
    let run = |engine: Engine| match engine {
        Engine::Parakeet | Engine::Ru => transcribe_offline_streaming(
            models.offline_pool(engine),
            engine,
            language,
            &audio_chunks,
            total,
            tx,
            threshold,
        ),
        Engine::Moonshine => {
            transcribe_en_streaming(models, &audio_chunks, language, total, tx, threshold)
        }
    };
    // Same reload fallback as the batch path (see transcribe_routed).
    let (engine, texts) = match run(engine) {
        Err(TranscribeError::ReloadFailed(_)) if engine == Engine::Parakeet => {
            let fallback = models
                .reload_fallback(language, &config.parakeet_langs)
                .ok_or(TranscribeError::ReloadFailed(engine.label()))?;
            metrics::counter!(crate::metrics::names::ROUTE_FALLBACK, "to" => fallback.label())
                .increment(1);
            (fallback, run(fallback)?)
        }
        other => (engine, other?),
    };

    let joined = texts.join(" ");
    let text = sanitize_utf8(joined.trim());
    let text = maybe_punctuate(models, &text, language, engine, None);

    Ok(TranscribeResult {
        text,
        chunks: Vec::new(),
        duration_ms: 0.0,
        audio_duration_ms: duration * 1000.0,
        speech_ms,
        words: Vec::new(),
    })
}

fn transcribe_offline_streaming(
    pool: Option<&std::sync::Arc<crate::pool::EvictablePool<RuRecognizer>>>,
    engine: Engine,
    language: &str,
    chunks: &[Vec<f32>],
    total: usize,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    threshold: f64,
) -> Result<Vec<String>, TranscribeError> {
    let pool = pool.ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = pool.acquire().map_err(|e| {
        tracing::warn!("{} pool acquire failed (streaming): {e}", engine.label());
        match e {
            crate::pool::AcquireError::ReinitFailed(_) => {
                TranscribeError::ReloadFailed(engine.label())
            }
            crate::pool::AcquireError::AllBusy => TranscribeError::NoRecognizer,
        }
    })?;
    let mut texts = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let result = rec.transcribe(16000, chunk);
        let text = result.text.trim().to_string();
        if !text.is_empty() && compression_ratio(&text) <= threshold {
            let _ = tx.blocking_send(StreamEvent {
                chunk_index: i,
                total_chunks: total,
                text: text.clone(),
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
    threshold: f64,
) -> Result<Vec<String>, TranscribeError> {
    let pool = models
        .en
        .as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = pool.acquire().map_err(|e| {
        tracing::warn!("EN pool acquire failed (streaming): {e}");
        TranscribeError::NoRecognizer
    })?;
    let mut texts = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let result = rec.transcribe(16000, chunk);
        let text = result.text.trim().to_string();
        if !text.is_empty() && compression_ratio(&text) <= threshold {
            let _ = tx.blocking_send(StreamEvent {
                chunk_index: i,
                total_chunks: total,
                text: text.clone(),
            });
            texts.push(text);
        }
    }
    Ok(texts)
}
