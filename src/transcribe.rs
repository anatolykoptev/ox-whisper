use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::metrics::names;

use flate2::Compression;
use flate2::write::ZlibEncoder;

use crate::audio::{AudioError, ensure_wav, load_wav};
use crate::chunking::{sanitize_utf8, split_text};
use crate::config::Config;
use crate::models::Models;
use crate::punctuate::add_punctuation;
use crate::recognizer::RuRecognizer;
use crate::routing::Engine;
use crate::vad::{apply_vad, lock_vad};
use crate::words::{
    WordTimestamp, compute_chunk_offsets, estimate_words_from_text, extract_words_with_confidence,
};

/// Try model timestamps first, fall back to proportional estimation.
fn extract_or_estimate(
    tokens: &[String],
    timestamps: &[f32],
    log_probs: &[f32],
    text: &str,
    chunk_samples: usize,
    offset: f32,
    words: &mut Vec<WordTimestamp>,
) {
    let before = words.len();
    extract_words_with_confidence(tokens, timestamps, log_probs, offset, words);
    if words.len() == before && !text.is_empty() {
        let dur = chunk_samples as f32 / 16000.0;
        words.extend(estimate_words_from_text(text, dur, offset));
    }
}

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
    #[error("{0} model failed to reload after idle eviction and no fallback model is loaded")]
    ReloadFailed(&'static str),
}

pub struct TranscribeResult {
    pub text: String,
    pub chunks: Vec<String>,
    pub duration_ms: f64,
    pub audio_duration_ms: f64,
    pub speech_ms: f64,
    pub words: Vec<WordTimestamp>,
}

/// Test instrumentation: records when a decode job starts and whether its input
/// still exists after a stall, to prove a job keeps its file after the handler
/// that spawned it was dropped.
#[cfg(test)]
pub(crate) mod probe {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    pub static EVENTS: Mutex<Vec<(PathBuf, &'static str, bool)>> = Mutex::new(Vec::new());

    pub fn enter(path: &Path, delay: std::time::Duration) {
        let push = |stage| {
            EVENTS
                .lock()
                .unwrap()
                .push((path.to_path_buf(), stage, path.exists()))
        };
        push("started");
        std::thread::sleep(delay);
        push("after_stall");
    }
}

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
    #[cfg(test)]
    probe::enter(audio_path, config.decode_delay);
    let wav = ensure_wav(audio_path, &config.upload_dir)?;
    let result = do_transcribe(
        models,
        config,
        wav.path(),
        language,
        vad_override,
        punctuate_override,
        max_chunk_len,
    );
    let elapsed = start.elapsed().as_secs_f64();
    metrics::histogram!(names::TRANSCRIBE_DURATION, "lang" => language.to_string()).record(elapsed);
    let mut res = result?;
    res.duration_ms = elapsed * 1000.0;
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
    let (samples, duration) = load_wav(wav_path)?;
    if config.max_audio_duration_s > 0.0 && duration > config.max_audio_duration_s {
        return Err(TranscribeError::TooLong(
            duration,
            config.max_audio_duration_s,
        ));
    }
    metrics::histogram!(names::AUDIO_DURATION).record(duration);

    let engine = models.route(language, &config.parakeet_langs);
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
                "batch",
            );
            let total_ms = duration * 1000.0;
            let pct = if total_ms > 0.0 {
                100.0 * vad_result.speech_ms / total_ms
            } else {
                0.0
            };
            tracing::info!(
                "VAD: {:.0}ms speech / {:.0}ms total ({:.0}%), {} segment(s), {} chunk(s)",
                vad_result.speech_ms,
                total_ms,
                pct,
                vad_result.segments,
                vad_result.chunks.len()
            );
            let ratio = vad_result.speech_ms / total_ms.max(1.0);
            let chunks_count = vad_result.chunks.len();
            metrics::gauge!(names::VAD_SPEECH_RATIO, "lang" => language.to_string()).set(ratio);
            metrics::counter!(names::CHUNKS_TOTAL, "lang" => language.to_string())
                .increment(chunks_count as u64);
            (vad_result.chunks, vad_result.speech_ms)
        } else {
            (split_audio_chunks(samples, max_chunk_samples), 0.0)
        }
    } else {
        (split_audio_chunks(samples, max_chunk_samples), 0.0)
    };

    let chunk_offsets = compute_chunk_offsets(&audio_chunks, 16000);
    let threshold = config.hallucination_threshold;
    let (engine, texts, words) = transcribe_routed(
        models,
        config,
        engine,
        language,
        &audio_chunks,
        &chunk_offsets,
        threshold,
    )?;

    let joined = texts.join(" ");
    let text = sanitize_utf8(joined.trim());

    // Skip external punctuation for Moonshine (punctuates natively) and
    // Parakeet; RU output goes through maybe_punctuate, which skips a RU model
    // with built-in punctuation.
    let skip_punct = engine != Engine::Ru;
    let text = if skip_punct {
        text
    } else {
        maybe_punctuate(models, &text, language, engine, punctuate_override)
    };
    let chunks = if max_chunk_len > 0 {
        split_text(&text, max_chunk_len)
    } else {
        Vec::new()
    };

    Ok(TranscribeResult {
        text,
        chunks,
        duration_ms: 0.0,
        audio_duration_ms: duration * 1000.0,
        speech_ms,
        words,
    })
}

/// Transcribes `chunks` on `engine`. If Parakeet's slot was idle-evicted and
/// fails to reload (e.g. out of memory), the request falls back to the model
/// that would serve the language without Parakeet — Moonshine for `en`, the
/// RU model for `ru` when one is loaded — instead of failing. Returns the
/// engine that produced the text.
pub(crate) fn transcribe_routed(
    models: &Models,
    config: &Config,
    engine: Engine,
    language: &str,
    chunks: &[Vec<f32>],
    chunk_offsets: &[f64],
    threshold: f64,
) -> Result<(Engine, Vec<String>, Vec<WordTimestamp>), TranscribeError> {
    let run = |engine: Engine| match engine {
        Engine::Parakeet | Engine::Ru => transcribe_offline(
            models.offline_pool(engine),
            engine,
            language,
            chunks,
            chunk_offsets,
            threshold,
        ),
        Engine::Moonshine => transcribe_en(models, chunks, language, chunk_offsets, threshold),
    };
    match run(engine) {
        Err(TranscribeError::ReloadFailed(_)) if engine == Engine::Parakeet => {
            let Some(fallback) = models.reload_fallback(language, &config.parakeet_langs) else {
                return Err(TranscribeError::ReloadFailed(engine.label()));
            };
            tracing::warn!(
                "Parakeet failed to reload; '{language}' falls back to {}",
                fallback.label()
            );
            metrics::counter!(names::ROUTE_FALLBACK, "to" => fallback.label()).increment(1);
            let (texts, words) = run(fallback)?;
            Ok((fallback, texts, words))
        }
        other => other.map(|(t, w)| (engine, t, w)),
    }
}

fn transcribe_en(
    models: &Models,
    chunks: &[Vec<f32>],
    language: &str,
    chunk_offsets: &[f64],
    threshold: f64,
) -> Result<(Vec<String>, Vec<WordTimestamp>), TranscribeError> {
    let pool = models
        .en
        .as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = pool.acquire().map_err(|e| {
        tracing::warn!("EN pool acquire failed: {e}");
        TranscribeError::NoRecognizer
    })?;
    metrics::gauge!(names::POOL_BUSY, "lang" => "en").increment(1.0);
    struct EnBusyGuard;
    impl Drop for EnBusyGuard {
        fn drop(&mut self) {
            metrics::gauge!(names::POOL_BUSY, "lang" => "en").decrement(1.0);
        }
    }
    let _busy = EnBusyGuard;
    let mut texts = Vec::new();
    let mut words = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let result = rec.transcribe(16000, chunk);
        let text = result.text.trim().to_string();
        let ratio = compression_ratio(&text);
        let offset = chunk_offsets.get(i).copied().unwrap_or(0.0) as f32;
        if text.is_empty() {
            // Moonshine sometimes returns empty on longer chunks — retry by splitting in half
            if chunk.len() > 32000 {
                let mid = chunk.len() / 2;
                for (j, half) in [&chunk[..mid], &chunk[mid..]].iter().enumerate() {
                    let retry = rec.transcribe(16000, half);
                    let rt = retry.text.trim().to_string();
                    if !rt.is_empty() && compression_ratio(&rt) <= threshold {
                        let half_offset = offset + if j == 1 { mid as f32 / 16000.0 } else { 0.0 };
                        extract_or_estimate(
                            &retry.tokens,
                            &retry.timestamps,
                            &retry.log_probs,
                            &rt,
                            half.len(),
                            half_offset,
                            &mut words,
                        );
                        texts.push(rt);
                    }
                }
                tracing::info!("EN chunk {}: empty, retried as 2 halves", i);
            } else {
                tracing::debug!("EN chunk {}: empty ({} samples)", i, chunk.len());
            }
        } else if ratio > threshold {
            // Retry with trimmed audio (drop 5% from edges to shift alignment)
            let trim = chunk.len() / 20;
            if trim > 0 && chunk.len() > trim * 2 + 1600 {
                let retry = rec.transcribe(16000, &chunk[trim..chunk.len() - trim]);
                let rt = retry.text.trim().to_string();
                if !rt.is_empty() && compression_ratio(&rt) <= threshold {
                    tracing::info!("EN chunk {}: retry ok (ratio {:.2} -> ok)", i, ratio);
                    extract_or_estimate(
                        &retry.tokens,
                        &retry.timestamps,
                        &retry.log_probs,
                        &rt,
                        chunk.len(),
                        offset,
                        &mut words,
                    );
                    texts.push(rt);
                    continue;
                }
            }
            tracing::warn!(
                "EN chunk {}: ratio {:.2}, skip: {:?}",
                i,
                ratio,
                text.chars().take(80).collect::<String>()
            );
            metrics::counter!(names::HALLUCINATION_REJECTED, "lang" => "en").increment(1);
        } else {
            extract_or_estimate(
                &result.tokens,
                &result.timestamps,
                &result.log_probs,
                &text,
                chunk.len(),
                offset,
                &mut words,
            );
            texts.push(text);
        }
    }
    Ok((texts, words))
}

/// Batch-decodes `chunks` on an offline pool (Parakeet or RU).
pub(crate) fn transcribe_offline(
    pool: Option<&std::sync::Arc<crate::pool::EvictablePool<RuRecognizer>>>,
    engine: Engine,
    language: &str,
    chunks: &[Vec<f32>],
    chunk_offsets: &[f64],
    threshold: f64,
) -> Result<(Vec<String>, Vec<WordTimestamp>), TranscribeError> {
    let label = engine.label();
    let pool = pool.ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = pool.acquire().map_err(|e| {
        tracing::warn!("{label} pool acquire failed: {e}");
        match e {
            crate::pool::AcquireError::ReinitFailed(_) => TranscribeError::ReloadFailed(label),
            crate::pool::AcquireError::AllBusy => TranscribeError::NoRecognizer,
        }
    })?;
    metrics::gauge!(names::POOL_BUSY, "lang" => label).increment(1.0);
    struct BusyGuard(&'static str);
    impl Drop for BusyGuard {
        fn drop(&mut self) {
            metrics::gauge!(names::POOL_BUSY, "lang" => self.0).decrement(1.0);
        }
    }
    let _busy = BusyGuard(label);
    let results = decode_in_batches(chunks, DECODE_BATCH_CHUNKS, |batch| {
        rec.transcribe_batch(16000, batch)
    });

    let mut texts = Vec::new();
    let mut words = Vec::new();
    for (i, r) in results.into_iter().enumerate() {
        let t = r.text.trim().to_string();
        if t.is_empty() {
            continue;
        }
        if compression_ratio(&t) > threshold {
            metrics::counter!(names::HALLUCINATION_REJECTED, "lang" => label).increment(1);
            continue;
        }
        let offset = chunk_offsets.get(i).copied().unwrap_or(0.0) as f32;

        extract_or_estimate(
            &r.tokens,
            &r.timestamps,
            &r.log_probs,
            &t,
            chunks[i].len(),
            offset,
            &mut words,
        );

        texts.push(t);
    }
    Ok((texts, words))
}

pub(crate) fn maybe_punctuate(
    models: &Models,
    text: &str,
    language: &str,
    engine: Engine,
    punctuate_override: Option<bool>,
) -> String {
    if wants_external_punct(
        language,
        engine,
        punctuate_override,
        models.punct.is_some(),
        models.ru_builtin_punct,
    ) {
        if let Some(ref m) = models.punct {
            if let Ok(p) = m.lock() {
                return add_punctuation(&p, text);
            }
        }
    }
    text.to_string()
}

/// Whether text goes through the external (English CNN-BiLSTM) punctuation
/// model. Never for output that is already punctuated — Parakeet, or a RU
/// model with built-in punctuation — not even when the client asked for
/// punctuation, since it would only rewrite it.
fn wants_external_punct(
    language: &str,
    engine: Engine,
    punctuate_override: Option<bool>,
    punct_loaded: bool,
    ru_builtin_punct: bool,
) -> bool {
    if engine == Engine::Parakeet || (engine == Engine::Ru && ru_builtin_punct) {
        return false;
    }
    match punctuate_override {
        Some(v) => v,
        None => (language == "en" || language == "ru") && punct_loaded,
    }
}

/// Chunks decoded together in one `transcribe_batch` call. Batch decoding
/// pads every chunk to the longest and keeps all activations alive at once,
/// so one call over a whole file grows memory with the file's length; with
/// `VAD_MAX_CHUNK_S=20` this bounds a call to about 80 s of audio.
pub(crate) const DECODE_BATCH_CHUNKS: usize = 4;

/// Runs `decode` over `chunks` in groups of at most `batch`, preserving order.
pub(crate) fn decode_in_batches<R>(
    chunks: &[Vec<f32>],
    batch: usize,
    mut decode: impl FnMut(&[&[f32]]) -> Vec<R>,
) -> Vec<R> {
    let mut out = Vec::with_capacity(chunks.len());
    for group in chunks.chunks(batch.max(1)) {
        let refs: Vec<&[f32]> = group.iter().map(|c| c.as_slice()).collect();
        out.extend(decode(&refs));
    }
    out
}

pub(crate) fn split_audio_chunks(samples: Vec<f32>, max_chunk_samples: usize) -> Vec<Vec<f32>> {
    if max_chunk_samples == 0 || samples.len() <= max_chunk_samples {
        return vec![samples];
    }
    samples
        .chunks(max_chunk_samples)
        .map(|c| c.to_vec())
        .collect()
}

pub(crate) fn compression_ratio(text: &str) -> f64 {
    if text.len() < 10 {
        return 0.0;
    }
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(text.as_bytes()).ok();
    let compressed = enc.finish().unwrap_or_default();
    text.len() as f64 / compressed.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long file is decoded in bounded groups, in order — never in one call.
    #[test]
    fn long_files_are_decoded_in_bounded_batches() {
        let chunks: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 4]).collect();
        let mut sizes = Vec::new();
        let out = decode_in_batches(&chunks, DECODE_BATCH_CHUNKS, |batch| {
            sizes.push(batch.len());
            batch.iter().map(|c| c[0] as usize).collect()
        });
        assert_eq!(out, (0..10).collect::<Vec<_>>());
        assert!(sizes.iter().all(|&n| n <= DECODE_BATCH_CHUNKS), "{sizes:?}");
        assert_eq!(sizes.len(), 10usize.div_ceil(DECODE_BATCH_CHUNKS));
    }

    #[test]
    fn already_punctuated_output_skips_the_punctuation_model() {
        use Engine::*;
        // Parakeet writes case and punctuation: never, even when asked.
        for lang in ["ru", "en", "uk"] {
            assert!(!wants_external_punct(lang, Parakeet, None, true, false));
            assert!(!wants_external_punct(
                lang,
                Parakeet,
                Some(true),
                true,
                false
            ));
        }
        // RU model with built-in punctuation: never, even when asked.
        assert!(!wants_external_punct("ru", Ru, None, true, true));
        assert!(!wants_external_punct("ru", Ru, Some(true), true, true));
        // RU model without it: yes when the model is loaded or when asked.
        assert!(wants_external_punct("ru", Ru, None, true, false));
        assert!(wants_external_punct("ru", Ru, Some(true), false, false));
        // Moonshine EN is unaffected by the RU flag.
        assert!(wants_external_punct("en", Moonshine, None, true, true));
        assert!(!wants_external_punct(
            "en",
            Moonshine,
            Some(false),
            true,
            true
        ));
    }
}
