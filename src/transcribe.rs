use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::metrics::names;

use flate2::Compression;
use flate2::write::ZlibEncoder;

use crate::audio::{AudioError, ensure_wav, load_wav};
use crate::chunking::sanitize_utf8;
use crate::config::Config;
use crate::models::{Models, PARAKEET_POOL_LABEL};
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
    #[error("audio too long: {0:.1}s exceeds max {1:.1}s")]
    TooLong(f64, f64),
    #[error("no recognizer available")]
    NoRecognizer,
    #[error("the Parakeet model failed to reload after idle eviction")]
    ReloadFailed,
}

pub struct TranscribeResult {
    pub text: String,
    pub duration_ms: f64,
    pub audio_duration_ms: f64,
    pub words: Vec<WordTimestamp>,
}

/// `lang` only labels metrics (the validated request language, or `auto`):
/// the one model transcribes whatever language it hears.
pub fn transcribe(
    models: &Models,
    config: &Config,
    audio_path: &Path,
    lang: &str,
    vad_override: Option<bool>,
) -> Result<TranscribeResult, TranscribeError> {
    let start = Instant::now();
    let wav = ensure_wav(audio_path)?;
    let result = do_transcribe(models, config, wav.path(), lang, vad_override);
    let elapsed = start.elapsed().as_secs_f64();
    metrics::histogram!(names::TRANSCRIBE_DURATION, "lang" => lang.to_string()).record(elapsed);
    let mut res = result?;
    res.duration_ms = elapsed * 1000.0;
    Ok(res)
}

fn do_transcribe(
    models: &Models,
    config: &Config,
    wav_path: &Path,
    lang: &str,
    vad_override: Option<bool>,
) -> Result<TranscribeResult, TranscribeError> {
    let (samples, duration) = load_wav(wav_path)?;
    if config.max_audio_duration_s > 0.0 && duration > config.max_audio_duration_s {
        return Err(TranscribeError::TooLong(
            duration,
            config.max_audio_duration_s,
        ));
    }
    metrics::histogram!(names::AUDIO_DURATION).record(duration);

    let use_vad =
        vad_override.unwrap_or(duration >= config.vad_min_duration_s && models.vad.is_some());

    let max_chunk_samples = config.max_chunk_s * 16000;
    let audio_chunks = if use_vad {
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
            metrics::gauge!(names::VAD_SPEECH_RATIO, "lang" => lang.to_string()).set(ratio);
            metrics::counter!(names::CHUNKS_TOTAL, "lang" => lang.to_string())
                .increment(chunks_count as u64);
            vad_result.chunks
        } else {
            split_audio_chunks(samples, max_chunk_samples)
        }
    } else {
        split_audio_chunks(samples, max_chunk_samples)
    };

    let chunk_offsets = compute_chunk_offsets(&audio_chunks, 16000);
    let (texts, words) = transcribe_chunks(
        models,
        &audio_chunks,
        &chunk_offsets,
        config.hallucination_threshold,
    )?;

    // Parakeet writes case and punctuation itself.
    let joined = texts.join(" ");
    let text = sanitize_utf8(joined.trim());

    Ok(TranscribeResult {
        text,
        duration_ms: 0.0,
        audio_duration_ms: duration * 1000.0,
        words,
    })
}

/// Batch-decodes `chunks` on the Parakeet pool, in bounded groups.
pub(crate) fn transcribe_chunks(
    models: &Models,
    chunks: &[Vec<f32>],
    chunk_offsets: &[f64],
    threshold: f64,
) -> Result<(Vec<String>, Vec<WordTimestamp>), TranscribeError> {
    let label = PARAKEET_POOL_LABEL;
    let pool = models.parakeet_pool()?;
    let mut rec = pool.acquire().map_err(|e| {
        tracing::warn!("{label} pool acquire failed: {e}");
        match e {
            crate::pool::AcquireError::ReinitFailed(_) => TranscribeError::ReloadFailed,
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
}
