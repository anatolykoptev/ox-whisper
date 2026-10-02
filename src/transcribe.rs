use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::metrics::names;

use flate2::Compression;
use flate2::write::ZlibEncoder;

use crate::audio::{AudioError, ensure_wav, load_wav};
use crate::chunking::{sanitize_utf8, split_at_quiet};
use crate::config::Config;
use crate::models::{Models, PARAKEET_POOL_LABEL};
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

/// `lang` only labels metrics (the validated request language, or `auto`):
/// the one model transcribes whatever language it hears.
pub fn transcribe(
    models: &Models,
    config: &Config,
    audio_path: &Path,
    lang: &str,
) -> Result<TranscribeResult, TranscribeError> {
    let start = Instant::now();
    #[cfg(test)]
    probe::enter(audio_path, config.decode_delay);
    let wav = ensure_wav(audio_path, &config.upload_dir)?;
    let result = do_transcribe(models, config, wav.path(), lang);
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
) -> Result<TranscribeResult, TranscribeError> {
    let (samples, duration) = load_wav(wav_path)?;
    if config.max_audio_duration_s > 0.0 && duration > config.max_audio_duration_s {
        return Err(TranscribeError::TooLong(
            duration,
            config.max_audio_duration_s,
        ));
    }
    metrics::histogram!(names::AUDIO_DURATION).record(duration);

    // Contiguous original audio, cut only at quiet points: nothing is dropped,
    // nothing is inserted. Speech detection (VAD) trimmed late onsets off quiet
    // clips and padded segments with digital zeros, which cost 1-4 WER points.
    let audio_chunks = plan_chunks(samples, config);
    metrics::counter!(names::CHUNKS_TOTAL, "lang" => lang.to_string())
        .increment(audio_chunks.len() as u64);

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

/// The chunks one decode is made of: the contiguous original audio, cut only at
/// quiet points. Shared by the batch endpoint and the WebSocket final decode so
/// neither drops, trims or pads audio.
pub(crate) fn plan_chunks(samples: Vec<f32>, config: &Config) -> Vec<Vec<f32>> {
    split_at_quiet(samples, config.max_chunk_samples())
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
/// `MAX_CHUNK_S=30` this bounds a call to about 120 s of audio.
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

#[cfg(test)]
mod plan_tests {
    use super::*;

    /// The chunk list both decode paths build. A clip whose first 3 s sit at
    /// about -80 dBFS (Silero starts such a segment seconds late, and the old
    /// path then dropped the head) must reach the model whole: no sample dropped,
    /// no zeros inserted, every chunk within the window.
    #[test]
    fn the_decode_path_neither_drops_nor_pads_a_quiet_onset() {
        let config = Config::from_lookup(&|_| None);
        let mut x: Vec<f32> = (0..16000 * 70)
            .map(|i| 0.3 * ((i as f32) * 0.21).sin())
            .collect();
        for s in &mut x[..16000 * 3] {
            *s *= 0.0001;
        }
        let chunks = plan_chunks(x.clone(), &config);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.len() <= config.max_chunk_samples()));
        let back: Vec<f32> = chunks.iter().flatten().copied().collect();
        assert_eq!(back.len(), x.len(), "samples dropped or inserted");
        assert!(back.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(back[..16000 * 3], x[..16000 * 3]);
    }

    #[test]
    fn audio_up_to_the_window_decodes_in_one_piece() {
        let config = Config::from_lookup(&|_| None);
        let chunks = plan_chunks(vec![0.1; config.max_chunk_samples()], &config);
        assert_eq!(chunks.len(), 1);
    }
}
