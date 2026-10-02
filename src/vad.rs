use sherpa_rs::silero_vad::SileroVad;

pub struct VadResult {
    pub chunks: Vec<Vec<f32>>,
    pub speech_ms: f64,
}

const WINDOW_SIZE: usize = 512;

/// Applies Voice Activity Detection to split audio into speech chunks.
///
/// Feeds samples through Silero VAD in 512-sample windows, collects
/// speech segments, and groups them into chunks of at most `max_chunk_s` seconds.
/// Single segments longer than the limit are force-split.
pub fn apply_vad(
    vad: &mut SileroVad,
    samples: &[f32],
    sample_rate: u32,
    pad_s: f32,
    max_chunk_s: usize,
) -> VadResult {
    let pad_samples = (pad_s * sample_rate as f32) as usize;
    // One detector is shared by every request. Start from a clean state:
    // `clear()` below only drops finished segments, so without a reset the
    // previous request's model state and buffer decide how this audio is
    // segmented — identical input then yields different chunks, sometimes
    // none at all (an empty transcript with HTTP 200).
    vad.reset();
    // Feed 512-sample windows
    let mut offset = 0;
    while offset + WINDOW_SIZE <= samples.len() {
        let window = samples[offset..offset + WINDOW_SIZE].to_vec();
        vad.accept_waveform(window);
        offset += WINDOW_SIZE;
    }

    // Pad remainder to 512 if any
    if offset < samples.len() {
        let mut padded = samples[offset..].to_vec();
        padded.resize(WINDOW_SIZE, 0.0);
        vad.accept_waveform(padded);
    }

    // Flush to finalize pending segments
    vad.flush();

    // Drain all detected speech segments
    let mut segments = Vec::new();
    while !vad.is_empty() {
        segments.push(vad.front());
        vad.pop();
    }

    // Calculate total speech duration in ms
    let speech_ms: f64 = segments
        .iter()
        .map(|s| s.samples.len() as f64 / sample_rate as f64 * 1000.0)
        .sum();

    // Group segments into chunks, force-splitting segments exceeding the limit.
    // A max_chunk_s of 0 means "no limit" — avoid the infinite loop that a
    // zero capacity would otherwise create.
    let max_chunk_samples = max_chunk_s * sample_rate as usize;
    let max_chunk_samples = if max_chunk_samples == 0 {
        usize::MAX
    } else {
        max_chunk_samples
    };
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    let mut current_chunk: Vec<f32> = Vec::new();

    for segment in segments {
        // Prepend 50ms of silence to reduce boundary artifacts
        let mut padded = Vec::with_capacity(pad_samples + segment.samples.len());
        padded.extend(std::iter::repeat_n(0.0f32, pad_samples));
        padded.extend_from_slice(&segment.samples);
        let mut seg_samples = &padded[..];

        // Force-split segments longer than max
        while !seg_samples.is_empty() {
            let remaining_capacity = max_chunk_samples.saturating_sub(current_chunk.len());
            if remaining_capacity == 0 {
                chunks.push(std::mem::take(&mut current_chunk));
                continue;
            }

            let take = seg_samples.len().min(remaining_capacity);
            current_chunk.extend_from_slice(&seg_samples[..take]);
            seg_samples = &seg_samples[take..];

            if current_chunk.len() >= max_chunk_samples {
                chunks.push(std::mem::take(&mut current_chunk));
            }
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    // Clear VAD state for reuse
    vad.clear();

    VadResult { chunks, speech_ms }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/vad")
            .join(name)
    }

    fn samples(name: &str) -> Vec<f32> {
        crate::audio::load_wav(&fixture(name))
            .expect("fixture wav")
            .0
    }

    /// The production VAD (same loader and settings as `Models::load`) applied
    /// with the production pad and chunk length.
    fn vad_run(vad: &mut SileroVad, samples: &[f32]) -> VadResult {
        let config = crate::config::Config::from_env();
        apply_vad(
            vad,
            samples,
            16000,
            config.vad_speech_pad_s,
            config.vad_max_chunk_s,
        )
    }

    fn production_vad() -> SileroVad {
        let mut config = crate::config::Config::from_env();
        config.vad_model = fixture("silero_vad.onnx").to_string_lossy().into_owned();
        // Sample-buffer capacity only; the default allocates an hour of audio.
        config.max_audio_duration_s = 60.0;
        crate::models::load_vad(&config)
            .expect("VAD fixture loads")
            .into_inner()
            .expect("fresh mutex")
    }

    /// The detector is shared by all requests. Segmenting a clip must not
    /// depend on what the previous request fed it. Fixture pair from the
    /// 2026-10-01 eval: after clip `a`, the un-reset detector returned zero
    /// segments for clip `b` (an empty transcript with HTTP 200).
    #[test]
    fn segmentation_does_not_depend_on_the_previous_request() {
        let a = samples("fleurs_en_a.wav");
        let b = samples("fleurs_en_b.wav");

        let fresh = vad_run(&mut production_vad(), &b);
        assert!(!fresh.chunks.is_empty(), "clip b is speech");

        let mut shared = production_vad();
        let _ = vad_run(&mut shared, &a);
        let after_a = vad_run(&mut shared, &b);
        assert_eq!(after_a.chunks.len(), fresh.chunks.len());
        assert_eq!(after_a.speech_ms, fresh.speech_ms);
        assert!(
            after_a.chunks == fresh.chunks,
            "chunks differ after a prior request"
        );

        // And the same clip twice in a row gives the same segmentation.
        let again = vad_run(&mut shared, &b);
        assert!(again.chunks == fresh.chunks);
    }
}
