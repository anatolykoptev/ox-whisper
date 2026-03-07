use sherpa_rs::silero_vad::SileroVad;

pub struct VadResult {
    pub chunks: Vec<Vec<f32>>,
    pub speech_ms: f64,
}

const WINDOW_SIZE: usize = 512;
const MAX_CHUNK_SECONDS: usize = 5;
const PAD_SAMPLES: usize = 800; // 50ms of silence at 16kHz

/// Applies Voice Activity Detection to split audio into speech chunks.
///
/// Feeds samples through Silero VAD in 512-sample windows, collects
/// speech segments, and groups them into chunks of at most 5 seconds.
/// Single segments longer than the limit are force-split.
pub fn apply_vad(vad: &mut SileroVad, samples: &[f32], sample_rate: u32) -> VadResult {
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

    // Group segments into chunks, force-splitting segments exceeding the limit
    let max_chunk_samples = MAX_CHUNK_SECONDS * sample_rate as usize;
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    let mut current_chunk: Vec<f32> = Vec::new();

    for segment in segments {
        // Prepend 50ms of silence to reduce boundary artifacts
        let mut padded = Vec::with_capacity(PAD_SAMPLES + segment.samples.len());
        padded.extend(std::iter::repeat_n(0.0f32, PAD_SAMPLES));
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
