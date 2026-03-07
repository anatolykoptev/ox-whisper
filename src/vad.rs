use sherpa_rs::silero_vad::SileroVad;

pub struct VadResult {
    pub chunks: Vec<Vec<f32>>,
    pub speech_ms: f64,
}

const WINDOW_SIZE: usize = 512;
const MAX_CHUNK_SECONDS: usize = 25;

/// Applies Voice Activity Detection to split audio into speech chunks.
///
/// Feeds samples through Silero VAD in 512-sample windows, collects
/// speech segments, and groups them into chunks of at most 25 seconds.
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

    // Group segments into chunks (max 25s per chunk)
    let max_chunk_samples = MAX_CHUNK_SECONDS * sample_rate as usize;
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    let mut current_chunk: Vec<f32> = Vec::new();

    for segment in segments {
        if !current_chunk.is_empty()
            && current_chunk.len() + segment.samples.len() > max_chunk_samples
        {
            chunks.push(std::mem::take(&mut current_chunk));
        }
        current_chunk.extend_from_slice(&segment.samples);
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    // Clear VAD state for reuse
    vad.clear();

    VadResult { chunks, speech_ms }
}
