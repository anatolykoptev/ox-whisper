use crate::config::Config;
use crate::models::Models;
use crate::vad::apply_vad;
use crate::words::WordTimestamp;
use crate::ws_types::{Alternative, Channel, ServerMessage};

/// Per-connection state for a WebSocket streaming session.
pub struct WsSession {
    pub sample_rate: u32,
    buffer: Vec<f32>,
    total_samples: usize,
    speech_detected: bool,
    last_interim_samples: usize,
}

impl WsSession {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            buffer: Vec::new(),
            total_samples: 0,
            speech_detected: false,
            last_interim_samples: 0,
        }
    }

    /// Decode and append incoming audio data to the internal buffer.
    pub fn push_audio(&mut self, data: &[u8], encoding: &str) {
        let samples = decode_pcm(data, encoding);
        self.total_samples += samples.len();
        self.buffer.extend(samples);
    }

    /// Check if enough audio has accumulated for an interim result.
    pub fn should_emit_interim(&self, interval_s: f32) -> bool {
        let threshold = (interval_s * self.sample_rate as f32) as usize;
        (self.total_samples - self.last_interim_samples) >= threshold
    }

    /// Mark that an interim result was emitted at the current position.
    pub fn mark_interim(&mut self) {
        self.last_interim_samples = self.total_samples;
    }

    /// Take the accumulated audio buffer for transcription.
    pub fn take_buffer(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.buffer)
    }

    /// Current timestamp in seconds based on total samples received.
    pub fn timestamp_s(&self) -> f64 {
        self.total_samples as f64 / self.sample_rate as f64
    }

    /// Run VAD on the current buffer. Returns server messages and whether speech ended.
    pub fn run_vad_check(
        &mut self, models: &Models, config: &Config,
    ) -> (Vec<ServerMessage>, bool) {
        let mut messages = Vec::new();
        let vad_mutex = match models.vad {
            Some(ref v) => v,
            None => return (messages, false),
        };
        let mut vad = match vad_mutex.lock() {
            Ok(v) => v,
            Err(_) => return (messages, false),
        };

        let result = apply_vad(
            &mut vad, &self.buffer, self.sample_rate,
            config.vad_speech_pad_s, config.vad_max_chunk_s,
        );

        let has_speech = !result.chunks.is_empty() && result.speech_ms > 0.0;

        if has_speech && !self.speech_detected {
            self.speech_detected = true;
            messages.push(ServerMessage::SpeechStarted {
                timestamp_s: self.timestamp_s(),
            });
        }

        // Speech final = VAD found speech followed by silence
        let speech_final = self.speech_detected
            && result.chunks.len() >= 1
            && result.speech_ms > 0.0;

        if speech_final {
            self.speech_detected = false;
        }

        (messages, speech_final)
    }

    /// Create a final Results message from transcription output.
    pub fn store_final(
        &self, text: String, words: Vec<WordTimestamp>, from_finalize: bool,
    ) -> ServerMessage {
        let confidence = avg_word_confidence(&words);
        ServerMessage::Results {
            is_final: true,
            speech_final: !from_finalize,
            from_finalize,
            channel: Channel {
                alternatives: vec![Alternative {
                    transcript: text,
                    confidence,
                    words,
                }],
            },
            speech_started_s: None,
        }
    }

    /// Create an interim (non-final) Results message.
    pub fn interim_result(&self, text: String, words: Vec<WordTimestamp>) -> ServerMessage {
        let confidence = avg_word_confidence(&words);
        ServerMessage::Results {
            is_final: false,
            speech_final: false,
            from_finalize: false,
            channel: Channel {
                alternatives: vec![Alternative {
                    transcript: text,
                    confidence,
                    words,
                }],
            },
            speech_started_s: None,
        }
    }
}

fn avg_word_confidence(words: &[WordTimestamp]) -> f32 {
    let confs: Vec<f32> = words.iter().filter_map(|w| w.confidence).collect();
    if confs.is_empty() { 0.0 } else { confs.iter().sum::<f32>() / confs.len() as f32 }
}

/// Decode raw PCM bytes into f32 samples.
pub fn decode_pcm(data: &[u8], encoding: &str) -> Vec<f32> {
    match encoding {
        "pcm_f32le" | "f32le" => {
            // Every 4 bytes is one f32 sample (little-endian)
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        _ => {
            // Default: pcm_s16le — every 2 bytes is one i16 sample
            data.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_s16le_zeros() {
        let data = [0u8; 4]; // two zero samples
        let samples = decode_pcm(&data, "pcm_s16le");
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.0).abs() < f32::EPSILON);
        assert!((samples[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn decode_s16le_max() {
        // i16::MAX = 32767 → 32767/32768 ≈ 0.99997
        let bytes = 32767_i16.to_le_bytes();
        let samples = decode_pcm(&bytes, "pcm_s16le");
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.99997).abs() < 0.001);
    }

    #[test]
    fn decode_f32le() {
        let val: f32 = 0.5;
        let bytes = val.to_le_bytes();
        let samples = decode_pcm(&bytes, "pcm_f32le");
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn decode_s16le_odd_bytes() {
        // 3 bytes → only 1 complete sample (first 2 bytes)
        let data = [0u8, 0, 0xFF];
        let samples = decode_pcm(&data, "pcm_s16le");
        assert_eq!(samples.len(), 1);
    }
}
