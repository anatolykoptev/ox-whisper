use crate::config::Config;
use crate::models::Models;
use crate::vad::{apply_vad, lock_vad};
use crate::words::WordTimestamp;
use crate::ws_types::{Alternative, Channel, ServerMessage};

/// The session buffer is at its cap.
#[derive(Debug, PartialEq, Eq)]
pub struct BufferFull {
    pub max_samples: usize,
}

impl std::fmt::Display for BufferFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "audio buffer limit of {} s reached; send Finalize more often or enable vad",
            self.max_samples / 16000
        )
    }
}

/// Per-connection state for a WebSocket streaming session.
pub struct WsSession {
    pub sample_rate: u32,
    /// Most samples the buffer may hold. Enforced in [`Self::push_audio`], the
    /// only place that grows it.
    max_samples: usize,
    buffer: Vec<f32>,
    total_samples: usize,
    speech_detected: bool,
    last_interim_samples: usize,
}

impl WsSession {
    pub fn new(sample_rate: u32, max_samples: usize) -> Self {
        Self {
            sample_rate,
            max_samples,
            buffer: Vec::new(),
            total_samples: 0,
            speech_detected: false,
            last_interim_samples: 0,
        }
    }

    /// Decode and append incoming audio data to the internal buffer.
    ///
    /// Refuses audio that would take the buffer past its cap (nothing is
    /// appended): an unbounded buffer is decoded again in full for every
    /// interim result, which pins the recognizer and grows memory without
    /// limit.
    pub fn push_audio(&mut self, data: &[u8], encoding: &str) -> Result<(), BufferFull> {
        let samples = decode_pcm(data, encoding);
        if self.buffer.len() + samples.len() > self.max_samples {
            return Err(BufferFull {
                max_samples: self.max_samples,
            });
        }
        self.total_samples += samples.len();
        self.buffer.extend(samples);
        Ok(())
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

    /// Take the accumulated audio buffer for transcription (destructive).
    pub fn take_buffer(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.buffer)
    }

    /// Clone the current buffer for interim transcription (non-destructive).
    pub fn peek_buffer(&self) -> Vec<f32> {
        self.buffer.clone()
    }

    /// Current timestamp in seconds based on total samples received.
    pub fn timestamp_s(&self) -> f64 {
        self.total_samples as f64 / self.sample_rate as f64
    }

    /// Run VAD on the current buffer. Returns server messages and whether speech ended.
    /// Only triggers speech_final when we have >= 1s of audio and VAD found segments.
    pub fn run_vad_check(
        &mut self,
        models: &Models,
        config: &Config,
    ) -> (Vec<ServerMessage>, bool) {
        let mut messages = Vec::new();

        // Need at least 1s of audio for meaningful VAD
        if self.buffer.len() < self.sample_rate as usize {
            return (messages, false);
        }

        let vad_mutex = match models.vad {
            Some(ref v) => v,
            None => return (messages, false),
        };
        let mut vad = lock_vad(vad_mutex);

        let result = apply_vad(
            &mut vad,
            &self.buffer,
            self.sample_rate,
            config.vad_speech_pad_s,
            config.vad_max_chunk_s,
            // Runs on every frame past 1 s, so silence is the normal outcome
            // here: a separate label keeps `ws` free of expected silence.
            "ws_poll",
        );

        let has_speech = !result.chunks.is_empty() && result.speech_ms > 0.0;

        if has_speech && !self.speech_detected {
            self.speech_detected = true;
            let offset = (self.total_samples - self.buffer.len()) as f64 / self.sample_rate as f64;
            messages.push(ServerMessage::SpeechStarted {
                timestamp_s: offset,
            });
        }

        // Speech final = VAD found speech AND speech portion is shorter than buffer
        // (meaning there's trailing silence — utterance ended)
        let speech_samples: usize = result.chunks.iter().map(|c| c.len()).sum();
        let speech_final = has_speech && speech_samples < self.buffer.len() * 9 / 10;

        if speech_final {
            // Replace buffer with just the speech chunks for cleaner transcription
            self.buffer = result.chunks.into_iter().flatten().collect();
            self.speech_detected = false;
        }

        (messages, speech_final)
    }

    /// Create a final Results message from transcription output.
    pub fn store_final(
        &self,
        text: String,
        words: Vec<WordTimestamp>,
        from_finalize: bool,
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
    if confs.is_empty() {
        0.0
    } else {
        confs.iter().sum::<f32>() / confs.len() as f32
    }
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

    /// The per-frame VAD check meets silence all the time; it must not feed the
    /// `ws` no-speech counter, which alerts treat as a real empty transcript.
    #[test]
    fn polling_silence_is_not_counted_as_a_ws_no_speech_result() {
        let recorder = crate::metrics::test_recorder::CountingRecorder::default();
        let mut config = Config::from_lookup(&|_| None);
        config.vad_model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/vad/silero_vad.onnx")
            .to_string_lossy()
            .into_owned();
        // Sample-buffer capacity only; the default allocates an hour of audio.
        config.max_audio_duration_s = 60.0;
        let mut models = Models::empty();
        models.vad = crate::models::load_vad(&config);
        assert!(models.vad.is_some(), "VAD fixture loads");

        let mut session = WsSession::new(16000, 16000 * 60);
        session
            .push_audio(&vec![0u8; 2 * 16000 * 2], "pcm_s16le")
            .unwrap();
        let (_, speech_final) =
            metrics::with_local_recorder(&recorder, || session.run_vad_check(&models, &config));

        assert!(!speech_final);
        let count = |caller| recorder.count("oxwhisper_vad_no_speech_total", &[("caller", caller)]);
        assert_eq!(count("ws_poll"), 1, "the check ran and saw silence");
        assert_eq!(count("ws"), 0);
    }

    #[test]
    fn the_buffer_never_grows_past_its_cap() {
        let mut session = WsSession::new(16000, 8);
        // 4 samples (8 bytes) at a time.
        assert_eq!(session.push_audio(&[0u8; 8], "pcm_s16le"), Ok(()));
        assert_eq!(session.push_audio(&[0u8; 8], "pcm_s16le"), Ok(()));
        assert_eq!(
            session.push_audio(&[0u8; 2], "pcm_s16le"),
            Err(BufferFull { max_samples: 8 })
        );
        assert_eq!(
            session.peek_buffer().len(),
            8,
            "a refused push appends nothing"
        );
        // Taking the buffer (finalize / speech end) frees the room again.
        session.take_buffer();
        assert_eq!(session.push_audio(&[0u8; 8], "pcm_s16le"), Ok(()));
    }

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
