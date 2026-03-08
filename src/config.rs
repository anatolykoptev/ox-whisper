use std::env;

/// Application configuration parsed from environment variables.
pub struct Config {
    /// Server port (MOONSHINE_PORT, default: 8092)
    pub port: u16,
    /// English models directory (MOONSHINE_MODELS_DIR, default: "/models")
    pub models_dir: String,
    /// Russian models directory (ZIPFORMER_RU_DIR, default: "/ru-models")
    pub ru_models_dir: String,
    /// Silero VAD model path (SILERO_VAD_MODEL, default: "/vad/silero_vad.onnx")
    pub vad_model: String,
    /// Punctuation model path (PUNCT_MODEL, default: "/punct/model.int8.onnx")
    pub punct_model: String,
    /// Punctuation BPE vocab path (PUNCT_VOCAB, default: "/punct/bpe.vocab")
    pub punct_vocab: String,
    /// Number of threads for inference (MOONSHINE_THREADS, default: 4)
    pub num_threads: i32,
    /// Minimum VAD segment duration in seconds (VAD_MIN_DURATION_S, default: 10.0)
    pub vad_min_duration_s: f64,
    /// Maximum audio duration in seconds (MAX_AUDIO_DURATION_S, default: 0 = no limit)
    pub max_audio_duration_s: f64,
    /// Number of recognizer instances per model (POOL_SIZE, default: 2)
    pub pool_size: usize,
    /// VAD speech probability threshold (VAD_THRESHOLD, default: 0.5)
    pub vad_threshold: f32,
    /// Minimum silence duration to split segments, seconds (VAD_MIN_SILENCE_S, default: 0.5)
    pub vad_min_silence_s: f32,
    /// Padding added around speech segments, seconds (VAD_SPEECH_PAD_S, default: 0.05)
    pub vad_speech_pad_s: f32,
    /// Minimum speech duration for VAD segments, seconds (VAD_MIN_SPEECH_S, default: 0.25)
    pub vad_min_speech_s: f32,
    /// Maximum chunk duration for VAD grouping, seconds (VAD_MAX_CHUNK_S, default: 20)
    pub vad_max_chunk_s: usize,
    /// Maximum chunk duration for non-VAD splitting, seconds (MAX_CHUNK_S, default: 20)
    pub max_chunk_s: usize,
    /// Compression ratio threshold for hallucination guard (HALLUCINATION_THRESHOLD, default: 2.4)
    pub hallucination_threshold: f64,
    /// Maximum upload body size in MB (MAX_BODY_SIZE_MB, default: 50)
    pub max_body_size_mb: usize,
    /// ONNX execution provider (ONNX_PROVIDER, default: "cpu")
    pub provider: String,
    /// Diarization segmentation model path (DIARIZE_SEGMENTATION_MODEL)
    pub diarize_segmentation_model: String,
    /// Diarization embedding model path (DIARIZE_EMBEDDING_MODEL)
    pub diarize_embedding_model: String,
}

impl Config {
    /// Parses configuration from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        Self {
            port: env::var("MOONSHINE_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8092),
            models_dir: env::var("MOONSHINE_MODELS_DIR")
                .unwrap_or_else(|_| "/models".to_string()),
            ru_models_dir: env::var("ZIPFORMER_RU_DIR")
                .unwrap_or_else(|_| "/ru-models".to_string()),
            vad_model: env::var("SILERO_VAD_MODEL")
                .unwrap_or_else(|_| "/vad/silero_vad.onnx".to_string()),
            punct_model: env::var("PUNCT_MODEL")
                .unwrap_or_else(|_| "/punct/model.int8.onnx".to_string()),
            punct_vocab: env::var("PUNCT_VOCAB")
                .unwrap_or_else(|_| "/punct/bpe.vocab".to_string()),
            num_threads: env::var("MOONSHINE_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            vad_min_duration_s: env::var("VAD_MIN_DURATION_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10.0),
            max_audio_duration_s: env::var("MAX_AUDIO_DURATION_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            pool_size: env::var("POOL_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            vad_threshold: env::var("VAD_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.5),
            vad_min_silence_s: env::var("VAD_MIN_SILENCE_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.5),
            vad_speech_pad_s: env::var("VAD_SPEECH_PAD_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.05),
            vad_min_speech_s: env::var("VAD_MIN_SPEECH_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.25),
            vad_max_chunk_s: env::var("VAD_MAX_CHUNK_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            max_chunk_s: env::var("MAX_CHUNK_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            hallucination_threshold: env::var("HALLUCINATION_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.4),
            max_body_size_mb: env::var("MAX_BODY_SIZE_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
            provider: env::var("ONNX_PROVIDER")
                .unwrap_or_else(|_| "cpu".to_string()),
            diarize_segmentation_model: env::var("DIARIZE_SEGMENTATION_MODEL")
                .unwrap_or_else(|_| "/diarize/segmentation.onnx".to_string()),
            diarize_embedding_model: env::var("DIARIZE_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "/diarize/embedding.onnx".to_string()),
        }
    }
}
