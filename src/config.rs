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
    /// Punctuation vocab path (PUNCT_VOCAB, default: "/punct/bpe.vocab")
    pub punct_vocab: String,
    /// Number of threads for inference (MOONSHINE_THREADS, default: 4)
    pub num_threads: i32,
    /// Minimum VAD segment duration in seconds (VAD_MIN_DURATION_S, default: 10.0)
    pub vad_min_duration_s: f64,
    /// Maximum audio duration in seconds (MAX_AUDIO_DURATION_S, default: 300.0)
    pub max_audio_duration_s: f64,
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
                .unwrap_or(300.0),
        }
    }
}
