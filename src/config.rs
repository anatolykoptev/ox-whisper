use std::env;

/// Application configuration parsed from environment variables.
pub struct Config {
    /// Server port (MOONSHINE_PORT, default: 8092)
    pub port: u16,
    /// English models directory (MOONSHINE_MODELS_DIR, default: "/models")
    pub models_dir: String,
    /// Russian models directory (ZIPFORMER_RU_DIR, default: "/ru-models")
    pub ru_models_dir: String,
    /// Parakeet TDT v3 model directory (PARAKEET_DIR, default: "/parakeet-models")
    pub parakeet_dir: String,
    /// Languages routed to Parakeet (PARAKEET_LANGS, comma-separated, default:
    /// the 25 European languages Parakeet TDT v3 covers). An empty value or
    /// `off` disables Parakeet: it is not loaded and every language falls back
    /// to the Zipformer (ru) / Moonshine (other) route — a rollback without a
    /// rebuild.
    pub parakeet_langs: Vec<String>,
    /// Idle-eviction threshold for Parakeet, seconds
    /// (PARAKEET_IDLE_EVICT_SECS, default: OX_WHISPER_IDLE_EVICT_SECS, 0 = never)
    pub parakeet_idle_evict_secs: u64,
    /// Parakeet recognizer instances (PARAKEET_POOL_SIZE, default: POOL_SIZE).
    /// Each instance holds its own ~1.1 GB encoder session.
    pub parakeet_pool_size: usize,
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
    /// How long a request waits for a busy recognizer before failing, seconds
    /// (POOL_ACQUIRE_TIMEOUT_S, default: 30)
    pub pool_acquire_timeout_s: u64,
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
    /// Prometheus metrics port (OXWHISPER_PROM_PORT, default: 9092)
    pub prom_port: u16,
    /// Idle eviction threshold in seconds (OX_WHISPER_IDLE_EVICT_SECS, default: 0 = disabled)
    pub idle_evict_secs: u64,
    /// Text-to-speech child process settings (TTS_* env vars)
    pub tts: TtsConfig,
}

/// Text-to-speech child-process configuration.
///
/// The TTS engine is an external `tts-server` binary run as a child process on
/// the loopback interface. When `upstream_url` is set no child is managed and
/// the supervisor hands out that URL instead.
// Several fields are read only by tts::supervisor — dead in the bin target
// until the speech proxy lands (see src/tts/mod.rs).
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct TtsConfig {
    /// Enable TTS supervision (TTS_ENABLED, default: false)
    pub enabled: bool,
    /// Path to the tts-server binary (TTS_BIN, default: "/opt/qwentts/tts-server")
    pub bin: String,
    /// Talker model path, passed as `--model` (TTS_MODEL)
    pub model: String,
    /// Codec model path, passed as `--codec` (TTS_CODEC)
    pub codec: String,
    /// Loopback port the child binds (TTS_PORT, default: 8093)
    pub port: u16,
    /// CPU threads for the child via `QT_N_THREADS` (TTS_THREADS,
    /// default: max(1, available_parallelism - 1))
    pub threads: usize,
    /// `--max-batch` value (TTS_MAX_BATCH, default: 2)
    pub max_batch: usize,
    /// Stop the child after this many seconds without in-flight requests
    /// (TTS_IDLE_STOP_SECS, default: 600, 0 = never stop)
    pub idle_stop_secs: u64,
    /// Startup health-check timeout (TTS_STARTUP_TIMEOUT_SECS, default: 60)
    pub startup_timeout_secs: u64,
    /// External TTS endpoint; when set no child is spawned (TTS_UPSTREAM_URL)
    pub upstream_url: Option<String>,
}

/// Languages Parakeet TDT 0.6B v3 transcribes (its model card's 25 European
/// languages), as ISO 639-1 codes.
pub const PARAKEET_V3_LANGS: &[&str] = &[
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];

/// Parses `PARAKEET_LANGS`. Unset → every Parakeet v3 language; empty or
/// `off`/`none` → no language (Parakeet disabled).
pub fn parse_parakeet_langs(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw else {
        return PARAKEET_V3_LANGS.iter().map(|l| l.to_string()).collect();
    };
    let trimmed = raw.trim().to_lowercase();
    if matches!(trimmed.as_str(), "" | "off" | "none") {
        return Vec::new();
    }
    let mut langs: Vec<String> = Vec::new();
    for lang in trimmed.split(',').map(str::trim).filter(|l| !l.is_empty()) {
        if !PARAKEET_V3_LANGS.contains(&lang) {
            tracing::warn!("PARAKEET_LANGS: '{lang}' is not a Parakeet v3 language; ignoring it");
            continue;
        }
        if !langs.iter().any(|l| l == lang) {
            langs.push(lang.to_string());
        }
    }
    langs
}

/// Parses a numeric env var with a minimum. An unparsable or too-small value
/// warns and falls back to `default` — the repo's config convention.
fn env_num<T>(name: &str, min: T, default: T) -> T
where
    T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display,
{
    match env::var(name) {
        Ok(v) => match v.trim().parse::<T>() {
            Ok(n) if n >= min => n,
            _ => {
                tracing::warn!("{name}={v:?} invalid (want >= {min}); using default {default}");
                default
            }
        },
        Err(_) => default,
    }
}

impl TtsConfig {
    /// Parses TTS configuration from environment variables. `moonshine_port`
    /// and `prom_port` are the ports already claimed by this process — the
    /// TTS child must not share them.
    pub fn from_env(moonshine_port: u16, prom_port: u16) -> Self {
        const TTS_PORT_DEFAULT: u16 = 8093;
        let default_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let mut port = env_num("TTS_PORT", 1, TTS_PORT_DEFAULT);
        if port == moonshine_port || port == prom_port {
            tracing::warn!(
                "TTS_PORT={port} collides with the STT or metrics port; using {TTS_PORT_DEFAULT}"
            );
            port = TTS_PORT_DEFAULT;
        }
        if port == moonshine_port || port == prom_port {
            tracing::warn!("default TTS port {port} also collides — set TTS_PORT to a free port");
        }
        Self {
            enabled: env::var("TTS_ENABLED")
                .map(|v| matches!(v.trim().to_lowercase().as_str(), "true" | "1"))
                .unwrap_or(false),
            bin: env::var("TTS_BIN").unwrap_or_else(|_| "/opt/qwentts/tts-server".to_string()),
            model: env::var("TTS_MODEL").unwrap_or_default(),
            codec: env::var("TTS_CODEC").unwrap_or_default(),
            port,
            threads: env_num("TTS_THREADS", 1, default_threads),
            max_batch: env_num("TTS_MAX_BATCH", 1, 2),
            idle_stop_secs: env_num("TTS_IDLE_STOP_SECS", 0, 600),
            startup_timeout_secs: env_num("TTS_STARTUP_TIMEOUT_SECS", 1, 60),
            upstream_url: env::var("TTS_UPSTREAM_URL").ok().filter(|s| !s.is_empty()),
        }
    }
}

impl Config {
    /// Parses configuration from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        let port = env::var("MOONSHINE_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8092);
        let prom_port = env::var("OXWHISPER_PROM_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9092);
        let idle_evict_secs = env::var("OX_WHISPER_IDLE_EVICT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let pool_size = env::var("POOL_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        Self {
            port,
            models_dir: env::var("MOONSHINE_MODELS_DIR").unwrap_or_else(|_| "/models".to_string()),
            ru_models_dir: env::var("ZIPFORMER_RU_DIR")
                .unwrap_or_else(|_| "/ru-models".to_string()),
            parakeet_dir: env::var("PARAKEET_DIR")
                .unwrap_or_else(|_| "/parakeet-models".to_string()),
            parakeet_langs: parse_parakeet_langs(env::var("PARAKEET_LANGS").ok().as_deref()),
            parakeet_pool_size: env_num("PARAKEET_POOL_SIZE", 1, pool_size.max(1)),
            vad_model: env::var("SILERO_VAD_MODEL")
                .unwrap_or_else(|_| "/vad/silero_vad.onnx".to_string()),
            punct_model: env::var("PUNCT_MODEL")
                .unwrap_or_else(|_| "/punct/model.int8.onnx".to_string()),
            punct_vocab: env::var("PUNCT_VOCAB").unwrap_or_else(|_| "/punct/bpe.vocab".to_string()),
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
            pool_size,
            pool_acquire_timeout_s: env_num("POOL_ACQUIRE_TIMEOUT_S", 0, 30),
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
            provider: env::var("ONNX_PROVIDER").unwrap_or_else(|_| "cpu".to_string()),
            diarize_segmentation_model: env::var("DIARIZE_SEGMENTATION_MODEL")
                .unwrap_or_else(|_| "/diarize/segmentation.onnx".to_string()),
            diarize_embedding_model: env::var("DIARIZE_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "/diarize/embedding.onnx".to_string()),
            prom_port,
            idle_evict_secs,
            parakeet_idle_evict_secs: env_num("PARAKEET_IDLE_EVICT_SECS", 0, idle_evict_secs),
            tts: TtsConfig::from_env(port, prom_port),
        }
    }
}
