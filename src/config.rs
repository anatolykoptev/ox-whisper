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
    /// (PARAKEET_IDLE_EVICT_SECS, default: 0 = never; it does not inherit
    /// OX_WHISPER_IDLE_EVICT_SECS — a reload more than doubles RSS)
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
    /// Where uploads are written: the system temp dir. A field rather than a
    /// variable so tests can point a handler at a scratch directory.
    pub upload_dir: std::path::PathBuf,
    /// Longest audio a WebSocket session may buffer, seconds (WS_MAX_BUFFER_S,
    /// default: 120). Past it the session gets an error and is closed.
    pub ws_max_buffer_s: usize,
    /// Test-only: how long `transcribe` stalls before reading its input, so a
    /// test can drop the handler while the blocking job is running.
    #[cfg(test)]
    pub decode_delay: std::time::Duration,
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

fn real_env(name: &str) -> Option<String> {
    env::var(name).ok()
}

/// A variable parsed as `T`; unset or unparsable is `None`.
fn parse_env<T: std::str::FromStr>(get: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<T> {
    get(name).and_then(|v| v.parse().ok())
}

/// Parakeet is kept resident unless an operator opts in to eviction: a reload
/// holds the old and the new encoder at once, more than doubling RSS (3.2 GB to
/// 5.5 GB measured), and the next request pays the cold start. It does NOT
/// inherit `OX_WHISPER_IDLE_EVICT_SECS`, which is meant for the small models.
pub const PARAKEET_IDLE_EVICT_DEFAULT_SECS: u64 = 0;

/// Parses a numeric env var with a minimum. An unparsable or too-small value
/// warns and falls back to `default` — the repo's config convention.
fn env_num<T>(get: &dyn Fn(&str) -> Option<String>, name: &str, min: T, default: T) -> T
where
    T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display,
{
    match get(name) {
        Some(v) => match v.trim().parse::<T>() {
            Ok(n) if n >= min => n,
            _ => {
                tracing::warn!("{name}={v:?} invalid (want >= {min}); using default {default}");
                default
            }
        },
        None => default,
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
        let mut port = env_num(&real_env, "TTS_PORT", 1, TTS_PORT_DEFAULT);
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
            threads: env_num(&real_env, "TTS_THREADS", 1, default_threads),
            max_batch: env_num(&real_env, "TTS_MAX_BATCH", 1, 2),
            idle_stop_secs: env_num(&real_env, "TTS_IDLE_STOP_SECS", 0, 600),
            startup_timeout_secs: env_num(&real_env, "TTS_STARTUP_TIMEOUT_SECS", 1, 60),
            upstream_url: env::var("TTS_UPSTREAM_URL").ok().filter(|s| !s.is_empty()),
        }
    }
}

impl Config {
    /// The WebSocket buffer cap in samples at the 16 kHz the decoder assumes,
    /// tightened to `MAX_AUDIO_DURATION_S` when that is set: a stream may not
    /// buffer more than a batch upload may carry.
    pub fn ws_max_buffer_samples(&self) -> usize {
        let mut seconds = self.ws_max_buffer_s as f64;
        if self.max_audio_duration_s > 0.0 {
            seconds = seconds.min(self.max_audio_duration_s);
        }
        (seconds * 16000.0) as usize
    }

    /// Parses configuration from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        Self::from_lookup(&real_env)
    }

    /// Like [`Self::from_env`] over any variable source, so tests can set a
    /// configuration without touching the process environment.
    pub fn from_lookup(get: &dyn Fn(&str) -> Option<String>) -> Self {
        let port = parse_env(get, "MOONSHINE_PORT").unwrap_or(8092);
        let prom_port = parse_env(get, "OXWHISPER_PROM_PORT").unwrap_or(9092);
        let idle_evict_secs = parse_env(get, "OX_WHISPER_IDLE_EVICT_SECS").unwrap_or(0);
        let pool_size = parse_env(get, "POOL_SIZE").unwrap_or(2);
        Self {
            port,
            models_dir: get("MOONSHINE_MODELS_DIR").unwrap_or_else(|| "/models".to_string()),
            ru_models_dir: get("ZIPFORMER_RU_DIR").unwrap_or_else(|| "/ru-models".to_string()),
            parakeet_dir: get("PARAKEET_DIR").unwrap_or_else(|| "/parakeet-models".to_string()),
            parakeet_langs: parse_parakeet_langs(get("PARAKEET_LANGS").as_deref()),
            parakeet_pool_size: env_num(get, "PARAKEET_POOL_SIZE", 1, pool_size.max(1)),
            vad_model: get("SILERO_VAD_MODEL")
                .unwrap_or_else(|| "/vad/silero_vad.onnx".to_string()),
            punct_model: get("PUNCT_MODEL").unwrap_or_else(|| "/punct/model.int8.onnx".to_string()),
            punct_vocab: get("PUNCT_VOCAB").unwrap_or_else(|| "/punct/bpe.vocab".to_string()),
            num_threads: parse_env(get, "MOONSHINE_THREADS").unwrap_or(4),
            vad_min_duration_s: parse_env(get, "VAD_MIN_DURATION_S").unwrap_or(10.0),
            max_audio_duration_s: parse_env(get, "MAX_AUDIO_DURATION_S").unwrap_or(0.0),
            pool_size,
            pool_acquire_timeout_s: env_num(get, "POOL_ACQUIRE_TIMEOUT_S", 0, 30),
            vad_threshold: parse_env(get, "VAD_THRESHOLD").unwrap_or(0.5),
            vad_min_silence_s: parse_env(get, "VAD_MIN_SILENCE_S").unwrap_or(0.5),
            vad_speech_pad_s: parse_env(get, "VAD_SPEECH_PAD_S").unwrap_or(0.05),
            vad_min_speech_s: parse_env(get, "VAD_MIN_SPEECH_S").unwrap_or(0.25),
            vad_max_chunk_s: parse_env(get, "VAD_MAX_CHUNK_S").unwrap_or(20),
            max_chunk_s: parse_env(get, "MAX_CHUNK_S").unwrap_or(20),
            hallucination_threshold: parse_env(get, "HALLUCINATION_THRESHOLD").unwrap_or(2.4),
            max_body_size_mb: parse_env(get, "MAX_BODY_SIZE_MB").unwrap_or(50),
            provider: get("ONNX_PROVIDER").unwrap_or_else(|| "cpu".to_string()),
            diarize_segmentation_model: get("DIARIZE_SEGMENTATION_MODEL")
                .unwrap_or_else(|| "/diarize/segmentation.onnx".to_string()),
            diarize_embedding_model: get("DIARIZE_EMBEDDING_MODEL")
                .unwrap_or_else(|| "/diarize/embedding.onnx".to_string()),
            prom_port,
            idle_evict_secs,
            upload_dir: env::temp_dir(),
            ws_max_buffer_s: env_num(get, "WS_MAX_BUFFER_S", 1, 120),
            #[cfg(test)]
            decode_delay: std::time::Duration::ZERO,
            parakeet_idle_evict_secs: env_num(
                get,
                "PARAKEET_IDLE_EVICT_SECS",
                0,
                PARAKEET_IDLE_EVICT_DEFAULT_SECS,
            ),
            tts: TtsConfig::from_env(port, prom_port),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k| {
            pairs
                .iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| v.to_string())
        }
    }

    /// The global idle threshold is for the small models. Parakeet must stay
    /// resident unless its own variable says otherwise: a reload more than
    /// doubles RSS.
    #[test]
    fn parakeet_does_not_inherit_the_global_eviction_threshold() {
        let cfg = Config::from_lookup(&lookup(&[("OX_WHISPER_IDLE_EVICT_SECS", "600")]));
        assert_eq!(cfg.idle_evict_secs, 600, "the global knob still applies");
        assert_eq!(cfg.parakeet_idle_evict_secs, 0);
    }

    #[test]
    fn parakeet_eviction_is_opt_in_through_its_own_variable() {
        let cfg = Config::from_lookup(&lookup(&[
            ("OX_WHISPER_IDLE_EVICT_SECS", "600"),
            ("PARAKEET_IDLE_EVICT_SECS", "900"),
        ]));
        assert_eq!(cfg.parakeet_idle_evict_secs, 900);
    }

    #[test]
    fn websocket_buffer_cap_is_in_16khz_samples_and_bounded_by_the_batch_limit() {
        let cfg = Config::from_lookup(&lookup(&[]));
        assert_eq!(cfg.ws_max_buffer_samples(), 120 * 16000);

        let cfg = Config::from_lookup(&lookup(&[("WS_MAX_BUFFER_S", "30")]));
        assert_eq!(cfg.ws_max_buffer_samples(), 30 * 16000);

        let cfg = Config::from_lookup(&lookup(&[
            ("WS_MAX_BUFFER_S", "600"),
            ("MAX_AUDIO_DURATION_S", "60"),
        ]));
        assert_eq!(cfg.ws_max_buffer_samples(), 60 * 16000);

        // 0 would disable the cap: refused, the default applies.
        let cfg = Config::from_lookup(&lookup(&[("WS_MAX_BUFFER_S", "0")]));
        assert_eq!(cfg.ws_max_buffer_samples(), 120 * 16000);
    }
}
