use std::env;

/// Application configuration parsed from environment variables.
pub struct Config {
    /// Server port (MOONSHINE_PORT, default: 8092 — the name predates Parakeet
    /// and is kept because deployed compose files set it)
    pub port: u16,
    /// Parakeet TDT v3 model directory (PARAKEET_DIR, default: "/parakeet-models")
    pub parakeet_dir: String,
    /// Idle-eviction threshold for Parakeet, seconds
    /// (PARAKEET_IDLE_EVICT_SECS, default: 0 = never; a reload more than
    /// doubles RSS)
    pub parakeet_idle_evict_secs: u64,
    /// Parakeet recognizer instances (PARAKEET_POOL_SIZE, default: 1).
    /// Each instance holds its own ~3.2 GB encoder session.
    pub parakeet_pool_size: usize,
    /// Silero VAD model path (SILERO_VAD_MODEL, default: "/vad/silero_vad.onnx")
    pub vad_model: String,
    /// Number of threads for inference (MOONSHINE_THREADS, default: 4 — name
    /// kept for deployed compose files)
    pub num_threads: i32,
    /// Maximum audio duration in seconds (MAX_AUDIO_DURATION_S, default: 0 = no limit)
    pub max_audio_duration_s: f64,
    /// How long a request waits for a busy recognizer before failing, seconds
    /// (POOL_ACQUIRE_TIMEOUT_S, default: 30)
    pub pool_acquire_timeout_s: u64,
    /// VAD speech probability threshold, WebSocket speech detection only (VAD_THRESHOLD, default: 0.5)
    pub vad_threshold: f32,
    /// Minimum silence duration to split segments, seconds (VAD_MIN_SILENCE_S, default: 0.5)
    pub vad_min_silence_s: f32,
    /// Padding added around speech segments, seconds (VAD_SPEECH_PAD_S, default: 0.05)
    pub vad_speech_pad_s: f32,
    /// Minimum speech duration for VAD segments, seconds (VAD_MIN_SPEECH_S, default: 0.25)
    pub vad_min_speech_s: f32,
    /// Maximum chunk duration for VAD grouping, seconds (VAD_MAX_CHUNK_S, default: 20)
    pub vad_max_chunk_s: usize,
    /// Decode window, seconds (MAX_CHUNK_S, default: 30): audio longer than this is
    /// cut at the quietest point of each window's last fifth.
    pub max_chunk_s: usize,
    /// Compression ratio threshold for hallucination guard (HALLUCINATION_THRESHOLD, default: 2.4)
    pub hallucination_threshold: f64,
    /// Maximum upload body size in MB (MAX_BODY_SIZE_MB, default: 50)
    pub max_body_size_mb: usize,
    /// ONNX execution provider (ONNX_PROVIDER, default: "cpu")
    pub provider: String,
    /// Prometheus metrics port (OXWHISPER_PROM_PORT, default: 9092)
    pub prom_port: u16,
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
}

/// Settings that older versions read and this one ignores. An operator who
/// still sets one expects an effect — `PARAKEET_LANGS=off` was the rollback to
/// the old models — so startup says it is dead instead of leaving a silent
/// no-op.
const REMOVED_SETTINGS: &[&str] = &[
    "PARAKEET_LANGS",
    "ZIPFORMER_RU_DIR",
    "MOONSHINE_MODELS_DIR",
    "PUNCT_MODEL",
    "PUNCT_VOCAB",
    "DIARIZE_SEGMENTATION_MODEL",
    "DIARIZE_EMBEDDING_MODEL",
    "POOL_SIZE",
    "OX_WHISPER_IDLE_EVICT_SECS",
    "TTS_ENABLED",
    // Batch and WebSocket-final decodes no longer run VAD: they decode contiguous
    // audio cut at quiet points, so there is no minimum duration to switch on.
    "VAD_MIN_DURATION_S",
];

/// The [`REMOVED_SETTINGS`] that `get` finds set.
pub fn removed_settings_present(get: &dyn Fn(&str) -> Option<String>) -> Vec<&'static str> {
    REMOVED_SETTINGS
        .iter()
        .copied()
        .filter(|name| get(name).is_some())
        .collect()
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

impl Config {
    /// The decode window in 16 kHz samples: the longest chunk one decode call gets.
    pub fn max_chunk_samples(&self) -> usize {
        self.max_chunk_s * 16000
    }

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
        Self {
            port: parse_env(get, "MOONSHINE_PORT").unwrap_or(8092),
            parakeet_dir: get("PARAKEET_DIR").unwrap_or_else(|| "/parakeet-models".to_string()),
            parakeet_pool_size: env_num(get, "PARAKEET_POOL_SIZE", 1, 1),
            parakeet_idle_evict_secs: env_num(
                get,
                "PARAKEET_IDLE_EVICT_SECS",
                0,
                PARAKEET_IDLE_EVICT_DEFAULT_SECS,
            ),
            vad_model: get("SILERO_VAD_MODEL")
                .unwrap_or_else(|| "/vad/silero_vad.onnx".to_string()),
            num_threads: parse_env(get, "MOONSHINE_THREADS").unwrap_or(4),
            max_audio_duration_s: parse_env(get, "MAX_AUDIO_DURATION_S").unwrap_or(0.0),
            pool_acquire_timeout_s: env_num(get, "POOL_ACQUIRE_TIMEOUT_S", 0, 30),
            vad_threshold: parse_env(get, "VAD_THRESHOLD").unwrap_or(0.5),
            vad_min_silence_s: parse_env(get, "VAD_MIN_SILENCE_S").unwrap_or(0.5),
            vad_speech_pad_s: parse_env(get, "VAD_SPEECH_PAD_S").unwrap_or(0.05),
            vad_min_speech_s: parse_env(get, "VAD_MIN_SPEECH_S").unwrap_or(0.25),
            vad_max_chunk_s: parse_env(get, "VAD_MAX_CHUNK_S").unwrap_or(20),
            max_chunk_s: env_num(get, "MAX_CHUNK_S", 1, 30),
            hallucination_threshold: parse_env(get, "HALLUCINATION_THRESHOLD").unwrap_or(2.4),
            max_body_size_mb: parse_env(get, "MAX_BODY_SIZE_MB").unwrap_or(50),
            provider: get("ONNX_PROVIDER").unwrap_or_else(|| "cpu".to_string()),
            prom_port: parse_env(get, "OXWHISPER_PROM_PORT").unwrap_or(9092),
            upload_dir: env::temp_dir(),
            ws_max_buffer_s: env_num(get, "WS_MAX_BUFFER_S", 1, 120),
            #[cfg(test)]
            decode_delay: std::time::Duration::ZERO,
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

    /// The old global idle threshold no longer exists. A deployment that still
    /// sets it must not get Parakeet evicted: a reload more than doubles RSS.
    #[test]
    fn the_old_global_eviction_threshold_does_not_reach_parakeet() {
        let cfg = Config::from_lookup(&lookup(&[("OX_WHISPER_IDLE_EVICT_SECS", "600")]));
        assert_eq!(cfg.parakeet_idle_evict_secs, 0);
    }

    #[test]
    fn leftover_settings_are_reported_not_silently_ignored() {
        let got = removed_settings_present(&lookup(&[
            ("PARAKEET_LANGS", "off"),
            ("POOL_SIZE", "2"),
            ("PARAKEET_DIR", "/m"),
        ]));
        assert_eq!(got, vec!["PARAKEET_LANGS", "POOL_SIZE"]);
        assert!(removed_settings_present(&lookup(&[("PARAKEET_DIR", "/m")])).is_empty());
    }

    #[test]
    fn one_parakeet_slot_by_default() {
        // Each slot holds a ~3.2 GB encoder: the default must not be 2.
        let cfg = Config::from_lookup(&lookup(&[("POOL_SIZE", "2")]));
        assert_eq!(cfg.parakeet_pool_size, 1);
        let cfg = Config::from_lookup(&lookup(&[("PARAKEET_POOL_SIZE", "2")]));
        assert_eq!(cfg.parakeet_pool_size, 2);
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

    #[test]
    fn the_decode_window_defaults_to_30_s_and_refuses_zero() {
        assert_eq!(Config::from_lookup(&lookup(&[])).max_chunk_s, 30);
        let cfg = Config::from_lookup(&lookup(&[("MAX_CHUNK_S", "0")]));
        assert_eq!(cfg.max_chunk_samples(), 30 * 16000);
        let cfg = Config::from_lookup(&lookup(&[("MAX_CHUNK_S", "20")]));
        assert_eq!(cfg.max_chunk_samples(), 20 * 16000);
    }

    #[test]
    fn the_retired_vad_duration_gate_is_reported() {
        assert_eq!(
            removed_settings_present(&lookup(&[("VAD_MIN_DURATION_S", "10")])),
            vec!["VAD_MIN_DURATION_S"]
        );
    }
}
