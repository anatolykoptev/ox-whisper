use std::path::Path;
use std::sync::{Arc, Mutex};

use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};

use crate::config::Config;
use crate::metrics::names as metric_names;
use crate::pool::EvictablePool;
use crate::transcribe::TranscribeError;

/// Model id reported by `/health` and `/v1/models`.
pub const PARAKEET_MODEL_NAME: &str = "parakeet-tdt-0.6b-v3";

/// Pool label in metrics.
pub const PARAKEET_POOL_LABEL: &str = "parakeet";

pub struct Models {
    /// Parakeet TDT v3, the only recognizer. `None` only in tests:
    /// [`Models::load`] refuses to start without it.
    pub parakeet: Option<Arc<EvictablePool<TransducerRecognizer>>>,
    pub vad: Option<Mutex<SileroVad>>,
    /// Eviction loop handles — aborted on drop to stop background tasks.
    eviction_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl Drop for Models {
    fn drop(&mut self) {
        for handle in &self.eviction_handles {
            handle.abort();
        }
    }
}

impl Models {
    /// Loads Parakeet and the VAD. A missing or broken Parakeet is an error:
    /// with no other model to fall back to, a service that started anyway would
    /// answer `/health` while failing every transcription.
    pub fn load(config: &Config) -> Result<Self, String> {
        let parakeet = load_parakeet(config).ok_or_else(|| {
            format!(
                "Parakeet model not loaded from {} (see the log above); \
                 ox-whisper has no other model to serve with",
                config.parakeet_dir
            )
        })?;
        let vad = load_vad(config);
        warmup(&parakeet);

        // Parakeet is kept resident by default: reloading its encoder is slow
        // and more than doubles RSS (PARAKEET_IDLE_EVICT_SECS opts in).
        let mut eviction_handles = Vec::new();
        if config.parakeet_idle_evict_secs > 0 {
            let quarter = std::time::Duration::from_secs(config.parakeet_idle_evict_secs / 4);
            let tick = quarter.max(std::time::Duration::from_secs(5));
            tracing::info!(
                ?tick,
                idle_evict_secs = config.parakeet_idle_evict_secs,
                "Parakeet idle eviction enabled"
            );
            eviction_handles.push(parakeet.spawn_eviction_loop(tick));
        }

        Ok(Self {
            parakeet: Some(parakeet),
            vad,
            eviction_handles,
        })
    }

    /// Empty `Models` for tests that exercise HTTP handlers without loading
    /// real model files.
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            parakeet: None,
            vad: None,
            eviction_handles: Vec::new(),
        }
    }

    /// The Parakeet pool, or the error a request gets when there is none.
    pub fn parakeet_pool(
        &self,
    ) -> Result<&Arc<EvictablePool<TransducerRecognizer>>, TranscribeError> {
        self.parakeet.as_ref().ok_or(TranscribeError::NoRecognizer)
    }

    /// Whether Parakeet can serve right now: loaded, and not stuck on a
    /// failed reload. Reads state only — taking a pool slot here would make
    /// every healthcheck compete with requests.
    pub fn parakeet_ready(&self) -> bool {
        self.parakeet.as_ref().is_some_and(|p| p.is_healthy())
    }
}

/// Loads Parakeet TDT 0.6B v3 from a sherpa-onnx export: `encoder`, `decoder`
/// and `joiner` as `.onnx` (fp32 or fp16; the encoder's weights sit next to it
/// in `encoder.weights`) or, failing that, `.int8.onnx`, plus `tokens.txt`.
/// The model type is fixed here — never inferred by reading the multi-GB
/// encoder into memory.
fn load_parakeet(config: &Config) -> Option<Arc<EvictablePool<TransducerRecognizer>>> {
    let dir = &config.parakeet_dir;
    let encoder = find_model_file(dir, "encoder");
    if !Path::new(&encoder).exists() {
        tracing::error!("Parakeet encoder not found at {encoder}");
        return None;
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let unresolved = unresolved_external_data(Path::new(&encoder), &cwd);
    if !unresolved.is_empty() {
        tracing::error!(
            "Parakeet: {encoder} keeps its weights in {unresolved:?}, which onnxruntime \
             resolves from the working directory ({}); run the container with \
             working_dir {dir}. Parakeet not loaded",
            cwd.display()
        );
        return None;
    }
    if encoder.ends_with(".int8.onnx") {
        tracing::warn!(
            "Parakeet encoder {encoder} is the int8 export, about 4 WER points worse than \
             fp32 on the same clips"
        );
    } else {
        tracing::info!("Parakeet encoder {encoder}");
    }
    let cfg = TransducerConfig {
        encoder,
        decoder: find_model_file(dir, "decoder"),
        joiner: find_model_file(dir, "joiner"),
        tokens: format!("{dir}/tokens.txt"),
        num_threads: config.num_threads,
        sample_rate: 16000,
        // Parakeet uses 128 mel bins; sherpa-onnx also reads `feat_dim` from
        // the encoder metadata and overrides this value.
        feature_dim: 128,
        decoding_method: "greedy_search".to_string(),
        model_type: "nemo_transducer".to_string(),
        provider: Some(config.provider.clone()),
        ..Default::default()
    };
    let mut recognizers = Vec::new();
    for i in 0..config.parakeet_pool_size {
        match TransducerRecognizer::new(cfg.clone()) {
            Ok(r) => {
                tracing::info!(
                    "Parakeet recognizer {}/{} loaded",
                    i + 1,
                    config.parakeet_pool_size
                );
                recognizers.push(r);
            }
            Err(e) => {
                tracing::error!(
                    "Parakeet recognizer {}/{} failed: {}",
                    i + 1,
                    config.parakeet_pool_size,
                    e
                );
                break;
            }
        }
    }
    if recognizers.is_empty() {
        return None;
    }
    let size = recognizers.len();
    let factory: Arc<dyn Fn() -> Result<TransducerRecognizer, anyhow::Error> + Send + Sync> =
        Arc::new(move || {
            TransducerRecognizer::new(cfg.clone())
                .map_err(|e| anyhow::anyhow!("Parakeet reinit failed: {e}"))
        });
    let pool = EvictablePool::from_items(recognizers, config.parakeet_idle_evict_secs, factory)
        .with_acquire_timeout(std::time::Duration::from_secs(
            config.pool_acquire_timeout_s,
        ));
    metrics::gauge!(metric_names::POOL_SIZE, "lang" => PARAKEET_POOL_LABEL).set(size as f64);
    Some(Arc::new(pool))
}

/// Files beside `encoder` that it references as external data and that
/// onnxruntime would not find. sherpa-onnx 1.12.28 hands onnxruntime the model
/// as a byte buffer, so external data (the fp32 export's `encoder.weights`, or
/// `*.onnx.data`) is looked up relative to the process working directory, not
/// the model directory — and a miss throws through the C API and aborts the
/// process. An encoder references a file by name, so any sibling whose name
/// occurs in the encoder's bytes counts. Encoders over 512 MiB are
/// self-contained (int8): their weights are inline.
fn unresolved_external_data(encoder: &Path, cwd: &Path) -> Vec<String> {
    const MAX_GRAPH_BYTES: u64 = 512 << 20;
    let Some(dir) = encoder.parent() else {
        return Vec::new();
    };
    match std::fs::metadata(encoder) {
        Ok(m) if m.len() <= MAX_GRAPH_BYTES => {}
        _ => return Vec::new(),
    }
    let Ok(graph) = std::fs::read(encoder) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut unresolved = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path == encoder || !path.is_file() || !contains(&graph, name.as_bytes()) {
            continue;
        }
        let seen = cwd.join(name);
        let same = match (seen.canonicalize(), path.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if !same {
            unresolved.push(name.to_string());
        }
    }
    unresolved.sort();
    unresolved
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Finds a decoder/joiner/encoder file: the `.onnx` (fp32/fp16) export, then
/// the `.int8.onnx` one. The int8 export failed the accuracy gate, so it is
/// used only when nothing better is there — never because it sorts first.
fn find_model_file(dir: &str, name: &str) -> String {
    let full = format!("{}/{}.onnx", dir, name);
    if Path::new(&full).exists() {
        return full;
    }
    let int8 = format!("{}/{}.int8.onnx", dir, name);
    if Path::new(&int8).exists() {
        return int8;
    }
    full
}

pub(crate) fn load_vad(config: &Config) -> Option<Mutex<SileroVad>> {
    if !Path::new(&config.vad_model).exists() {
        tracing::warn!("VAD model not found at {}, skipping", config.vad_model);
        return None;
    }
    let cfg = SileroVadConfig {
        model: config.vad_model.clone(),
        threshold: config.vad_threshold,
        min_silence_duration: config.vad_min_silence_s,
        min_speech_duration: config.vad_min_speech_s,
        window_size: 512,
        ..Default::default()
    };
    let vad_max = if config.max_audio_duration_s > 0.0 {
        config.max_audio_duration_s
    } else {
        3600.0
    };
    match SileroVad::new(cfg, vad_max as f32) {
        Ok(v) => {
            tracing::info!("VAD loaded from {}", config.vad_model);
            Some(Mutex::new(v))
        }
        Err(e) => {
            tracing::error!("VAD load failed: {}", e);
            None
        }
    }
}

fn warmup(pool: &Arc<EvictablePool<TransducerRecognizer>>) {
    if let Ok(mut r) = pool.try_acquire() {
        let _ = r.transcribe(16000, &[0.0f32; 16000]);
        tracing::info!("Parakeet warmup complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With one model and no fallback, starting without it must fail loudly:
    /// a running service whose every request errors is worse than a crash.
    #[test]
    fn starting_without_the_parakeet_model_is_an_error() {
        let missing = std::env::temp_dir()
            .join(format!("oxw-no-models-{}", uuid::Uuid::new_v4()))
            .to_string_lossy()
            .into_owned();
        let mut config = Config::from_lookup(&|_| None);
        config.parakeet_dir = missing.clone();
        config.vad_model = format!("{missing}/vad.onnx");
        let err = match Models::load(&config) {
            Err(e) => e,
            Ok(_) => panic!("startup must fail without Parakeet"),
        };
        assert!(err.contains(&missing), "{err}");
    }

    #[test]
    fn the_full_precision_export_wins_over_int8() {
        let dir = scratch("pick-model");
        let d = dir.to_string_lossy().into_owned();
        // Only int8 present: used, but only because nothing better exists.
        std::fs::write(dir.join("encoder.int8.onnx"), b"x").unwrap();
        assert_eq!(
            find_model_file(&d, "encoder"),
            format!("{d}/encoder.int8.onnx")
        );
        // Both present: the full-precision file, whatever the sort order.
        std::fs::write(dir.join("encoder.onnx"), b"x").unwrap();
        assert_eq!(find_model_file(&d, "encoder"), format!("{d}/encoder.onnx"));
        // Neither: the full-precision path, so the error names the right file.
        assert_eq!(find_model_file(&d, "decoder"), format!("{d}/decoder.onnx"));
        for f in ["encoder.onnx", "encoder.int8.onnx"] {
            std::fs::remove_file(dir.join(f)).unwrap();
        }
        std::fs::remove_dir(&dir).unwrap();
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("oxw-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// An fp32 encoder referencing `encoder.weights`: unresolvable from an
    /// unrelated working directory, resolvable from the model directory.
    #[test]
    fn external_weights_must_resolve_from_the_working_directory() {
        let model = scratch("p32-model");
        let elsewhere = scratch("p32-cwd");
        let encoder = model.join("encoder.onnx");
        std::fs::write(&encoder, b"\x08\x01location\x12\x0fencoder.weights").unwrap();
        std::fs::write(model.join("encoder.weights"), b"w").unwrap();
        std::fs::write(model.join("tokens.txt"), b"t").unwrap();

        assert_eq!(
            unresolved_external_data(&encoder, &elsewhere),
            vec!["encoder.weights".to_string()]
        );
        // A different file of the same name in the working directory is not it.
        std::fs::write(elsewhere.join("encoder.weights"), b"other").unwrap();
        assert_eq!(unresolved_external_data(&encoder, &elsewhere).len(), 1);
        // Positive case: the container runs with working_dir = model dir.
        assert!(unresolved_external_data(&encoder, &model).is_empty());

        for f in ["encoder.onnx", "encoder.weights", "tokens.txt"] {
            std::fs::remove_file(model.join(f)).unwrap();
        }
        std::fs::remove_file(elsewhere.join("encoder.weights")).unwrap();
        std::fs::remove_dir(&model).unwrap();
        std::fs::remove_dir(&elsewhere).unwrap();
    }

    /// Any external-data name counts (`*.onnx.data`), and a self-contained
    /// encoder that references nothing passes.
    #[test]
    fn onnx_data_sidecars_count_and_self_contained_encoders_pass() {
        let model = scratch("p16-model");
        let elsewhere = scratch("p16-cwd");
        let encoder = model.join("encoder.onnx");
        std::fs::write(&encoder, b"location encoder.onnx.data").unwrap();
        std::fs::write(model.join("encoder.onnx.data"), b"w").unwrap();
        assert_eq!(
            unresolved_external_data(&encoder, &elsewhere),
            vec!["encoder.onnx.data".to_string()]
        );
        let int8 = model.join("encoder.int8.onnx");
        std::fs::write(&int8, b"self-contained graph").unwrap();
        assert!(unresolved_external_data(&int8, &elsewhere).is_empty());

        for f in ["encoder.onnx", "encoder.onnx.data", "encoder.int8.onnx"] {
            std::fs::remove_file(model.join(f)).unwrap();
        }
        std::fs::remove_dir(&model).unwrap();
        std::fs::remove_dir(&elsewhere).unwrap();
    }
}
