use std::path::Path;
use std::sync::Mutex;

use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use sherpa_rs::nemo_ctc::{NemoCtcConfig, NemoCtcRecognizer};
use sherpa_rs::online_punctuate::{OnlinePunctuation, OnlinePunctuationConfig};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};

use crate::config::Config;
use crate::metrics::names as metric_names;
use crate::pool::EvictablePool;
use crate::recognizer::RuRecognizer;
use crate::routing::Engine;

pub struct Models {
    pub en: Option<std::sync::Arc<EvictablePool<MoonshineRecognizer>>>,
    pub ru: Option<std::sync::Arc<EvictablePool<RuRecognizer>>>,
    /// Model id of the loaded RU model ("none" when absent), read once at load
    /// so `/health` and `/v1/models` never take a pool slot.
    pub ru_model_name: &'static str,
    /// The RU model punctuates itself (GigaAM v3 transducer): the external
    /// punctuation model is skipped for `ru` on every path.
    pub ru_builtin_punct: bool,
    /// Parakeet TDT v3 — serves the languages in `PARAKEET_LANGS`.
    pub parakeet: Option<std::sync::Arc<EvictablePool<RuRecognizer>>>,
    pub vad: Option<Mutex<SileroVad>>,
    pub punct: Option<Mutex<OnlinePunctuation>>,
    pub diarize: Option<crate::diarize::DiarizeEngine>,
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
    pub fn load(config: &Config) -> Self {
        let en = load_moonshine(config);
        let parakeet = if config.parakeet_langs.is_empty() {
            tracing::info!("Parakeet disabled (PARAKEET_LANGS is empty or off)");
            None
        } else {
            load_parakeet(config)
        };
        // Do not keep a second RU model resident when Parakeet already serves
        // Russian. If Parakeet failed to load, RU falls back to its own model.
        let ru = if !needs_ru_model(parakeet.is_some(), &config.parakeet_langs) {
            tracing::info!("RU is served by Parakeet; RU model not loaded");
            None
        } else {
            load_ru(config)
        };
        let vad = load_vad(config);
        let punct = load_punctuation(config);
        let diarize = crate::diarize::DiarizeEngine::load(
            &config.diarize_segmentation_model,
            &config.diarize_embedding_model,
        );

        warmup(&en, "EN");
        warmup(&ru, "RU");
        warmup(&parakeet, "Parakeet");
        let (ru_model_name, ru_builtin_punct) = ru
            .as_ref()
            .and_then(|p| p.try_acquire().ok())
            .map(|r| (r.model_name(), r.has_builtin_punct()))
            .unwrap_or(("none", false));

        let mut eviction_handles = Vec::new();

        // Spawn idle-eviction loop if threshold is configured.
        // ── M1: tick = max(idle_secs / 4, 5s) to keep max latency ≤ 1.25× threshold.
        if config.idle_evict_secs > 0 {
            // ── Open question: warn on aggressive threshold.
            if config.idle_evict_secs < 30 {
                tracing::warn!(
                    idle_evict_secs = config.idle_evict_secs,
                    "aggressive eviction threshold may cause excessive cold-starts (recommended ≥ 30s)"
                );
            }
            let quarter = std::time::Duration::from_secs(config.idle_evict_secs / 4);
            let tick = quarter.max(std::time::Duration::from_secs(5));
            tracing::info!(
                ?tick,
                idle_evict_secs = config.idle_evict_secs,
                "idle eviction enabled"
            );
            if let Some(ref pool) = en {
                eviction_handles.push(pool.spawn_eviction_loop(tick));
            }
            if let Some(ref pool) = ru {
                eviction_handles.push(pool.spawn_eviction_loop(tick));
            }
        }
        // Parakeet has its own threshold (PARAKEET_IDLE_EVICT_SECS): reloading
        // its encoder is slow and can fail where the small models would not.
        if let Some(ref pool) = parakeet
            && config.parakeet_idle_evict_secs > 0
        {
            let quarter = std::time::Duration::from_secs(config.parakeet_idle_evict_secs / 4);
            let tick = quarter.max(std::time::Duration::from_secs(5));
            eviction_handles.push(pool.spawn_eviction_loop(tick));
        }

        Self {
            en,
            ru,
            ru_model_name,
            ru_builtin_punct,
            parakeet,
            vad,
            punct,
            diarize,
            eviction_handles,
        }
    }

    /// Empty `Models` for tests that exercise HTTP handlers without loading
    /// real model files.
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            en: None,
            ru: None,
            ru_model_name: "none",
            ru_builtin_punct: false,
            parakeet: None,
            vad: None,
            punct: None,
            diarize: None,
            eviction_handles: Vec::new(),
        }
    }

    /// The engine that serves `language` with the models actually loaded.
    pub fn route(&self, language: &str, parakeet_langs: &[String]) -> Engine {
        Engine::route(language, self.parakeet.is_some(), parakeet_langs)
    }

    /// The batch-decoded pool behind `engine`; `None` for Moonshine, whose
    /// pool has a different type (`self.en`).
    pub fn offline_pool(
        &self,
        engine: Engine,
    ) -> Option<&std::sync::Arc<EvictablePool<RuRecognizer>>> {
        match engine {
            Engine::Parakeet => self.parakeet.as_ref(),
            Engine::Ru => self.ru.as_ref(),
            Engine::Moonshine => None,
        }
    }
}

impl Models {
    /// The engine a language falls back to when Parakeet cannot reload: the
    /// route without Parakeet, if that model is loaded.
    pub fn reload_fallback(&self, language: &str, parakeet_langs: &[String]) -> Option<Engine> {
        reload_fallback(
            language,
            parakeet_langs,
            self.ru.is_some(),
            self.en.is_some(),
        )
    }
}

fn reload_fallback(
    language: &str,
    parakeet_langs: &[String],
    ru_loaded: bool,
    en_loaded: bool,
) -> Option<Engine> {
    match Engine::route(language, false, parakeet_langs) {
        Engine::Ru if ru_loaded => Some(Engine::Ru),
        Engine::Moonshine if en_loaded => Some(Engine::Moonshine),
        _ => None,
    }
}

/// Whether the RU model has to be loaded: always, unless Parakeet is loaded
/// and routes `ru` itself.
fn needs_ru_model(parakeet_loaded: bool, parakeet_langs: &[String]) -> bool {
    Engine::route("ru", parakeet_loaded, parakeet_langs) != Engine::Parakeet
}

fn load_moonshine(config: &Config) -> Option<std::sync::Arc<EvictablePool<MoonshineRecognizer>>> {
    let merged_path = format!("{}/decoder_model_merged.ort", config.models_dir);
    let preprocess_path = format!("{}/preprocess.onnx", config.models_dir);

    let moonshine_cfg = if Path::new(&merged_path).exists() {
        tracing::info!("Detected Moonshine v2 model format");
        MoonshineConfig {
            encoder: format!("{}/encoder_model.ort", config.models_dir),
            merged_decoder: merged_path,
            tokens: format!("{}/tokens.txt", config.models_dir),
            num_threads: Some(config.num_threads),
            provider: Some(config.provider.clone()),
            ..Default::default()
        }
    } else if Path::new(&preprocess_path).exists() {
        tracing::info!("Detected Moonshine v1 model format");
        MoonshineConfig {
            preprocessor: preprocess_path,
            encoder: format!("{}/encode.int8.onnx", config.models_dir),
            uncached_decoder: format!("{}/uncached_decode.int8.onnx", config.models_dir),
            cached_decoder: format!("{}/cached_decode.int8.onnx", config.models_dir),
            tokens: format!("{}/tokens.txt", config.models_dir),
            num_threads: Some(config.num_threads),
            provider: Some(config.provider.clone()),
            ..Default::default()
        }
    } else {
        tracing::warn!(
            "EN model not found at {} (no v2 merged decoder or v1 preprocessor), skipping",
            config.models_dir
        );
        return None;
    };

    // Pre-fill pool: try creating pool_size recognizers eagerly.
    let mut recognizers = Vec::new();
    for i in 0..config.pool_size {
        match MoonshineRecognizer::new(moonshine_cfg.clone()) {
            Ok(r) => {
                tracing::info!("EN recognizer {}/{} loaded", i + 1, config.pool_size);
                recognizers.push(r);
            }
            Err(e) => {
                tracing::error!("EN recognizer {}/{} failed: {}", i + 1, config.pool_size, e);
                break;
            }
        }
    }

    if recognizers.is_empty() {
        return None;
    }

    let size = recognizers.len();
    // Factory for lazy reinit after eviction.
    let cfg_for_factory = moonshine_cfg.clone();
    let factory: std::sync::Arc<
        dyn Fn() -> Result<MoonshineRecognizer, anyhow::Error> + Send + Sync,
    > = std::sync::Arc::new(move || {
        MoonshineRecognizer::new(cfg_for_factory.clone())
            .map_err(|e| anyhow::anyhow!("MoonshineRecognizer reinit failed: {e}"))
    });
    let pool = EvictablePool::from_items(recognizers, config.idle_evict_secs, factory)
        .with_acquire_timeout(std::time::Duration::from_secs(
            config.pool_acquire_timeout_s,
        ));
    metrics::gauge!(metric_names::POOL_SIZE, "lang" => "en").set(size as f64);
    Some(std::sync::Arc::new(pool))
}

/// Loads Parakeet TDT 0.6B v3 from a sherpa-onnx export: `encoder`, `decoder`
/// and `joiner` as `.int8.onnx` (preferred) or `.onnx` (fp32; the encoder's
/// weights sit next to it in `encoder.weights`), plus `tokens.txt`.
/// The model type is fixed here — never inferred by reading the multi-GB
/// encoder into memory, which `detect_transducer_type` would do.
fn load_parakeet(config: &Config) -> Option<std::sync::Arc<EvictablePool<RuRecognizer>>> {
    let dir = &config.parakeet_dir;
    let encoder = find_model_file(dir, "encoder");
    if !Path::new(&encoder).exists() {
        tracing::warn!(
            "Parakeet model not found at {dir}; {:?} fall back to the RU/Moonshine models",
            config.parakeet_langs
        );
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
    let variant = if encoder.ends_with(".int8.onnx") {
        "int8"
    } else {
        "fp32/fp16"
    };
    tracing::info!("Parakeet encoder {encoder} ({variant})");
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
                recognizers.push(RuRecognizer::Parakeet(r));
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
    let factory: std::sync::Arc<dyn Fn() -> Result<RuRecognizer, anyhow::Error> + Send + Sync> =
        std::sync::Arc::new(move || {
            TransducerRecognizer::new(cfg.clone())
                .map(RuRecognizer::Parakeet)
                .map_err(|e| anyhow::anyhow!("Parakeet reinit failed: {e}"))
        });
    let pool = EvictablePool::from_items(recognizers, config.parakeet_idle_evict_secs, factory)
        .with_acquire_timeout(std::time::Duration::from_secs(
            config.pool_acquire_timeout_s,
        ));
    metrics::gauge!(metric_names::POOL_SIZE, "lang" => Engine::Parakeet.label()).set(size as f64);
    Some(std::sync::Arc::new(pool))
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

fn load_ru(config: &Config) -> Option<std::sync::Arc<EvictablePool<RuRecognizer>>> {
    // Try NeMo CTC (GigaAM) first — faster, better WER
    let nemo_model = format!("{}/model.int8.onnx", config.ru_models_dir);
    let nemo_tokens = format!("{}/tokens.txt", config.ru_models_dir);
    if Path::new(&nemo_model).exists()
        && !Path::new(&format!("{}/encoder.int8.onnx", config.ru_models_dir)).exists()
    {
        return load_nemo_ctc(config, &nemo_model, &nemo_tokens);
    }

    // Fall back to Zipformer Transducer
    let encoder_path = format!("{}/encoder.int8.onnx", config.ru_models_dir);
    if !Path::new(&encoder_path).exists() {
        tracing::warn!("RU model not found at {}, skipping", config.ru_models_dir);
        return None;
    }
    load_zipformer(config, &encoder_path)
}

fn load_nemo_ctc(
    config: &Config,
    model: &str,
    tokens: &str,
) -> Option<std::sync::Arc<EvictablePool<RuRecognizer>>> {
    let nemo_cfg = NemoCtcConfig {
        model: model.to_string(),
        tokens: tokens.to_string(),
        num_threads: Some(config.num_threads),
        provider: Some(config.provider.clone()),
        ..Default::default()
    };
    let mut recognizers = Vec::new();
    for i in 0..config.pool_size {
        match NemoCtcRecognizer::new(nemo_cfg.clone()) {
            Ok(r) => {
                tracing::info!("RU NeMo CTC (GigaAM) {}/{} loaded", i + 1, config.pool_size);
                recognizers.push(RuRecognizer::NemoCtc(r));
            }
            Err(e) => {
                tracing::error!("RU NeMo CTC {}/{} failed: {}", i + 1, config.pool_size, e);
                break;
            }
        }
    }
    if recognizers.is_empty() {
        return None;
    }
    let size = recognizers.len();
    let cfg_for_factory = nemo_cfg.clone();
    let factory: std::sync::Arc<dyn Fn() -> Result<RuRecognizer, anyhow::Error> + Send + Sync> =
        std::sync::Arc::new(move || {
            NemoCtcRecognizer::new(cfg_for_factory.clone())
                .map(RuRecognizer::NemoCtc)
                .map_err(|e| anyhow::anyhow!("NemoCtcRecognizer reinit failed: {e}"))
        });
    let pool = EvictablePool::from_items(recognizers, config.idle_evict_secs, factory)
        .with_acquire_timeout(std::time::Duration::from_secs(
            config.pool_acquire_timeout_s,
        ));
    metrics::gauge!(metric_names::POOL_SIZE, "lang" => "ru").set(size as f64);
    Some(std::sync::Arc::new(pool))
}

fn load_zipformer(
    config: &Config,
    encoder_path: &str,
) -> Option<std::sync::Arc<EvictablePool<RuRecognizer>>> {
    let model_type = detect_transducer_type(encoder_path);
    let is_nemo = model_type == "nemo_transducer";
    let decoder_path = find_model_file(&config.ru_models_dir, "decoder");
    let joiner_path = find_model_file(&config.ru_models_dir, "joiner");
    let transducer_cfg = TransducerConfig {
        encoder: encoder_path.to_string(),
        decoder: decoder_path,
        joiner: joiner_path,
        tokens: format!("{}/tokens.txt", config.ru_models_dir),
        num_threads: config.num_threads,
        sample_rate: 16000,
        feature_dim: 80,
        decoding_method: "greedy_search".to_string(),
        model_type,
        provider: Some(config.provider.clone()),
        ..Default::default()
    };
    let mut recognizers = Vec::new();
    for i in 0..config.pool_size {
        match TransducerRecognizer::new(transducer_cfg.clone()) {
            Ok(r) => {
                let variant = if is_nemo {
                    "NeMo GigaAM v3"
                } else {
                    "Zipformer"
                };
                tracing::info!(
                    "RU {} ({}) {}/{} loaded",
                    variant,
                    transducer_cfg.model_type,
                    i + 1,
                    config.pool_size
                );
                let rec = if is_nemo {
                    RuRecognizer::NemoTransducer(r)
                } else {
                    RuRecognizer::Transducer(r)
                };
                recognizers.push(rec);
            }
            Err(e) => {
                tracing::error!("RU Transducer {}/{} failed: {}", i + 1, config.pool_size, e);
                break;
            }
        }
    }
    if recognizers.is_empty() {
        return None;
    }
    let size = recognizers.len();
    let cfg_for_factory = transducer_cfg.clone();
    let factory: std::sync::Arc<dyn Fn() -> Result<RuRecognizer, anyhow::Error> + Send + Sync> =
        std::sync::Arc::new(move || {
            TransducerRecognizer::new(cfg_for_factory.clone())
                .map(|r| {
                    if is_nemo {
                        RuRecognizer::NemoTransducer(r)
                    } else {
                        RuRecognizer::Transducer(r)
                    }
                })
                .map_err(|e| anyhow::anyhow!("TransducerRecognizer reinit failed: {e}"))
        });
    let pool = EvictablePool::from_items(recognizers, config.idle_evict_secs, factory)
        .with_acquire_timeout(std::time::Duration::from_secs(
            config.pool_acquire_timeout_s,
        ));
    metrics::gauge!(metric_names::POOL_SIZE, "lang" => "ru").set(size as f64);
    Some(std::sync::Arc::new(pool))
}

/// Detect transducer model type from encoder ONNX metadata.
/// Returns "nemo_transducer" for GigaAM/NeMo models, "transducer" otherwise.
fn detect_transducer_type(encoder_path: &str) -> String {
    // Check for NeMo metadata markers via file content scan.
    // NeMo transducer encoders contain "EncDecRNNTBPEModel" or "is_giga_am" in metadata.
    if let Ok(data) = std::fs::read(encoder_path) {
        let haystack = String::from_utf8_lossy(&data);
        if haystack.contains("EncDecRNNTBPEModel") || haystack.contains("is_giga_am") {
            tracing::info!("Detected NeMo transducer model (GigaAM)");
            return "nemo_transducer".to_string();
        }
    }
    "transducer".to_string()
}

/// Find decoder/joiner model file, preferring .int8.onnx over .onnx.
fn find_model_file(dir: &str, name: &str) -> String {
    let int8 = format!("{}/{}.int8.onnx", dir, name);
    if Path::new(&int8).exists() {
        return int8;
    }
    format!("{}/{}.onnx", dir, name)
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

fn load_punctuation(config: &Config) -> Option<Mutex<OnlinePunctuation>> {
    if !Path::new(&config.punct_model).exists() {
        tracing::warn!(
            "Punctuation model not found at {}, skipping",
            config.punct_model
        );
        return None;
    }
    let cfg = OnlinePunctuationConfig {
        cnn_bilstm: config.punct_model.clone(),
        bpe_vocab: config.punct_vocab.clone(),
        ..Default::default()
    };
    match OnlinePunctuation::new(cfg) {
        Ok(p) => {
            tracing::info!("Punctuation loaded from {}", config.punct_model);
            Some(Mutex::new(p))
        }
        Err(e) => {
            tracing::error!("Punctuation load failed: {}", e);
            None
        }
    }
}

fn warmup<T: Warmable + Send + 'static>(
    pool: &Option<std::sync::Arc<EvictablePool<T>>>,
    label: &str,
) {
    if let Some(p) = pool
        && let Ok(mut r) = p.try_acquire()
    {
        r.warmup();
        tracing::info!("{} warmup complete", label);
    }
}

trait Warmable {
    fn warmup(&mut self);
}
impl Warmable for MoonshineRecognizer {
    fn warmup(&mut self) {
        let _ = self.transcribe(16000, &[0.0f32; 16000]);
    }
}
impl Warmable for RuRecognizer {
    fn warmup(&mut self) {
        let _ = self.transcribe(16000, &[0.0f32; 16000]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_parakeet_langs;

    #[test]
    fn ru_model_is_skipped_only_when_parakeet_is_loaded_and_routes_ru() {
        let all = parse_parakeet_langs(None);
        assert!(!needs_ru_model(true, &all));
        // Parakeet missing → the RU model must still load.
        assert!(needs_ru_model(false, &all));
        // Parakeet loaded but `ru` not routed to it.
        assert!(needs_ru_model(true, &parse_parakeet_langs(Some("en,uk"))));
        // Rollback switch.
        assert!(needs_ru_model(false, &parse_parakeet_langs(Some("off"))));
    }

    #[test]
    fn missing_parakeet_dir_loads_nothing_and_routes_to_old_models() {
        let missing = std::env::temp_dir()
            .join(format!("oxw-no-models-{}", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let mut config = Config::from_env();
        config.parakeet_dir = missing.clone();
        config.parakeet_langs = parse_parakeet_langs(None);
        config.models_dir = missing.clone();
        config.ru_models_dir = missing.clone();
        config.vad_model = format!("{missing}/vad.onnx");
        config.punct_model = format!("{missing}/punct.onnx");
        config.idle_evict_secs = 0;
        let models = Models::load(&config);
        assert!(models.parakeet.is_none());
        assert_eq!(models.route("ru", &config.parakeet_langs), Engine::Ru);
        assert_eq!(
            models.route("en", &config.parakeet_langs),
            Engine::Moonshine
        );
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

    #[test]
    fn parakeet_reload_failure_falls_back_to_the_model_without_parakeet() {
        let all = parse_parakeet_langs(None);
        // en → Moonshine when it is loaded.
        assert_eq!(
            reload_fallback("en", &all, false, true),
            Some(Engine::Moonshine)
        );
        // ru → the RU model only when one is loaded; otherwise no fallback.
        assert_eq!(reload_fallback("ru", &all, true, true), Some(Engine::Ru));
        assert_eq!(reload_fallback("ru", &all, false, true), None);
        // Nothing loaded → no fallback.
        assert_eq!(reload_fallback("en", &all, false, false), None);
    }
}
