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

/// `/health` precision of the `*.onnx` set (fp32 in the shipped image).
pub const PRECISION_FULL: &str = "full";
/// `/health` precision of the `*.int8.onnx` set.
pub const PRECISION_INT8: &str = "int8";

/// Pool label in metrics.
pub const PARAKEET_POOL_LABEL: &str = "parakeet";

pub struct Models {
    /// Parakeet TDT v3, the only recognizer. `None` only in tests:
    /// [`Models::load`] refuses to start without it.
    pub parakeet: Option<Arc<EvictablePool<TransducerRecognizer>>>,
    pub vad: Option<Mutex<SileroVad>>,
    /// Which export is loaded ([`PRECISION_FULL`] or [`PRECISION_INT8`]), for
    /// `/health`. `None` only in tests.
    pub parakeet_precision: Option<&'static str>,
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
        let (parakeet, precision) = load_parakeet(config).ok_or_else(|| {
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
            parakeet_precision: Some(precision),
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
            parakeet_precision: None,
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
fn load_parakeet(
    config: &Config,
) -> Option<(Arc<EvictablePool<TransducerRecognizer>>, &'static str)> {
    let dir = &config.parakeet_dir;
    let files = match pick_model_files(dir) {
        Ok(f) => f,
        Err(e) => {
            tracing::error!("{e}");
            return None;
        }
    };
    let precision = files.precision();
    let encoder = files.encoder.clone();
    let cwd = std::env::current_dir().unwrap_or_default();
    let unresolved = match unresolved_external_data(Path::new(&encoder), &cwd) {
        Ok(u) => u,
        Err(e) => {
            // Not knowing is not a pass: a missed reference aborts the process
            // inside onnxruntime later, with no message.
            tracing::error!(
                "Parakeet: cannot check {encoder} for external weights: {e}. Parakeet not loaded"
            );
            return None;
        }
    };
    if !unresolved.is_empty() {
        tracing::error!(
            "Parakeet: {encoder} keeps its weights in {unresolved:?}, which onnxruntime \
             resolves from the working directory ({}); run the container with \
             working_dir {dir}. Parakeet not loaded",
            cwd.display()
        );
        return None;
    }
    if files.int8 {
        tracing::warn!(
            "Parakeet: using the int8 set ({encoder}), about 4 WER points worse than fp32 on \
             the same clips"
        );
    } else {
        tracing::info!("Parakeet: using the full-precision set ({encoder})");
    }
    let cfg = TransducerConfig {
        encoder,
        decoder: files.decoder,
        joiner: files.joiner,
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
    Some((Arc::new(pool), precision))
}

/// Files beside `encoder` that it references as external data and that
/// onnxruntime would not find. sherpa-onnx 1.12.28 hands onnxruntime the model
/// as a byte buffer, so external data (the fp32 export's `encoder.weights`, or
/// `*.onnx.data`) is looked up relative to the process working directory, not
/// the model directory — and a miss throws through the C API and aborts the
/// process. An encoder references a file by name, so any sibling whose name
/// occurs in the encoder's bytes counts. The encoder is scanned as a stream,
/// whatever its size (the int8 graph is over 2 GB), and any I/O error is an
/// error: a check that could not run must not read as "nothing missing".
fn unresolved_external_data(encoder: &Path, cwd: &Path) -> Result<Vec<String>, String> {
    unresolved_external_data_in(encoder, cwd, SCAN_BLOCK)
}

/// Bytes read per step when scanning an encoder for sibling file names.
const SCAN_BLOCK: usize = 8 << 20;

fn unresolved_external_data_in(
    encoder: &Path,
    cwd: &Path,
    block: usize,
) -> Result<Vec<String>, String> {
    let Some(dir) = encoder.parent() else {
        return Ok(Vec::new());
    };
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    let mut siblings = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| format!("reading {}: {e}", dir.display()))?
            .path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path != encoder && path.is_file() && !name.is_empty() {
            siblings.push((name.to_string(), path));
        }
    }
    let names: Vec<&str> = siblings.iter().map(|(n, _)| n.as_str()).collect();
    let referenced = referenced_names(encoder, &names, block)
        .map_err(|e| format!("reading {}: {e}", encoder.display()))?;
    let mut unresolved = Vec::new();
    for ((name, path), referenced) in siblings.iter().zip(referenced) {
        if !referenced {
            continue;
        }
        let same = match (cwd.join(name).canonicalize(), path.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if !same {
            unresolved.push(name.clone());
        }
    }
    unresolved.sort();
    Ok(unresolved)
}

/// For each of `names`, whether its bytes occur in the file. Reads `block`
/// bytes at a time and carries the longest name's length minus one over to the
/// next block, so a name that straddles a block boundary is still found.
fn referenced_names(path: &Path, names: &[&str], block: usize) -> std::io::Result<Vec<bool>> {
    use memchr::memmem::Finder;
    use std::io::Read;
    let finders: Vec<Finder> = names.iter().map(|n| Finder::new(n.as_bytes())).collect();
    let mut found = vec![false; names.len()];
    if names.is_empty() {
        return Ok(found);
    }
    let keep = names.iter().map(|n| n.len()).max().unwrap_or(1) - 1;
    let mut file = std::fs::File::open(path)?;
    let mut buf = vec![0u8; keep + block.max(1)];
    let mut carried = 0;
    loop {
        let n = file.read(&mut buf[carried..])?;
        if n == 0 {
            return Ok(found);
        }
        let end = carried + n;
        for (finder, hit) in finders.iter().zip(found.iter_mut()) {
            if !*hit && finder.find(&buf[..end]).is_some() {
                *hit = true;
            }
        }
        carried = keep.min(end);
        buf.copy_within(end - carried..end, 0);
    }
}

/// The encoder, decoder and joiner of one export, all of one precision.
#[derive(Debug, PartialEq, Eq)]
struct ModelFiles {
    encoder: String,
    decoder: String,
    joiner: String,
    int8: bool,
}

impl ModelFiles {
    /// The label `/health` reports for this set.
    fn precision(&self) -> &'static str {
        if self.int8 {
            PRECISION_INT8
        } else {
            PRECISION_FULL
        }
    }
}

/// Chooses the model set in `dir`: the full-precision (`*.onnx`, fp32/fp16)
/// set when its encoder, decoder and joiner all exist, otherwise the
/// `*.int8.onnx` set, which failed the accuracy gate and is never picked
/// because it sorts first. Mixing precisions (an fp32 encoder with an int8
/// decoder) is not a configuration anyone measured, so a half-present set is
/// an error that names what is missing.
fn pick_model_files(dir: &str) -> Result<ModelFiles, String> {
    let set = |suffix: &str| {
        let f = |name: &str| format!("{dir}/{name}{suffix}");
        (f("encoder"), f("decoder"), f("joiner"))
    };
    let complete = |(e, d, j): &(String, String, String)| {
        [e, d, j].iter().all(|p| Path::new(p.as_str()).exists())
    };
    let full = set(".onnx");
    if complete(&full) {
        return Ok(ModelFiles {
            encoder: full.0,
            decoder: full.1,
            joiner: full.2,
            int8: false,
        });
    }
    let int8 = set(".int8.onnx");
    if complete(&int8) {
        return Ok(ModelFiles {
            encoder: int8.0,
            decoder: int8.1,
            joiner: int8.2,
            int8: true,
        });
    }
    let missing: Vec<&str> = [&full.0, &full.1, &full.2]
        .into_iter()
        .filter(|p| !Path::new(p.as_str()).exists())
        .map(|p| p.as_str())
        .collect();
    Err(format!(
        "no complete Parakeet model set in {dir}: missing {missing:?} (and no complete *.int8.onnx set)"
    ))
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

    /// One precision for the whole set, chosen by completeness, never by
    /// which file happens to sort first.
    #[test]
    fn one_precision_is_chosen_for_the_whole_set() {
        let dir = scratch("pick-model");
        let d = dir.to_string_lossy().into_owned();
        let touch = |names: &[&str]| {
            for n in names {
                std::fs::write(dir.join(n), b"x").unwrap();
            }
        };
        let rm = |names: &[&str]| {
            for n in names {
                std::fs::remove_file(dir.join(n)).unwrap();
            }
        };
        let full = ["encoder.onnx", "decoder.onnx", "joiner.onnx"];
        let int8 = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"];

        // Nothing: an error naming the full-precision files.
        let err = pick_model_files(&d).unwrap_err();
        assert!(err.contains("encoder.onnx"), "{err}");

        // Only int8: used, as a complete set.
        touch(&int8);
        let got = pick_model_files(&d).unwrap();
        assert!(
            got.int8 && got.decoder.ends_with("decoder.int8.onnx"),
            "{got:?}"
        );
        assert_eq!(got.precision(), "int8");

        // Both complete: full precision, whatever the sort order.
        touch(&full);
        let got = pick_model_files(&d).unwrap();
        assert!(!got.int8 && got.joiner.ends_with("/joiner.onnx"), "{got:?}");
        assert_eq!(got.precision(), "full");

        // A full-precision encoder alone does not drag an int8 decoder along:
        // the full set is incomplete, so the whole int8 set is used.
        rm(&["decoder.onnx", "joiner.onnx"]);
        let got = pick_model_files(&d).unwrap();
        assert!(
            got.int8 && got.encoder.ends_with("encoder.int8.onnx"),
            "{got:?}"
        );

        // Neither set complete (fp32 encoder, int8 decoder/joiner): an error.
        rm(&["encoder.int8.onnx"]);
        let err = pick_model_files(&d).unwrap_err();
        assert!(
            err.contains("decoder.onnx") && err.contains("joiner.onnx"),
            "{err}"
        );

        rm(&["encoder.onnx", "decoder.int8.onnx", "joiner.int8.onnx"]);
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
            unresolved_external_data(&encoder, &elsewhere).unwrap(),
            vec!["encoder.weights".to_string()]
        );
        // A different file of the same name in the working directory is not it.
        std::fs::write(elsewhere.join("encoder.weights"), b"other").unwrap();
        assert_eq!(
            unresolved_external_data(&encoder, &elsewhere)
                .unwrap()
                .len(),
            1
        );
        // Positive case: the container runs with working_dir = model dir.
        assert!(
            unresolved_external_data(&encoder, &model)
                .unwrap()
                .is_empty()
        );

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
            unresolved_external_data(&encoder, &elsewhere).unwrap(),
            vec!["encoder.onnx.data".to_string()]
        );
        let int8 = model.join("encoder.int8.onnx");
        std::fs::write(&int8, b"self-contained graph").unwrap();
        assert!(
            unresolved_external_data(&int8, &elsewhere)
                .unwrap()
                .is_empty()
        );

        for f in ["encoder.onnx", "encoder.onnx.data", "encoder.int8.onnx"] {
            std::fs::remove_file(model.join(f)).unwrap();
        }
        std::fs::remove_dir(&model).unwrap();
        std::fs::remove_dir(&elsewhere).unwrap();
    }

    /// A check that could not run is an error, not an empty "all clear": an
    /// encoder that cannot be read (here, a directory where the file should be).
    #[test]
    fn an_unreadable_encoder_is_an_error_not_a_pass() {
        let model = scratch("unreadable");
        let encoder = model.join("encoder.onnx");
        std::fs::create_dir(&encoder).unwrap();
        std::fs::write(model.join("encoder.weights"), b"w").unwrap();
        let err = unresolved_external_data(&encoder, &model).unwrap_err();
        assert!(err.contains("encoder.onnx"), "{err}");
        // A model directory that cannot be listed is an error too.
        let gone = model.join("missing").join("encoder.onnx");
        assert!(unresolved_external_data(&gone, &model).is_err());
        std::fs::remove_file(model.join("encoder.weights")).unwrap();
        std::fs::remove_dir(&encoder).unwrap();
        std::fs::remove_dir(&model).unwrap();
    }

    /// A name that straddles a read-block boundary is still found.
    #[test]
    fn a_name_across_a_block_boundary_is_found() {
        let model = scratch("straddle");
        let elsewhere = scratch("straddle-cwd");
        let encoder = model.join("encoder.onnx");
        std::fs::write(model.join("encoder.weights"), b"w").unwrap();
        for offset in 0..8usize {
            let mut bytes = vec![0u8; 32];
            bytes[offset + 4..offset + 4 + 15].copy_from_slice(b"encoder.weights");
            std::fs::write(&encoder, &bytes).unwrap();
            // 8-byte blocks: the name crosses a boundary for most offsets.
            let got = unresolved_external_data_in(&encoder, &elsewhere, 8).unwrap();
            assert_eq!(got, vec!["encoder.weights".to_string()], "offset {offset}");
        }
        std::fs::write(&encoder, vec![0u8; 32]).unwrap();
        assert!(
            unresolved_external_data_in(&encoder, &elsewhere, 8)
                .unwrap()
                .is_empty()
        );
        std::fs::remove_file(model.join("encoder.weights")).unwrap();
        std::fs::remove_file(&encoder).unwrap();
        std::fs::remove_dir(&model).unwrap();
        std::fs::remove_dir(&elsewhere).unwrap();
    }

    /// The scan has no size cutoff: an encoder over 512 MiB that names an
    /// unresolvable sidecar is reported like a small one (sparse file, so the
    /// test stores almost nothing).
    #[test]
    fn a_large_encoder_is_scanned_not_skipped() {
        use std::io::{Seek, SeekFrom, Write};
        let model = scratch("large");
        let elsewhere = scratch("large-cwd");
        let encoder = model.join("encoder.onnx");
        std::fs::write(model.join("encoder.weights"), b"w").unwrap();
        let mut f = std::fs::File::create(&encoder).unwrap();
        f.seek(SeekFrom::Start(600 << 20)).unwrap();
        f.write_all(b"encoder.weights").unwrap();
        drop(f);
        assert_eq!(
            unresolved_external_data(&encoder, &elsewhere).unwrap(),
            vec!["encoder.weights".to_string()]
        );
        std::fs::remove_file(model.join("encoder.weights")).unwrap();
        std::fs::remove_file(&encoder).unwrap();
        std::fs::remove_dir(&model).unwrap();
        std::fs::remove_dir(&elsewhere).unwrap();
    }
}
