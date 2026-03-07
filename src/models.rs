use std::path::Path;
use std::sync::Mutex;

use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use sherpa_rs::nemo_ctc::{NemoCtcConfig, NemoCtcRecognizer};
use sherpa_rs::online_punctuate::{OnlinePunctuation, OnlinePunctuationConfig};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};

use crate::config::Config;
use crate::pool::Pool;
use crate::recognizer::RuRecognizer;

pub struct Models {
    pub en: Option<Pool<MoonshineRecognizer>>,
    pub ru: Option<Pool<RuRecognizer>>,
    pub vad: Option<Mutex<SileroVad>>,
    pub punct: Option<Mutex<OnlinePunctuation>>,
}

impl Models {
    pub fn load(config: &Config) -> Self {
        let en = load_moonshine(config);
        let ru = load_ru(config);
        let vad = load_vad(config);
        let punct = load_punctuation(config);

        warmup(&en, "EN");
        warmup(&ru, "RU");

        Self { en, ru, vad, punct }
    }
}

fn load_moonshine(config: &Config) -> Option<Pool<MoonshineRecognizer>> {
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
        None
    } else {
        Some(Pool::new(recognizers))
    }
}

fn load_ru(config: &Config) -> Option<Pool<RuRecognizer>> {
    // Try NeMo CTC (GigaAM) first — faster, better WER
    let nemo_model = format!("{}/model.int8.onnx", config.ru_models_dir);
    let nemo_tokens = format!("{}/tokens.txt", config.ru_models_dir);
    if Path::new(&nemo_model).exists() && !Path::new(&format!("{}/encoder.int8.onnx", config.ru_models_dir)).exists() {
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

fn load_nemo_ctc(config: &Config, model: &str, tokens: &str) -> Option<Pool<RuRecognizer>> {
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
    if recognizers.is_empty() { None } else { Some(Pool::new(recognizers)) }
}

fn load_zipformer(config: &Config, encoder_path: &str) -> Option<Pool<RuRecognizer>> {
    let transducer_cfg = TransducerConfig {
        encoder: encoder_path.to_string(),
        decoder: format!("{}/decoder.int8.onnx", config.ru_models_dir),
        joiner: format!("{}/joiner.int8.onnx", config.ru_models_dir),
        tokens: format!("{}/tokens.txt", config.ru_models_dir),
        num_threads: config.num_threads,
        provider: Some(config.provider.clone()),
        ..Default::default()
    };
    let mut recognizers = Vec::new();
    for i in 0..config.pool_size {
        match TransducerRecognizer::new(transducer_cfg.clone()) {
            Ok(r) => {
                tracing::info!("RU Zipformer {}/{} loaded", i + 1, config.pool_size);
                recognizers.push(RuRecognizer::Transducer(r));
            }
            Err(e) => {
                tracing::error!("RU Zipformer {}/{} failed: {}", i + 1, config.pool_size, e);
                break;
            }
        }
    }
    if recognizers.is_empty() { None } else { Some(Pool::new(recognizers)) }
}

fn load_vad(config: &Config) -> Option<Mutex<SileroVad>> {
    if !Path::new(&config.vad_model).exists() {
        tracing::warn!("VAD model not found at {}, skipping", config.vad_model);
        return None;
    }
    let cfg = SileroVadConfig {
        model: config.vad_model.clone(), threshold: config.vad_threshold,
        min_silence_duration: config.vad_min_silence_s, min_speech_duration: 0.25,
        window_size: 512, ..Default::default()
    };
    match SileroVad::new(cfg, config.max_audio_duration_s as f32) {
        Ok(v) => { tracing::info!("VAD loaded from {}", config.vad_model); Some(Mutex::new(v)) }
        Err(e) => { tracing::error!("VAD load failed: {}", e); None }
    }
}

fn load_punctuation(config: &Config) -> Option<Mutex<OnlinePunctuation>> {
    if !Path::new(&config.punct_model).exists() {
        tracing::warn!("Punctuation model not found at {}, skipping", config.punct_model);
        return None;
    }
    let cfg = OnlinePunctuationConfig {
        cnn_bilstm: config.punct_model.clone(),
        bpe_vocab: config.punct_vocab.clone(),
        ..Default::default()
    };
    match OnlinePunctuation::new(cfg) {
        Ok(p) => { tracing::info!("Punctuation loaded from {}", config.punct_model); Some(Mutex::new(p)) }
        Err(e) => { tracing::error!("Punctuation load failed: {}", e); None }
    }
}

fn warmup(pool: &Option<Pool<impl Warmable>>, label: &str) {
    if let Some(p) = pool {
        if let Some(mut r) = p.acquire() {
            r.warmup();
            tracing::info!("{} warmup complete", label);
        }
    }
}

trait Warmable { fn warmup(&mut self); }
impl Warmable for MoonshineRecognizer {
    fn warmup(&mut self) { let _ = self.transcribe(16000, &[0.0f32; 16000]); }
}
impl Warmable for RuRecognizer {
    fn warmup(&mut self) { let _ = self.transcribe(16000, &[0.0f32; 16000]); }
}
