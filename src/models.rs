use std::path::Path;
use std::sync::Mutex;

use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use sherpa_rs::online_punctuate::{OnlinePunctuation, OnlinePunctuationConfig};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};

use crate::config::Config;
use crate::pool::Pool;

pub struct Models {
    pub en: Option<Pool<MoonshineRecognizer>>,
    pub ru: Option<Pool<TransducerRecognizer>>,
    pub vad: Option<Mutex<SileroVad>>,
    pub punct: Option<Mutex<OnlinePunctuation>>,
}

impl Models {
    pub fn load(config: &Config) -> Self {
        let en = load_moonshine(config);
        let ru = load_zipformer(config);
        let vad = load_vad(config);
        let punct = load_punctuation(config);

        warmup_en(&en);
        warmup_ru(&ru);

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

fn load_zipformer(config: &Config) -> Option<Pool<TransducerRecognizer>> {
    let encoder_path = format!("{}/encoder.int8.onnx", config.ru_models_dir);
    if !Path::new(&encoder_path).exists() {
        tracing::warn!("RU model not found at {}, skipping", encoder_path);
        return None;
    }

    let transducer_cfg = TransducerConfig {
        encoder: encoder_path,
        decoder: format!("{}/decoder.int8.onnx", config.ru_models_dir),
        joiner: format!("{}/joiner.int8.onnx", config.ru_models_dir),
        tokens: format!("{}/tokens.txt", config.ru_models_dir),
        num_threads: config.num_threads,
        ..Default::default()
    };

    let mut recognizers = Vec::new();
    for i in 0..config.pool_size {
        match TransducerRecognizer::new(transducer_cfg.clone()) {
            Ok(r) => {
                tracing::info!("RU recognizer {}/{} loaded", i + 1, config.pool_size);
                recognizers.push(r);
            }
            Err(e) => {
                tracing::error!("RU recognizer {}/{} failed: {}", i + 1, config.pool_size, e);
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

fn load_vad(config: &Config) -> Option<Mutex<SileroVad>> {
    if !Path::new(&config.vad_model).exists() {
        tracing::warn!("VAD model not found at {}, skipping", config.vad_model);
        return None;
    }

    let vad_cfg = SileroVadConfig {
        model: config.vad_model.clone(),
        threshold: 0.5,
        min_silence_duration: 0.5,
        min_speech_duration: 0.25,
        window_size: 512,
        ..Default::default()
    };

    match SileroVad::new(vad_cfg, config.max_audio_duration_s as f32) {
        Ok(vad) => {
            tracing::info!("VAD model loaded from {}", config.vad_model);
            Some(Mutex::new(vad))
        }
        Err(e) => {
            tracing::error!("Failed to load VAD model: {}", e);
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

    let punct_cfg = OnlinePunctuationConfig {
        cnn_bilstm: config.punct_model.clone(),
        bpe_vocab: config.punct_vocab.clone(),
        ..Default::default()
    };

    match OnlinePunctuation::new(punct_cfg) {
        Ok(punct) => {
            tracing::info!("Punctuation model loaded from {}", config.punct_model);
            Some(Mutex::new(punct))
        }
        Err(e) => {
            tracing::error!("Failed to load punctuation model: {}", e);
            None
        }
    }
}

fn warmup_en(en: &Option<Pool<MoonshineRecognizer>>) {
    if let Some(pool) = en {
        if let Some(mut rec) = pool.acquire() {
            let silence = vec![0.0f32; 16000];
            let _ = rec.transcribe(16000, &silence);
            tracing::info!("EN warmup complete");
        }
    }
}

fn warmup_ru(ru: &Option<Pool<TransducerRecognizer>>) {
    if let Some(pool) = ru {
        if let Some(mut rec) = pool.acquire() {
            let silence = vec![0.0f32; 16000];
            let _ = rec.transcribe(16000, &silence);
            tracing::info!("RU warmup complete");
        }
    }
}
