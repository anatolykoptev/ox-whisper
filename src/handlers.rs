use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, State};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::models::Models;
use crate::recognizer::PARAKEET_MODEL_NAME;
use crate::routing::Engine;
use crate::transcribe;
use crate::tts::{TtsState, TtsSupervisor};

/// Languages `/health` has always reported for the Moonshine route.
const MOONSHINE_LANGS: &[&str] = &["ar", "en", "es", "ja", "uk", "vi", "zh"];

pub struct AppState {
    pub models: Models,
    pub config: Config,
    /// TTS child supervisor — `None` when `TTS_ENABLED` is false.
    pub tts: Option<Arc<TtsSupervisor>>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    status: &'static str,
    engine: &'static str,
    version: &'static str,
    vad: bool,
    punctuation: bool,
    languages: HashMap<&'static str, LanguageInfo>,
    tts: TtsHealth,
}

/// TTS status in `/health`. The endpoint always answers 200 — a TTS problem
/// must never restart the whole container via the healthcheck.
#[derive(Serialize)]
struct TtsHealth {
    enabled: bool,
    state: TtsState,
}

#[derive(Serialize)]
struct LanguageInfo {
    model: &'static str,
    ready: bool,
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let mut languages = HashMap::new();
    let langs = ["ar", "en", "es", "ja", "uk", "vi", "zh", "ru"]
        .into_iter()
        .chain(crate::config::PARAKEET_V3_LANGS.iter().copied());
    for lang in langs {
        if languages.contains_key(lang) {
            continue;
        }
        let engine = state.models.route(lang, &state.config.parakeet_langs);
        let (model, ready) = match engine {
            Engine::Parakeet => (PARAKEET_MODEL_NAME, true),
            // Read at load time: taking a pool slot here would make every
            // healthcheck compete with requests and block idle eviction.
            Engine::Ru => (state.models.ru_model_name, state.models.ru.is_some()),
            // Moonshine is listed only for the languages it always claimed.
            Engine::Moonshine if !MOONSHINE_LANGS.contains(&lang) => continue,
            Engine::Moonshine => ("moonshine-v2-base", state.models.en.is_some()),
        };
        languages.insert(lang, LanguageInfo { model, ready });
    }

    let tts = match &state.tts {
        Some(sup) => TtsHealth {
            enabled: true,
            state: sup.state(),
        },
        None => TtsHealth {
            enabled: false,
            state: TtsState::Disabled,
        },
    };

    Json(HealthResponse {
        status: "ok",
        engine: "sherpa-onnx",
        version: env!("CARGO_PKG_VERSION"),
        vad: state.models.vad.is_some(),
        punctuation: state.models.punct.is_some(),
        languages,
        tts,
    })
}

#[derive(Deserialize)]
pub struct TranscribeRequest {
    pub audio_path: String,
    #[serde(default = "default_language")]
    pub language: String,
    pub vad: Option<bool>,
    #[serde(default)]
    pub max_chunk_len: usize,
    pub punctuate: Option<bool>,
}

fn default_language() -> String {
    "en".to_string()
}

#[derive(Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<String>,
    pub duration_ms: f64,
    #[serde(skip_serializing_if = "is_zero")]
    pub speech_ms: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<crate::words::WordTimestamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

pub async fn transcribe_json(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TranscribeRequest>,
) -> Json<TranscribeResponse> {
    let endpoint = "transcribe_json";
    let start = std::time::Instant::now();

    let language = normalize_language(&req.language);
    let audio_path = req.audio_path.clone();
    let vad = req.vad;
    let punctuate = req.punctuate;
    let max_chunk_len = req.max_chunk_len;

    let result = tokio::task::spawn_blocking(move || {
        transcribe::transcribe(
            &state.models,
            &state.config,
            Path::new(&audio_path),
            &language,
            vad,
            punctuate,
            max_chunk_len,
        )
    })
    .await
    .unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

    let status = if result.is_ok() { "ok" } else { "err" };
    metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => status)
        .increment(1);
    metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
        .record(start.elapsed().as_secs_f64());

    to_response(result)
}

pub async fn transcribe_upload(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Json<TranscribeResponse> {
    let endpoint = "transcribe_upload";
    let start = std::time::Instant::now();

    match parse_upload(&mut multipart).await {
        Ok(upload) => {
            let path = upload.file_path;
            let language = upload.language;
            let vad = upload.vad;
            let punctuate = upload.punctuate;
            let max_chunk_len = upload.max_chunk_len;
            let p = path.clone();

            let result = tokio::task::spawn_blocking(move || {
                transcribe::transcribe(
                    &state.models,
                    &state.config,
                    &p,
                    &language,
                    vad,
                    punctuate,
                    max_chunk_len,
                )
            })
            .await
            .unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

            let _ = std::fs::remove_file(&path);
            let status = if result.is_ok() { "ok" } else { "err" };
            metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => status)
                .increment(1);
            metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
                .record(start.elapsed().as_secs_f64());
            to_response(result)
        }
        Err(msg) => {
            metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => "err")
                .increment(1);
            metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
                .record(start.elapsed().as_secs_f64());
            Json(TranscribeResponse {
                text: String::new(),
                chunks: Vec::new(),
                duration_ms: 0.0,
                speech_ms: 0.0,
                words: Vec::new(),
                confidence: None,
                error: Some(msg),
            })
        }
    }
}

pub(crate) struct UploadData {
    pub file_path: std::path::PathBuf,
    pub language: String,
    pub vad: Option<bool>,
    pub max_chunk_len: usize,
    pub punctuate: Option<bool>,
}

pub(crate) async fn parse_upload(multipart: &mut Multipart) -> Result<UploadData, String> {
    let mut file_path: Option<std::path::PathBuf> = None;
    let mut language = "en".to_string();
    let mut vad: Option<bool> = None;
    let mut max_chunk_len: usize = 0;
    let mut punctuate: Option<bool> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "audio" => {
                let ext = field
                    .file_name()
                    .and_then(|n| {
                        Path::new(n)
                            .extension()
                            .map(|e| e.to_string_lossy().to_string())
                    })
                    .unwrap_or_else(|| "wav".to_string());
                let tmp = format!("/tmp/{}.{}", uuid::Uuid::new_v4(), ext);
                let data = field.bytes().await.map_err(|e| e.to_string())?;
                std::fs::write(&tmp, &data).map_err(|e: std::io::Error| e.to_string())?;
                file_path = Some(std::path::PathBuf::from(tmp));
            }
            "language" => language = field.text().await.unwrap_or_default(),
            "vad" => vad = parse_bool(&field.text().await.unwrap_or_default()),
            "max_chunk_len" => {
                max_chunk_len = field.text().await.unwrap_or_default().parse().unwrap_or(0)
            }
            "punctuate" => punctuate = parse_bool(&field.text().await.unwrap_or_default()),
            _ => {}
        }
    }

    Ok(UploadData {
        file_path: file_path.ok_or("missing 'file' or 'audio' field")?,
        language: normalize_language(&language),
        vad,
        max_chunk_len,
        punctuate,
    })
}

fn to_response(
    result: Result<transcribe::TranscribeResult, transcribe::TranscribeError>,
) -> Json<TranscribeResponse> {
    match result {
        Ok(r) => {
            let confidence = {
                let confs: Vec<f32> = r.words.iter().filter_map(|w| w.confidence).collect();
                if confs.is_empty() {
                    None
                } else {
                    Some(confs.iter().sum::<f32>() / confs.len() as f32)
                }
            };
            Json(TranscribeResponse {
                text: r.text,
                chunks: r.chunks,
                duration_ms: r.duration_ms,
                speech_ms: r.speech_ms,
                words: r.words,
                confidence,
                error: None,
            })
        }
        Err(e) => Json(TranscribeResponse {
            text: String::new(),
            chunks: Vec::new(),
            duration_ms: 0.0,
            speech_ms: 0.0,
            words: Vec::new(),
            confidence: None,
            error: Some(e.to_string()),
        }),
    }
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

fn normalize_language(lang: &str) -> String {
    let normalized = lang.trim().to_lowercase();
    if normalized.is_empty() {
        "en".to_string()
    } else {
        normalized
    }
}
