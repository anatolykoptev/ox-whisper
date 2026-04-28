use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, State};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::models::Models;
use crate::transcribe;

pub struct AppState {
    pub models: Models,
    pub config: Config,
}

#[derive(Serialize)]
pub struct HealthResponse {
    status: &'static str,
    engine: &'static str,
    version: &'static str,
    vad: bool,
    punctuation: bool,
    languages: HashMap<&'static str, LanguageInfo>,
}

#[derive(Serialize)]
struct LanguageInfo {
    model: &'static str,
    ready: bool,
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let mut languages = HashMap::new();
    let en_ready = state.models.en.is_some();
    for lang in ["ar", "en", "es", "ja", "uk", "vi", "zh"] {
        languages.insert(lang, LanguageInfo {
            model: "moonshine-v2-base",
            ready: en_ready,
        });
    }
    let ru_model = state.models.ru.as_ref()
        .and_then(|p| p.acquire())
        .map(|r| r.model_name())
        .unwrap_or("none");
    languages.insert("ru", LanguageInfo {
        model: ru_model,
        ready: state.models.ru.is_some(),
    });

    Json(HealthResponse {
        status: "ok",
        engine: "sherpa-onnx",
        version: env!("CARGO_PKG_VERSION"),
        vad: state.models.vad.is_some(),
        punctuation: state.models.punct.is_some(),
        languages,
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

fn default_language() -> String { "en".to_string() }

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

fn is_zero(v: &f64) -> bool { *v == 0.0 }

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
            &state.models, &state.config, Path::new(&audio_path), &language,
            vad, punctuate, max_chunk_len,
        )
    }).await.unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

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
                    &state.models, &state.config, &p, &language,
                    vad, punctuate, max_chunk_len,
                )
            }).await.unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

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
                text: String::new(), chunks: Vec::new(),
                duration_ms: 0.0, speech_ms: 0.0,
                words: Vec::new(), confidence: None, error: Some(msg),
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
                let ext = field.file_name()
                    .and_then(|n| Path::new(n).extension().map(|e| e.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "wav".to_string());
                let tmp = format!("/tmp/{}.{}", uuid::Uuid::new_v4(), ext);
                let data = field.bytes().await.map_err(|e| e.to_string())?;
                std::fs::write(&tmp, &data).map_err(|e: std::io::Error| e.to_string())?;
                file_path = Some(std::path::PathBuf::from(tmp));
            }
            "language" => language = field.text().await.unwrap_or_default(),
            "vad" => vad = field.text().await.unwrap_or_default().parse().ok(),
            "max_chunk_len" => max_chunk_len = field.text().await.unwrap_or_default().parse().unwrap_or(0),
            "punctuate" => punctuate = field.text().await.unwrap_or_default().parse().ok(),
            _ => {}
        }
    }

    Ok(UploadData {
        file_path: file_path.ok_or("missing 'file' or 'audio' field")?,
        language: normalize_language(&language),
        vad, max_chunk_len, punctuate,
    })
}

fn to_response(result: Result<transcribe::TranscribeResult, transcribe::TranscribeError>) -> Json<TranscribeResponse> {
    match result {
        Ok(r) => {
            let confidence = {
                let confs: Vec<f32> = r.words.iter()
                    .filter_map(|w| w.confidence)
                    .collect();
                if confs.is_empty() {
                    None
                } else {
                    Some(confs.iter().sum::<f32>() / confs.len() as f32)
                }
            };
            Json(TranscribeResponse {
                text: r.text, chunks: r.chunks,
                duration_ms: r.duration_ms, speech_ms: r.speech_ms,
                words: r.words, confidence, error: None,
            })
        }
        Err(e) => Json(TranscribeResponse {
            text: String::new(), chunks: Vec::new(),
            duration_ms: 0.0, speech_ms: 0.0,
            words: Vec::new(), confidence: None, error: Some(e.to_string()),
        }),
    }
}

fn normalize_language(lang: &str) -> String {
    let normalized = lang.trim().to_lowercase();
    if normalized.is_empty() { "en".to_string() } else { normalized }
}
