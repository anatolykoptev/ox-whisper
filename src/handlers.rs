use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::language::{self, PARAKEET_V3_LANGS};
use crate::models::{Models, PARAKEET_MODEL_NAME};
use crate::tmpfile::TempFile;
use crate::transcribe;
use crate::upload::{next_part, store_audio_part, text_part};

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
    languages: HashMap<&'static str, LanguageInfo>,
}

#[derive(Serialize)]
struct LanguageInfo {
    model: &'static str,
    ready: bool,
}

/// Readiness, not liveness: 503 while Parakeet cannot serve (no model, or an
/// evicted slot that will not reload), so a probe that only checks the status
/// code sees it. Reads load-time and pool state only — never a recognizer
/// slot, which would make every healthcheck compete with requests.
pub async fn health(State(state): State<Arc<AppState>>) -> (StatusCode, Json<HealthResponse>) {
    let ready = state.models.parakeet_ready();
    let languages = PARAKEET_V3_LANGS
        .iter()
        .map(|lang| {
            (
                *lang,
                LanguageInfo {
                    model: PARAKEET_MODEL_NAME,
                    ready,
                },
            )
        })
        .collect();
    let code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        code,
        Json(HealthResponse {
            status: if ready { "ok" } else { "degraded" },
            engine: "sherpa-onnx",
            version: env!("CARGO_PKG_VERSION"),
            vad: state.models.vad.is_some(),
            languages,
        }),
    )
}

#[derive(Deserialize)]
pub struct TranscribeRequest {
    pub audio_path: String,
    /// Language hint; empty or `auto` means none. Checked against the
    /// languages Parakeet covers.
    #[serde(default)]
    pub language: String,
    pub vad: Option<bool>,
    #[serde(default)]
    pub max_chunk_len: usize,
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

/// Request count and latency for one endpoint call.
pub(crate) fn observe(endpoint: &'static str, ok: bool, start: std::time::Instant) {
    let status = if ok { "ok" } else { "err" };
    metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => status)
        .increment(1);
    metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
        .record(start.elapsed().as_secs_f64());
}

pub async fn transcribe_json(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TranscribeRequest>,
) -> Response {
    let endpoint = "transcribe_json";
    let start = std::time::Instant::now();

    let language = match language::resolve(&req.language) {
        Ok(l) => l,
        Err(e) => {
            observe(endpoint, false, start);
            return e.into_response();
        }
    };
    let audio_path = req.audio_path.clone();
    let vad = req.vad;
    let max_chunk_len = req.max_chunk_len;

    let result = tokio::task::spawn_blocking(move || {
        transcribe::transcribe(
            &state.models,
            &state.config,
            Path::new(&audio_path),
            language.unwrap_or("auto"),
            vad,
            max_chunk_len,
        )
    })
    .await
    .unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

    observe(endpoint, result.is_ok(), start);
    to_response(result).into_response()
}

pub async fn transcribe_upload(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let endpoint = "transcribe_upload";
    let start = std::time::Instant::now();

    let upload = match parse_upload(&mut multipart, &state.config.upload_dir).await {
        Ok(u) => u,
        Err(msg) => {
            observe(endpoint, false, start);
            return Json(TranscribeResponse {
                text: String::new(),
                chunks: Vec::new(),
                duration_ms: 0.0,
                speech_ms: 0.0,
                words: Vec::new(),
                confidence: None,
                error: Some(msg),
            })
            .into_response();
        }
    };
    let language = match language::resolve(&upload.language) {
        Ok(l) => l,
        Err(e) => {
            observe(endpoint, false, start);
            return e.into_response();
        }
    };
    let vad = upload.vad;
    let max_chunk_len = upload.max_chunk_len;
    // The blocking job owns the file: if this handler is dropped (the
    // client went away) the job still finishes, and the file goes with
    // it — never before.
    let file = upload.file;

    let result = tokio::task::spawn_blocking(move || {
        transcribe::transcribe(
            &state.models,
            &state.config,
            file.path(),
            language.unwrap_or("auto"),
            vad,
            max_chunk_len,
        )
    })
    .await
    .unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

    observe(endpoint, result.is_ok(), start);
    to_response(result).into_response()
}

pub(crate) struct UploadData {
    /// The uploaded audio; removed when the upload is dropped.
    pub file: TempFile,
    /// The language hint as sent; see [`language::resolve`].
    pub language: String,
    pub vad: Option<bool>,
    pub max_chunk_len: usize,
}

pub(crate) async fn parse_upload(
    multipart: &mut Multipart,
    dir: &Path,
) -> Result<UploadData, String> {
    let mut file: Option<TempFile> = None;
    let mut language = String::new();
    let mut vad: Option<bool> = None;
    let mut max_chunk_len: usize = 0;

    while let Some(field) = next_part(multipart).await? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "audio" => store_audio_part(field, dir, &mut file).await?,
            "language" => language = text_part(field).await?,
            "vad" => vad = parse_bool(&text_part(field).await?),
            "max_chunk_len" => max_chunk_len = text_part(field).await?.parse().unwrap_or(0),
            _ => {}
        }
    }

    Ok(UploadData {
        file: file.ok_or("missing 'file' or 'audio' field")?,
        language,
        vad,
        max_chunk_len,
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
