/// OpenAI-compatible /v1/audio/transcriptions endpoint.

use std::path::Path;
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::audio;
use crate::detect::{DetectResult, detect_language};
use crate::formats;
use crate::handlers::AppState;
use crate::openai::{
    JsonResponse, ResponseFormat, VerboseJsonResponse, words_to_openai, words_to_segments,
};
use crate::transcribe;

pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let upload = match parse_openai_upload(&mut multipart).await {
        Ok(u) => u,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };

    let file_path = upload.file_path.clone();
    let (language, lang_confidence) = if upload.language.is_empty() {
        let detection = detect_language_from_file(&state, &file_path);
        (detection.language, Some(detection.confidence))
    } else {
        (upload.language.clone(), None)
    };

    let format = upload.response_format;
    let want_words = upload.want_words;
    let path = file_path.clone();
    let detected_lang = language.clone();

    let result = tokio::task::spawn_blocking(move || {
        transcribe::transcribe(
            &state.models, &state.config, &path, &language,
            None, None, 0,
        )
    })
    .await
    .unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

    let _ = std::fs::remove_file(&file_path);

    match result {
        Ok(mut r) => {
            if !upload.custom_spelling.is_empty() {
                r.text = crate::spelling::apply_spelling(&r.text, &upload.custom_spelling);
                crate::spelling::apply_spelling_to_words(&mut r.words, &upload.custom_spelling);
            }
            format_response(format, &r, &detected_lang, want_words, lang_confidence, upload.extra)
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let mut data = Vec::new();
    if state.models.en.is_some() {
        data.push(serde_json::json!({
            "id": "moonshine-v2-base",
            "object": "model",
            "owned_by": "ox-whisper",
        }));
    }
    if let Some(pool) = state.models.ru.as_ref() {
        let name = pool
            .acquire()
            .map(|r| r.model_name())
            .unwrap_or("ru-model");
        data.push(serde_json::json!({
            "id": name,
            "object": "model",
            "owned_by": "ox-whisper",
        }));
    }
    axum::Json(serde_json::json!({ "object": "list", "data": data }))
}

// --- internals ---

struct OpenAIUpload {
    file_path: std::path::PathBuf,
    language: String,
    response_format: ResponseFormat,
    want_words: bool,
    custom_spelling: Vec<crate::spelling::SpellingRule>,
    extra: Option<serde_json::Value>,
}

async fn parse_openai_upload(multipart: &mut Multipart) -> Result<OpenAIUpload, String> {
    let mut file_path: Option<std::path::PathBuf> = None;
    let mut language = String::new();
    let mut response_format = ResponseFormat::default();
    let mut want_words = false;
    let mut custom_spelling = Vec::new();
    let mut extra: Option<serde_json::Value> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let ext = field
                    .file_name()
                    .and_then(|n| {
                        Path::new(n).extension().map(|e| e.to_string_lossy().to_string())
                    })
                    .unwrap_or_else(|| "wav".to_string());
                let tmp = format!("/tmp/{}.{}", uuid::Uuid::new_v4(), ext);
                let data = field.bytes().await.map_err(|e| e.to_string())?;
                std::fs::write(&tmp, &data).map_err(|e: std::io::Error| e.to_string())?;
                file_path = Some(std::path::PathBuf::from(tmp));
            }
            "language" => language = field.text().await.unwrap_or_default(),
            "response_format" => {
                let val = field.text().await.unwrap_or_default();
                let quoted = format!("\"{}\"", val);
                response_format = serde_json::from_str(&quoted).unwrap_or_default();
            }
            "timestamp_granularities[]" => {
                let val = field.text().await.unwrap_or_default();
                if val == "word" {
                    want_words = true;
                }
            }
            "custom_spelling" => {
                let val = field.text().await.unwrap_or_default();
                if let Ok(rules) = serde_json::from_str::<Vec<crate::spelling::SpellingRule>>(&val) {
                    custom_spelling = rules;
                }
            }
            "extra" => {
                let val = field.text().await.unwrap_or_default();
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&val) {
                    extra = Some(parsed);
                }
            }
            // model, temperature, prompt — accepted but ignored
            _ => { let _ = field.bytes().await; }
        }
    }

    Ok(OpenAIUpload {
        file_path: file_path.ok_or("missing 'file' field")?,
        language: normalize_language(&language),
        response_format,
        want_words,
        custom_spelling,
        extra,
    })
}

fn normalize_language(lang: &str) -> String {
    lang.trim().to_lowercase()
}

fn detect_language_from_file(state: &Arc<AppState>, path: &Path) -> DetectResult {
    let wav_result = audio::ensure_wav(path).and_then(|(wav_path, tmp)| {
        let result = audio::load_wav(&wav_path);
        if tmp {
            let _ = std::fs::remove_file(&wav_path);
        }
        result
    });
    match wav_result {
        Ok((samples, _)) => detect_language(&state.models, &samples),
        Err(e) => {
            tracing::warn!("language detection failed, defaulting to en: {e}");
            DetectResult {
                language: "en".to_string(),
                confidence: 0.0,
            }
        }
    }
}

fn format_response(
    format: ResponseFormat,
    result: &transcribe::TranscribeResult,
    language: &str,
    want_words: bool,
    language_confidence: Option<f64>,
    extra: Option<serde_json::Value>,
) -> Response {
    match format {
        ResponseFormat::Json => {
            let body = JsonResponse { text: result.text.clone(), extra };
            axum::Json(body).into_response()
        }
        ResponseFormat::VerboseJson => {
            let segments = words_to_segments(&result.words);
            let words = if want_words { words_to_openai(&result.words) } else { vec![] };
            let lang = if language.is_empty() { "en" } else { language };
            let body = VerboseJsonResponse {
                text: result.text.clone(),
                language: lang.to_string(),
                duration: result.duration_ms / 1000.0,
                segments,
                words,
                language_confidence,
                extra,
            };
            axum::Json(body).into_response()
        }
        ResponseFormat::Text => {
            (StatusCode::OK, [("content-type", "text/plain")], result.text.clone()).into_response()
        }
        ResponseFormat::Srt => {
            let body = formats::to_srt(&result.words);
            (StatusCode::OK, [("content-type", "text/plain")], body).into_response()
        }
        ResponseFormat::Vtt => {
            let body = formats::to_vtt(&result.words);
            (StatusCode::OK, [("content-type", "text/vtt")], body).into_response()
        }
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
        }
    });
    (status, axum::Json(body)).into_response()
}
