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
use crate::upload;

pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let endpoint = "openai_transcriptions";
    let start = std::time::Instant::now();

    let upload = match upload::parse_openai_upload(&mut multipart).await {
        Ok(u) => u,
        Err(msg) => {
            metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => "err")
                .increment(1);
            metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
                .record(start.elapsed().as_secs_f64());
            return error_response(StatusCode::BAD_REQUEST, &msg);
        }
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

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        transcribe::transcribe(
            &state_clone.models, &state_clone.config, &path, &language,
            None, None, 0,
        )
    })
    .await
    .unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

    let (response, ok) = match result {
        Ok(mut r) => {
            if upload.diarize && state.models.diarize.is_some() {
                run_diarization(&state, &file_path, &mut r, upload.diarize_speakers).await;
            }
            apply_post_processing(&upload, &mut r, &detected_lang);
            let utterances = if upload.diarize {
                crate::diarize::words_to_utterances(&r.words)
            } else {
                vec![]
            };
            (format_response(format, &r, &detected_lang, want_words, lang_confidence, upload.extra, utterances), true)
        }
        Err(e) => (error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()), false),
    };

    let status = if ok { "ok" } else { "err" };
    metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => status)
        .increment(1);
    metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
        .record(start.elapsed().as_secs_f64());

    let _ = std::fs::remove_file(&file_path);
    response
}

async fn run_diarization(
    state: &Arc<AppState>,
    file_path: &Path,
    result: &mut transcribe::TranscribeResult,
    num_speakers: Option<i32>,
) {
    if let Ok((wav_path, tmp)) = audio::ensure_wav(file_path) {
        if let Ok((samples, _)) = audio::load_wav(&wav_path) {
            if tmp { let _ = std::fs::remove_file(&wav_path); }
            let diarize_state = state.clone();
            let mut words = std::mem::take(&mut result.words);
            let diarized = tokio::task::spawn_blocking(move || {
                if let Some(ref engine) = diarize_state.models.diarize {
                    engine.assign_speakers(&samples, &mut words, num_speakers);
                }
                words
            }).await.unwrap_or_default();
            result.words = diarized;
        }
    }
}

fn apply_post_processing(
    upload: &upload::OpenAIUpload,
    r: &mut transcribe::TranscribeResult,
    detected_lang: &str,
) {
    if !upload.custom_spelling.is_empty() {
        r.text = crate::spelling::apply_spelling(&r.text, &upload.custom_spelling);
        crate::spelling::apply_spelling_to_words(&mut r.words, &upload.custom_spelling);
    }
    if upload.smart_format {
        r.text = crate::smart_format::smart_format(&r.text, detected_lang);
    }
    if upload.paragraphs {
        r.text = crate::paragraphs::split_paragraphs(
            &r.text, &r.words, crate::paragraphs::default_threshold(),
        );
    }
    if !upload.keywords.is_empty() {
        r.text = crate::spelling::apply_keyword_boost(&r.text, &upload.keywords, upload.keywords_boost);
        crate::spelling::apply_keyword_boost_to_words(&mut r.words, &upload.keywords, upload.keywords_boost);
    }
    if !upload.pii_types.is_empty() {
        let redactor = crate::pii::PiiRedactor::new();
        r.text = redactor.redact_text(&r.text, &upload.pii_types, upload.pii_format);
        redactor.redact_words(&mut r.words, &upload.pii_types, upload.pii_format);
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

fn detect_language_from_file(state: &Arc<AppState>, path: &Path) -> DetectResult {
    let wav_result = audio::ensure_wav(path).and_then(|(wav_path, tmp)| {
        let result = audio::load_wav(&wav_path);
        if tmp { let _ = std::fs::remove_file(&wav_path); }
        result
    });
    match wav_result {
        Ok((samples, _)) => detect_language(&state.models, &samples),
        Err(e) => {
            tracing::warn!("language detection failed, defaulting to en: {e}");
            DetectResult { language: "en".to_string(), confidence: 0.0 }
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
    utterances: Vec<crate::diarize::Utterance>,
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
                segments, words, language_confidence, utterances, extra,
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
        "error": { "message": message, "type": "invalid_request_error" }
    });
    (status, axum::Json(body)).into_response()
}
