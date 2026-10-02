/// OpenAI-compatible /v1/audio/transcriptions endpoint.
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::formats;
use crate::handlers::{AppState, observe};
use crate::language;
use crate::models::PARAKEET_MODEL_NAME;
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

    let upload = match upload::parse_openai_upload(&mut multipart, &state.config.upload_dir).await {
        Ok(u) => u,
        Err(msg) => {
            observe(endpoint, false, start);
            return error_response(StatusCode::BAD_REQUEST, &msg);
        }
    };

    // Before any decode: a language the model does not cover must not come
    // back as plausible text with a 200.
    let language = match language::resolve(&upload.language) {
        Ok(l) => l,
        Err(e) => {
            observe(endpoint, false, start);
            return e.into_response();
        }
    };

    // Shared with the blocking job: if this handler is dropped (the client
    // went away) the job keeps running, and the file lives until it is done.
    let job_file = upload.file.clone();
    let format = match crate::openai::parse_response_format(&upload.response_format) {
        Ok(f) => f,
        Err(e) => {
            observe(endpoint, false, start);
            return e.into_response();
        }
    };
    let want_words = upload.want_words;

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        transcribe::transcribe(
            &state_clone.models,
            &state_clone.config,
            job_file.path(),
            language.unwrap_or("auto"),
            None,
        )
    })
    .await
    .unwrap_or_else(|_| Err(transcribe::TranscribeError::NoRecognizer));

    let (response, ok) = match result {
        Ok(mut r) => {
            apply_post_processing(&upload, &mut r, language);
            (
                format_response(format, &r, language, want_words, upload.extra),
                true,
            )
        }
        Err(e) => (
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            false,
        ),
    };

    observe(endpoint, ok, start);
    response
}

pub(crate) fn apply_post_processing(
    upload: &upload::OpenAIUpload,
    r: &mut transcribe::TranscribeResult,
    language: Option<&str>,
) {
    if !upload.custom_spelling.is_empty() {
        r.text = crate::spelling::apply_spelling(&r.text, &upload.custom_spelling);
        crate::spelling::apply_spelling_to_words(&mut r.words, &upload.custom_spelling);
    }
    // The rules are per language. With no hint there is no language to apply
    // them for, and guessing "en" would run English rules over Russian audio.
    if upload.smart_format
        && let Some(lang) = language
    {
        r.text = crate::smart_format::smart_format(&r.text, lang);
    }
    if upload.paragraphs {
        r.text = crate::paragraphs::split_paragraphs(
            &r.text,
            &r.words,
            crate::paragraphs::default_threshold(),
        );
    }
    if !upload.keywords.is_empty() {
        r.text =
            crate::spelling::apply_keyword_boost(&r.text, &upload.keywords, upload.keywords_boost);
        crate::spelling::apply_keyword_boost_to_words(
            &mut r.words,
            &upload.keywords,
            upload.keywords_boost,
        );
    }
    if !upload.pii_types.is_empty() {
        let redactor = crate::pii::PiiRedactor::new();
        r.text = redactor.redact_text(&r.text, &upload.pii_types, upload.pii_format);
        redactor.redact_words(&mut r.words, &upload.pii_types, upload.pii_format);
    }
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let mut data = Vec::new();
    if state.models.parakeet.is_some() {
        data.push(serde_json::json!({
            "id": PARAKEET_MODEL_NAME,
            "object": "model",
            "owned_by": "ox-whisper",
        }));
    }
    axum::Json(serde_json::json!({ "object": "list", "data": data }))
}

pub(crate) fn format_response(
    format: ResponseFormat,
    result: &transcribe::TranscribeResult,
    language: Option<&str>,
    want_words: bool,
    extra: Option<serde_json::Value>,
) -> Response {
    match format {
        ResponseFormat::Json => {
            let body = JsonResponse {
                text: result.text.clone(),
                extra,
            };
            axum::Json(body).into_response()
        }
        ResponseFormat::VerboseJson => {
            let segments = words_to_segments(&result.words);
            let words = if want_words {
                words_to_openai(&result.words)
            } else {
                vec![]
            };
            let body = VerboseJsonResponse {
                text: result.text.clone(),
                language: language.map(str::to_string),
                duration: result.audio_duration_ms / 1000.0,
                segments,
                words,
                extra,
            };
            axum::Json(body).into_response()
        }
        ResponseFormat::Text => (
            StatusCode::OK,
            [("content-type", "text/plain")],
            result.text.clone(),
        )
            .into_response(),
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
