//! The HTTP surface: language routing (one model, so the only decision left is
//! accept or refuse), the response contract the API's clients read, and
//! readiness. Driven through the real handlers over an empty model set, so a
//! request that passes the language check fails later with a 500 — never a 400.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{FromRequest, Multipart, State};
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::config::Config;
use crate::handlers::{self, AppState};
use crate::models::Models;
use crate::tmpfile::{listing, scratch_dir};

const BOUNDARY: &str = "oxw-api-boundary";

fn state(dir: &std::path::Path) -> Arc<AppState> {
    let mut config = Config::from_lookup(&|_| None);
    config.upload_dir = dir.to_path_buf();
    Arc::new(AppState {
        models: Models::empty(),
        config,
        shutdown: crate::server::ShutdownSignal::inert(),
    })
}

/// A multipart body: one audio part, then `fields` as text parts.
async fn multipart(fields: &[(&str, &str)]) -> Multipart {
    let mut body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.ogg\"\r\n\
         Content-Type: audio/ogg\r\n\r\nnot really audio\r\n"
    );
    for (name, value) in fields {
        body.push_str(&format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        ));
    }
    body.push_str(&format!("--{BOUNDARY}--\r\n"));
    let req = Request::builder()
        .method("POST")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap();
    Multipart::from_request(req, &()).await.unwrap()
}

async fn json_of(res: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// The transcription endpoint called with `language`; returns its status,
/// labelled.
async fn statuses(language: &str) -> Vec<(&'static str, StatusCode)> {
    let dir = scratch_dir("api-lang");
    let st = state(&dir);
    let fields = [("language", language)];
    let mut out = Vec::new();

    let res =
        crate::handler_openai::transcriptions(State(st.clone()), multipart(&fields).await).await;
    out.push(("openai", res.status()));

    // A refused or failed request leaves nothing behind.
    assert!(listing(&dir).is_empty(), "stranded: {:?}", listing(&dir));
    std::fs::remove_dir(&dir).unwrap();
    out
}

/// A language Parakeet does not cover is a 400 — never a decode that returns
/// wrong-language text. (The WebSocket's check is in `ws_tests`.)
#[tokio::test]
async fn unsupported_languages_are_400_everywhere() {
    for lang in ["zh", "ja", "ar", "vi", "xx"] {
        for (endpoint, status) in statuses(lang).await {
            assert_eq!(status, StatusCode::BAD_REQUEST, "{endpoint} {lang}");
        }
    }
}

/// The positive control: what the API's clients send, and what clients commonly send,
/// gets past the language check (and then fails on the empty model set, which
/// is a 500, but not a 400).
#[tokio::test]
async fn supported_or_absent_languages_are_not_refused() {
    for lang in ["ru", "en", "uk", "ru-RU", "EN_us", "", "auto"] {
        for (endpoint, status) in statuses(lang).await {
            assert_ne!(status, StatusCode::BAD_REQUEST, "{endpoint} {lang:?}");
        }
    }
}

#[tokio::test]
async fn the_refusal_body_is_openai_style() {
    let dir = scratch_dir("api-body");
    let res = crate::handler_openai::transcriptions(
        State(state(&dir)),
        multipart(&[("language", "zh")]).await,
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let body = json_of(res).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "language");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("'zh' is not supported")
    );
    std::fs::remove_dir(&dir).unwrap();
}

/// Speaker diarization is gone. Answering a `diarize=true` request without
/// speakers would pass for a one-speaker result, so it is refused.
#[tokio::test]
async fn diarize_true_is_refused_and_false_is_not() {
    let dir = scratch_dir("api-diarize");
    let st = state(&dir);
    let res = crate::handler_openai::transcriptions(
        State(st.clone()),
        multipart(&[("diarize", "true")]).await,
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let res =
        crate::handler_openai::transcriptions(State(st), multipart(&[("diarize", "false")]).await)
            .await;
    assert_ne!(res.status(), StatusCode::BAD_REQUEST);
    assert!(listing(&dir).is_empty());
    std::fs::remove_dir(&dir).unwrap();
}

// --- the response contract clients read ---

fn result(text: &str) -> crate::transcribe::TranscribeResult {
    crate::transcribe::TranscribeResult {
        text: text.into(),
        duration_ms: 10.0,
        audio_duration_ms: 2500.0,
        words: vec![],
    }
}

fn format(fmt: crate::openai::ResponseFormat, lang: Option<&str>) -> Response {
    crate::handler_openai::format_response(fmt, &result("привет мир"), lang, false, None)
}

/// `json`, `verbose_json` and `text` are the formats the API's clients send
/// (Go and Python OpenAI-style clients); each keeps the fields its reader uses.
#[tokio::test]
async fn the_three_formats_clients_use_keep_their_shape() {
    use crate::openai::ResponseFormat::*;

    let json = json_of(format(Json, Some("ru"))).await;
    assert_eq!(json["text"], "привет мир");

    let verbose = json_of(format(VerboseJson, Some("ru"))).await;
    assert_eq!(verbose["text"], "привет мир");
    assert_eq!(verbose["language"], "ru");
    assert_eq!(verbose["duration"], 2.5);

    // No hint was sent: no language is reported, none is made up.
    let unhinted = json_of(format(VerboseJson, None)).await;
    assert_eq!(unhinted["text"], "привет мир");
    assert!(unhinted.get("language").is_none(), "{unhinted}");

    let res = format(Text, Some("ru"));
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "text/plain");
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
        .await
        .unwrap();
    assert_eq!(&bytes[..], "привет мир".as_bytes());
}

// --- readiness ---

#[tokio::test]
async fn health_is_503_without_a_model_and_200_with_a_healthy_pool() {
    use crate::pool::EvictablePool;
    use sherpa_rs::transducer::TransducerRecognizer;

    let dir = scratch_dir("api-health");
    let mut st = Arc::try_unwrap(state(&dir)).ok().expect("sole owner");

    // No model: not ready, and the body says why.
    let (code, body) = handlers::health(State(Arc::new(AppState {
        models: Models::empty(),
        config: Config::from_lookup(&|_| None),
        shutdown: crate::server::ShutdownSignal::inert(),
    })))
    .await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    let body = json_of(body.into_response()).await;
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["languages"]["ru"]["ready"], false);

    // A pool that exists and is healthy: ready. (No slots: a recognizer needs
    // real model files, and /health must not take one anyway.)
    let pool: EvictablePool<TransducerRecognizer> =
        EvictablePool::from_items(vec![], 0, Arc::new(|| Err(anyhow::anyhow!("never called"))));
    st.models.parakeet = Some(Arc::new(pool));
    let st = Arc::new(st);
    let (code, body) = handlers::health(State(st.clone())).await;
    assert_eq!(code, StatusCode::OK);
    let body = json_of(body.into_response()).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["languages"]["ru"]["ready"], true);
    assert_eq!(body["languages"]["en"]["model"], "parakeet-tdt-0.6b-v3");
    assert!(body["languages"].get("zh").is_none());

    // Precision is reported once the model set is known.
    let mut st = Arc::try_unwrap(st).ok().expect("sole owner");
    st.models.parakeet_precision = Some("int8");
    let st = Arc::new(st);
    let (_, body) = handlers::health(State(st.clone())).await;
    assert_eq!(json_of(body.into_response()).await["precision"], "int8");

    // A pool whose only slot is evicted still reloads on demand: ready. Once
    // that reload has failed, it is not ready, though the model is configured.
    let evicted: EvictablePool<TransducerRecognizer> =
        EvictablePool::evicted(1, 600, Arc::new(|| Err(anyhow::anyhow!("out of memory"))));
    let pool = Arc::new(evicted);
    let mut st = Arc::try_unwrap(st).ok().expect("sole owner");
    st.models.parakeet = Some(pool.clone());
    let st = Arc::new(st);
    let (code, _) = handlers::health(State(st.clone())).await;
    assert_eq!(code, StatusCode::OK);
    assert!(pool.try_acquire().is_err());
    let (code, _) = handlers::health(State(st)).await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    std::fs::remove_dir(&dir).unwrap();
}

// --- response_format ---

async fn status_with_format(format: &str) -> (StatusCode, serde_json::Value) {
    let dir = scratch_dir("api-format");
    let res = crate::handler_openai::transcriptions(
        State(state(&dir)),
        multipart(&[("response_format", format)]).await,
    )
    .await;
    let status = res.status();
    let body = json_of(res).await;
    assert!(listing(&dir).is_empty(), "stranded: {:?}", listing(&dir));
    std::fs::remove_dir(&dir).unwrap();
    (status, body)
}

/// An unknown `response_format` used to fall back to `json` with a 200: a
/// client that asked for `srt` got a JSON body.
#[tokio::test]
async fn an_unknown_response_format_is_a_400_not_a_silent_json() {
    for bad in ["verbose-json", "xml", "JSON"] {
        let (status, body) = status_with_format(bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
        assert_eq!(body["error"]["code"], "invalid_response_format", "{bad}");
        assert_eq!(body["error"]["param"], "response_format");
        let text = body.to_string().to_lowercase();
        for word in ["unsupported", "corrupted", "invalid file"] {
            assert!(!text.contains(word), "{text}");
        }
    }
}

/// Positive control: the five formats (and none) pass the check, so they fail
/// later on the empty model set with a 500, not a 400.
#[tokio::test]
async fn the_known_response_formats_are_not_refused() {
    for ok in ["", "json", "verbose_json", "text", "srt", "vtt"] {
        let (status, _) = status_with_format(ok).await;
        assert_ne!(status, StatusCode::BAD_REQUEST, "{ok:?}");
    }
}

// --- smart_format ---

/// `smart_format` rules are per language. With no language hint none apply:
/// English rules over Russian audio would be a guess dressed as a feature.
#[tokio::test]
async fn smart_format_applies_only_with_a_language_hint() {
    let dir = scratch_dir("api-smart");
    let upload =
        crate::upload::parse_openai_upload(&mut multipart(&[("smart_format", "true")]).await, &dir)
            .await
            .unwrap_or_else(|e| panic!("{e}"));

    let run = |lang: Option<&str>| {
        let mut r = result("twenty three dogs");
        crate::handler_openai::apply_post_processing(&upload, &mut r, lang);
        r.text
    };
    assert_eq!(
        run(Some("en")),
        "23 dogs",
        "control: with a hint it applies"
    );
    assert_eq!(run(None), "twenty three dogs", "no hint, no rules");

    drop(upload);
    std::fs::remove_dir(&dir).unwrap();
}
