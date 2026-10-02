use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;

use crate::config::Config;
use crate::language::PARAKEET_V3_LANGS;
use crate::models::{Models, PARAKEET_MODEL_NAME};

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

/// Request count and latency for one endpoint call.
pub(crate) fn observe(endpoint: &'static str, ok: bool, start: std::time::Instant) {
    let status = if ok { "ok" } else { "err" };
    metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => status)
        .increment(1);
    metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
        .record(start.elapsed().as_secs_f64());
}
