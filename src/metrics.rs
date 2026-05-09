//! Prometheus metrics for ox-whisper.
use std::net::SocketAddr;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

pub mod names {
    pub const REQUESTS_TOTAL: &str = "oxwhisper_requests_total";
    pub const REQUEST_DURATION: &str = "oxwhisper_request_duration_seconds";
    pub const TRANSCRIBE_DURATION: &str = "oxwhisper_transcribe_duration_seconds";
    pub const AUDIO_DURATION: &str = "oxwhisper_audio_duration_seconds";
    pub const VAD_SPEECH_RATIO: &str = "oxwhisper_vad_speech_ratio";
    pub const CHUNKS_TOTAL: &str = "oxwhisper_chunks_total";
    pub const HALLUCINATION_REJECTED: &str = "oxwhisper_hallucination_rejected_total";
    pub const POOL_SIZE: &str = "oxwhisper_recognizer_pool_size";
    pub const POOL_BUSY: &str = "oxwhisper_recognizer_pool_busy";
    pub const WS_ACTIVE: &str = "oxwhisper_ws_active_connections";
    pub const POOL_EVICTIONS: &str = "oxwhisper_pool_evictions_total";
    pub const POOL_COLD_STARTS: &str = "oxwhisper_pool_cold_starts_total";
    pub const POOL_REINIT_FAILURES: &str = "oxwhisper_pool_reinit_failures_total";
    pub const POOL_EVICTION_LOOP_PANICS: &str = "oxwhisper_pool_eviction_loop_panics_total";
    pub const POOL_MUTEX_POISONED: &str = "oxwhisper_pool_mutex_poisoned_total";
}

pub fn install_recorder() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install prometheus recorder")
}

pub async fn serve(handle: PrometheusHandle, addr: SocketAddr) {
    use axum::{Router, routing::get};
    let app = Router::new().route(
        "/metrics",
        get(move || {
            let h = handle.clone();
            async move { h.render() }
        }),
    );
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("metrics listener bind failed on {addr}: {e}");
            return;
        }
    };
    tracing::info!("metrics endpoint on http://{addr}/metrics");
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("metrics server stopped: {e}");
    }
}
