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
    /// VAD passes that found no speech at all (→ empty transcript), by caller:
    /// `batch`, `sse`, and `ws_poll` — the WebSocket check that runs on every
    /// frame, where silence is normal, so alert on `batch`/`sse` only.
    pub const VAD_NO_SPEECH: &str = "oxwhisper_vad_no_speech_total";
    /// Recoveries from a VAD mutex poisoned by a panic.
    pub const VAD_MUTEX_POISONED: &str = "oxwhisper_vad_mutex_poisoned_total";
    pub const HALLUCINATION_REJECTED: &str = "oxwhisper_hallucination_rejected_total";
    pub const POOL_SIZE: &str = "oxwhisper_recognizer_pool_size";
    pub const POOL_BUSY: &str = "oxwhisper_recognizer_pool_busy";
    pub const WS_ACTIVE: &str = "oxwhisper_ws_active_connections";
    /// Interim WebSocket decodes skipped because the decode could not run.
    pub const WS_INTERIM_SKIPPED: &str = "oxwhisper_ws_interim_skipped_total";
    /// WebSocket sessions closed because their audio buffer hit its cap.
    pub const WS_BUFFER_LIMIT: &str = "oxwhisper_ws_buffer_limit_total";
    pub const POOL_EVICTIONS: &str = "oxwhisper_pool_evictions_total";
    pub const POOL_COLD_STARTS: &str = "oxwhisper_pool_cold_starts_total";
    pub const POOL_REINIT_FAILURES: &str = "oxwhisper_pool_reinit_failures_total";
    pub const POOL_EVICTION_LOOP_PANICS: &str = "oxwhisper_pool_eviction_loop_panics_total";
    pub const POOL_MUTEX_POISONED: &str = "oxwhisper_pool_mutex_poisoned_total";
    /// Time an acquire waited for a busy pool slot (only waits are recorded).
    pub const POOL_ACQUIRE_WAIT: &str = "oxwhisper_pool_acquire_wait_seconds";
    /// Requests served by a fallback model because Parakeet failed to reload.
    pub const ROUTE_FALLBACK: &str = "oxwhisper_route_fallback_total";
    /// Acquires that gave up after the bounded wait (→ request error).
    pub const POOL_ACQUIRE_TIMEOUTS: &str = "oxwhisper_pool_acquire_timeouts_total";
    // Referenced only from tts::supervisor — dead in the bin target until the
    // speech proxy lands (see src/tts/mod.rs).
    #[cfg_attr(not(test), allow(dead_code))]
    pub const TTS_CHILD_UP: &str = "oxwhisper_tts_child_up";
    #[cfg_attr(not(test), allow(dead_code))]
    pub const TTS_CHILD_STARTS: &str = "oxwhisper_tts_child_starts_total";
    #[cfg_attr(not(test), allow(dead_code))]
    pub const TTS_CHILD_RESTARTS: &str = "oxwhisper_tts_child_restarts_total";
    #[cfg_attr(not(test), allow(dead_code))]
    pub const TTS_CHILD_STOPS: &str = "oxwhisper_tts_child_stops_total";
    #[cfg_attr(not(test), allow(dead_code))]
    pub const TTS_CHILD_START_FAILURES: &str = "oxwhisper_tts_child_start_failures_total";
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

/// Counting recorder for tests: `metrics::with_local_recorder` scopes it to the
/// calling thread, so tests stay independent.
#[cfg(test)]
pub mod test_recorder {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use metrics::{
        Counter, CounterFn, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
    };

    #[derive(Default)]
    struct Cell(AtomicU64);
    impl CounterFn for Cell {
        fn increment(&self, value: u64) {
            self.0.fetch_add(value, Ordering::Relaxed);
        }
        fn absolute(&self, value: u64) {
            self.0.store(value, Ordering::Relaxed);
        }
    }

    #[derive(Default)]
    pub struct CountingRecorder {
        counters: Mutex<HashMap<String, Arc<Cell>>>,
    }

    fn id(name: &str, labels: &[(&str, &str)]) -> String {
        let mut l: Vec<String> = labels.iter().map(|(k, v)| format!("{k}={v}")).collect();
        l.sort();
        format!("{name}{{{}}}", l.join(","))
    }

    impl CountingRecorder {
        pub fn count(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
            self.counters
                .lock()
                .unwrap()
                .get(&id(name, labels))
                .map(|c| c.0.load(Ordering::Relaxed))
                .unwrap_or(0)
        }
    }

    impl Recorder for CountingRecorder {
        fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> Counter {
            let labels: Vec<(&str, &str)> = key.labels().map(|l| (l.key(), l.value())).collect();
            let cell = self
                .counters
                .lock()
                .unwrap()
                .entry(id(key.name(), &labels))
                .or_default()
                .clone();
            Counter::from_arc(cell)
        }
        fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> Gauge {
            Gauge::noop()
        }
        fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
            Histogram::noop()
        }
    }
}
