//! Tests for the TTS child supervisor. The fake child is
//! `tests/fake_tts.py` — a python3 script serving `GET /health` -> 200 with
//! marker-file driven crash/delay variants. If python3 is missing these tests
//! FAIL, never skip.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serial_test::serial;
use tokio::time::Instant;

use super::{TtsState, TtsSupervisor};
use crate::config::TtsConfig;

fn require_python3() {
    let ok = std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    assert!(
        ok,
        "python3 is required for tests/fake_tts.py — failing, not skipping"
    );
}

fn fake_bin() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fake_tts.py");
    assert!(p.exists(), "fake_tts.py missing at {p:?}");
    p
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn test_config(bin: PathBuf, port: u16, idle_stop_secs: u64) -> TtsConfig {
    TtsConfig {
        enabled: true,
        bin: bin.to_string_lossy().into_owned(),
        model: "talker.gguf".into(),
        codec: "codec.gguf".into(),
        port,
        threads: 1,
        max_batch: 1,
        idle_stop_secs,
        startup_timeout_secs: 10,
        upstream_url: None,
    }
}

fn marker(name: &str, port: u16) -> PathBuf {
    std::env::temp_dir().join(format!("oxw_fake_tts_{name}_{port}"))
}

fn cleanup_markers(port: u16) {
    for name in ["crash", "delay", "oom"] {
        let _ = std::fs::remove_file(marker(name, port));
    }
}

fn metrics() -> &'static PrometheusHandle {
    static H: OnceLock<PrometheusHandle> = OnceLock::new();
    H.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .expect("install prometheus recorder")
    })
}

/// Current value of a rendered counter/gauge line, e.g.
/// `oxwhisper_tts_child_stops_total{reason="idle"}`. 0 when absent.
fn metric(name: &str) -> f64 {
    metrics()
        .render()
        .lines()
        .find(|l| l.starts_with(name))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

async fn wait_state(sup: &TtsSupervisor, want: TtsState, within: Duration) -> TtsState {
    let deadline = Instant::now() + within;
    loop {
        let s = sup.state();
        if s == want || Instant::now() >= deadline {
            return s;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn http_client() -> Client<HttpConnector, Empty<Bytes>> {
    Client::builder(TokioExecutor::new()).build(HttpConnector::new())
}

// ── T1: /health is 200 while TTS is down ────────────────────────────────
// Mutation: in the health handler return StatusCode::SERVICE_UNAVAILABLE when
// tts state != Ready -> RED.
#[tokio::test]
#[serial]
async fn t1_health_200_while_tts_down() {
    let sup = Arc::new(TtsSupervisor::new(test_config(
        PathBuf::from("/nonexistent/tts-server"),
        free_port(),
        0,
    )));
    let state = Arc::new(crate::handlers::AppState {
        models: crate::models::Models::empty(),
        config: crate::config::Config::from_env(),
        tts: Some(sup),
    });
    let app = axum::Router::new()
        .route("/health", axum::routing::get(crate::handlers::health))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let uri: hyper::Uri = format!("http://{addr}/health").parse().unwrap();
    let res = http_client().get(uri).await.unwrap();
    assert_eq!(
        res.status(),
        hyper::StatusCode::OK,
        "/health must stay 200 when the tts child is down"
    );
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["tts"]["enabled"], true);
    assert_ne!(
        json["tts"]["state"], "ready",
        "tts.state must not be 'ready' while the child cannot run"
    );
}

// ── T2: idle stop ───────────────────────────────────────────────────────
// Mutation: replace the idle comparison in the idle loop with `false` -> RED.
#[tokio::test]
#[serial]
async fn t2_idle_stop() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap(); // fake stays up
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 1)));
    let before = metric("oxwhisper_tts_child_stops_total{reason=\"idle\"}");
    let _idle = sup.spawn_idle_loop(Duration::from_millis(100));

    sup.ensure_ready().await.expect("fake child should start");
    assert_eq!(sup.state(), TtsState::Ready);

    let s = wait_state(&sup, TtsState::Stopped, Duration::from_secs(5)).await;
    assert_eq!(s, TtsState::Stopped, "idle child must be stopped within 5s");
    assert!(
        metric("oxwhisper_tts_child_stops_total{reason=\"idle\"}") - before >= 1.0,
        "stops_total{{reason=\"idle\"}} must increase"
    );
    sup.shutdown();
    cleanup_markers(port);
}

// ── T3: no idle stop while a guard is alive ─────────────────────────────
// Mutation: ignore the guard count in the idle check -> RED.
#[tokio::test]
#[serial]
async fn t3_guard_blocks_idle_stop() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 1)));
    let _idle = sup.spawn_idle_loop(Duration::from_millis(100));

    sup.ensure_ready().await.expect("fake child should start");
    let guard = sup.guard();
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(
        sup.state(),
        TtsState::Ready,
        "child must stay up while a request guard is held"
    );
    drop(guard);
    sup.shutdown();
    cleanup_markers(port);
}

// ── T4: crash -> restart ────────────────────────────────────────────────
// Mutation: make ensure_ready return an error when state is Crashed -> RED.
#[tokio::test]
#[serial]
async fn t4_crash_then_restart() {
    require_python3();
    let port = free_port();
    cleanup_markers(port); // no crash marker -> fake serves then exits(42)
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0)));
    let before = metric("oxwhisper_tts_child_restarts_total");

    sup.ensure_ready()
        .await
        .expect("first start should succeed");
    assert_eq!(sup.state(), TtsState::Ready);

    let s = wait_state(&sup, TtsState::Crashed, Duration::from_secs(5)).await;
    assert_eq!(
        s,
        TtsState::Crashed,
        "unrequested exit must mark state Crashed"
    );
    assert!(
        metric("oxwhisper_tts_child_restarts_total") - before >= 1.0,
        "restarts_total must increase on unrequested exit"
    );

    // Crash marker now exists -> next spawn serves normally.
    let url = sup
        .ensure_ready()
        .await
        .expect("ensure_ready must restart a crashed child");
    assert_eq!(sup.state(), TtsState::Ready);
    assert_eq!(url, format!("http://127.0.0.1:{port}"));
    sup.shutdown();
    cleanup_markers(port);
}

// ── T5: single-flight start ─────────────────────────────────────────────
// Mutation: remove the start lock -> RED.
#[tokio::test]
#[serial]
async fn t5_single_flight() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    std::fs::write(marker("delay", port), "500").unwrap(); // 500ms slow start
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0)));
    let before = metric("oxwhisper_tts_child_starts_total");

    let mut set = tokio::task::JoinSet::new();
    for _ in 0..10 {
        let s = Arc::clone(&sup);
        set.spawn(async move { s.ensure_ready().await });
    }
    let mut oks = 0;
    while let Some(r) = set.join_next().await {
        let res = r.expect("ensure_ready task panicked");
        assert!(res.is_ok(), "ensure_ready failed: {res:?}");
        oks += 1;
    }
    assert_eq!(oks, 10);
    assert_eq!(
        metric("oxwhisper_tts_child_starts_total") - before,
        1.0,
        "10 concurrent ensure_ready must share ONE child start"
    );
    sup.shutdown();
    cleanup_markers(port);
}

// ── T6: oom_score_adj applied (Linux) ───────────────────────────────────
// Mutation: delete the pre_exec write -> RED.
#[cfg(target_os = "linux")]
#[tokio::test]
#[serial]
async fn t6_oom_score_adj_applied() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0)));

    sup.ensure_ready().await.expect("fake child should start");

    let path = marker("oom", port);
    let mut val = String::new();
    for _ in 0..60 {
        if let Ok(v) = std::fs::read_to_string(&path) {
            val = v;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        val.trim(),
        "1000",
        "tts child must run with oom_score_adj=1000 (read from {path:?})"
    );
    sup.shutdown();
    cleanup_markers(port);
}
