//! Tests for the TTS child supervisor. The fake child is
//! `tests/fake_tts.py` — a python3 script serving `GET /health` -> 200 with
//! marker-file driven crash/delay/exit/hang variants and a SIGTERM marker.
//! If python3 is missing these tests FAIL, never skip.

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

use super::{TtsError, TtsState, TtsSupervisor};
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

fn test_config(
    bin: PathBuf,
    port: u16,
    idle_stop_secs: u64,
    startup_timeout_secs: u64,
) -> TtsConfig {
    TtsConfig {
        enabled: true,
        bin: bin.to_string_lossy().into_owned(),
        model: "talker.gguf".into(),
        codec: "codec.gguf".into(),
        port,
        threads: 1,
        max_batch: 1,
        idle_stop_secs,
        startup_timeout_secs,
        upstream_url: None,
    }
}

fn marker(name: &str, port: u16) -> PathBuf {
    std::env::temp_dir().join(format!("oxw_fake_tts_{name}_{port}"))
}

fn cleanup_markers(port: u16) {
    for name in ["crash", "delay", "oom", "exit", "hang", "sigterm"] {
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

/// Poll a rendered metric until it reaches `want` or `within` elapses.
/// Metrics like `stops_total` are bumped by the monitor task after the reap,
/// so they can lag the state transition — always wait, never assert at once.
async fn wait_metric(name: &str, want: f64, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    loop {
        if metric(name) >= want || Instant::now() >= deadline {
            return metric(name) >= want;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
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
        10,
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
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 1, 10)));
    let before = metric("oxwhisper_tts_child_stops_total{reason=\"idle\"}");
    let _idle = sup.spawn_idle_loop(Duration::from_millis(100));

    let _ = sup.ensure_ready().await.expect("fake child should start");
    assert_eq!(sup.state(), TtsState::Ready);

    let s = wait_state(&sup, TtsState::Stopped, Duration::from_secs(5)).await;
    assert_eq!(s, TtsState::Stopped, "idle child must be stopped within 5s");
    // stops_total is bumped after the monitor reaps — wait for the metric,
    // do not assert it right after wait_state.
    assert!(
        wait_metric(
            "oxwhisper_tts_child_stops_total{reason=\"idle\"}",
            before + 1.0,
            Duration::from_secs(3),
        )
        .await,
        "stops_total{{reason=\"idle\"}} must increase"
    );
    sup.shutdown().await;
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
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 1, 10)));
    let _idle = sup.spawn_idle_loop(Duration::from_millis(100));

    let (_url, guard) = sup.ensure_ready().await.expect("fake child should start");
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(
        sup.state(),
        TtsState::Ready,
        "child must stay up while a request guard is held"
    );
    drop(guard);
    sup.shutdown().await;
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
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0, 10)));
    let before = metric("oxwhisper_tts_child_restarts_total");

    let (_url, _g) = sup
        .ensure_ready()
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
        wait_metric(
            "oxwhisper_tts_child_restarts_total",
            before + 1.0,
            Duration::from_secs(3),
        )
        .await,
        "restarts_total must increase on unrequested exit"
    );

    // Crash marker now exists -> next spawn serves normally.
    let (url, _g) = sup
        .ensure_ready()
        .await
        .expect("ensure_ready must restart a crashed child");
    assert_eq!(sup.state(), TtsState::Ready);
    assert_eq!(url, format!("http://127.0.0.1:{port}"));
    sup.shutdown().await;
    cleanup_markers(port);
}

// ── T5: single-flight start ─────────────────────────────────────────────
// Mutation: join no in-flight generation, always launch a leader -> RED.
#[tokio::test]
#[serial]
async fn t5_single_flight() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    std::fs::write(marker("delay", port), "500").unwrap(); // 500ms slow start
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0, 10)));
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
    sup.shutdown().await;
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
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0, 10)));

    let _ = sup.ensure_ready().await.expect("fake child should start");

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
    sup.shutdown().await;
    cleanup_markers(port);
}

// ── R1: dropped caller does not abort the start ─────────────────────────
// Mutation: run the start inline in the caller instead of the spawned leader
// -> RED.
#[tokio::test]
#[serial]
async fn r1_dropped_caller_keeps_start() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    std::fs::write(marker("delay", port), "1000").unwrap(); // 1s model load
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0, 10)));

    let r = tokio::time::timeout(Duration::from_millis(200), sup.ensure_ready()).await;
    assert!(
        r.is_err(),
        "the caller must time out while the child is still loading"
    );

    let s = wait_state(&sup, TtsState::Ready, Duration::from_secs(5)).await;
    assert_eq!(
        s,
        TtsState::Ready,
        "the spawned leader must finish the start after the caller is dropped"
    );
    sup.shutdown().await;
    cleanup_markers(port);
}

// ── R2: a failed start is shared by every waiter ────────────────────────
// Mutation: let each waiter start its own generation -> RED.
#[tokio::test]
#[serial]
async fn r2_failed_start_shared_by_waiters() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("exit", port), "").unwrap(); // child exits at once
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0, 10)));
    let before = metric("oxwhisper_tts_child_starts_total");

    let mut set = tokio::task::JoinSet::new();
    for _ in 0..10 {
        let s = Arc::clone(&sup);
        set.spawn(async move { s.ensure_ready().await });
    }
    let mut errs = 0;
    while let Some(r) = set.join_next().await {
        let res = r.expect("ensure_ready task panicked");
        match res {
            Err(TtsError::StartupExit(_)) => errs += 1,
            other => panic!("every waiter must see the shared start error, got {other:?}"),
        }
    }
    assert_eq!(errs, 10, "all 10 waiters get the generation's error");
    assert_eq!(
        metric("oxwhisper_tts_child_starts_total") - before,
        1.0,
        "10 waiters on one failed generation must cause exactly ONE spawn"
    );
    cleanup_markers(port);
}

// ── R3: health probe timeout ────────────────────────────────────────────
// Mutation: remove the per-probe timeout -> RED (the probe hangs forever).
#[tokio::test]
#[serial]
async fn r3_hanging_listener_times_out() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("hang", port), "").unwrap(); // accepts, never answers
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0, 3)));
    let before = metric("oxwhisper_tts_child_stops_total{reason=\"startup_timeout\"}");

    let res = tokio::time::timeout(Duration::from_secs(6), sup.ensure_ready())
        .await
        .expect("ensure_ready hung — the per-probe timeout is missing");
    assert!(
        matches!(res, Err(TtsError::StartupTimeout(3))),
        "expected a startup-timeout error, got {res:?}"
    );
    assert!(
        wait_metric(
            "oxwhisper_tts_child_stops_total{reason=\"startup_timeout\"}",
            before + 1.0,
            Duration::from_secs(3),
        )
        .await,
        "stops_total{{reason=\"startup_timeout\"}} must increase"
    );

    let res2 = tokio::time::timeout(Duration::from_secs(8), sup.ensure_ready())
        .await
        .expect("a second ensure_ready must not hang");
    assert!(
        res2.is_err(),
        "the second start must fail too, got {res2:?}"
    );
    sup.shutdown().await;
    cleanup_markers(port);
}

// ── R4: guard release re-opens the idle window ──────────────────────────
// Mutation: delete `inner.in_flight = inner.in_flight.saturating_sub(1)` in
// TtsGuard::drop -> RED.
#[tokio::test]
#[serial]
async fn r4_guard_drop_allows_idle_stop() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 1, 10)));
    let _idle = sup.spawn_idle_loop(Duration::from_millis(100));

    let (_url, guard) = sup.ensure_ready().await.expect("fake child should start");
    drop(guard);

    let s = wait_state(&sup, TtsState::Stopped, Duration::from_secs(5)).await;
    assert_eq!(
        s,
        TtsState::Stopped,
        "dropping the returned guard must re-open the idle window"
    );
    sup.shutdown().await;
    cleanup_markers(port);
}

// ── R5: idle stop fires at the configured threshold, not earlier ────────
// Mutation: compare `elapsed() >= Duration::ZERO` in idle_tick -> RED.
#[tokio::test]
#[serial]
async fn r5_idle_stop_exact_threshold() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 3, 10)));
    let _idle = sup.spawn_idle_loop(Duration::from_millis(200));

    let (_url, guard) = sup.ensure_ready().await.expect("fake child should start");
    drop(guard); // the idle clock starts here
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(
        sup.state(),
        TtsState::Ready,
        "child must still be Ready 1.5s into a 3s idle threshold"
    );
    let s = wait_state(&sup, TtsState::Stopped, Duration::from_secs(6)).await;
    assert_eq!(
        s,
        TtsState::Stopped,
        "child must be stopped shortly after the 3s idle threshold"
    );
    sup.shutdown().await;
    cleanup_markers(port);
}

// ── R6: port collision ──────────────────────────────────────────────────
// Mutation: skip the pre-spawn bind check -> RED.
#[tokio::test]
#[serial]
async fn r6_port_busy_fails_start() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap(); // stays up once bound
    // A foreign process owns the port and answers /health.
    let mut foreign = std::process::Command::new("python3")
        .arg(fake_bin())
        .arg("--port")
        .arg(port.to_string())
        .spawn()
        .expect("spawn the foreign listener");
    let deadline = Instant::now() + Duration::from_secs(3);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            Instant::now() < deadline,
            "foreign listener never bound {port}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 0, 10)));
    let before = metric("oxwhisper_tts_child_start_failures_total{reason=\"port_busy\"}");
    let res = tokio::time::timeout(Duration::from_secs(5), sup.ensure_ready())
        .await
        .expect("ensure_ready hung on a busy port");
    assert!(
        matches!(res, Err(TtsError::PortBusy(p)) if p == port),
        "expected a port-busy error, got {res:?}"
    );
    assert!(
        wait_metric(
            "oxwhisper_tts_child_start_failures_total{reason=\"port_busy\"}",
            before + 1.0,
            Duration::from_secs(2),
        )
        .await,
        "start_failures_total{{reason=\"port_busy\"}} must increase"
    );
    assert_ne!(
        sup.state(),
        TtsState::Ready,
        "a foreign listener's /health must never commit Ready"
    );

    let _ = foreign.kill();
    let _ = foreign.wait();
    sup.shutdown().await;
    cleanup_markers(port);
}

// ── R7: stop delivers SIGTERM before SIGKILL ────────────────────────────
// Mutation: go straight to `start_kill` in the monitor stop path -> RED.
#[tokio::test]
#[serial]
async fn r7_stop_sends_sigterm() {
    require_python3();
    let port = free_port();
    cleanup_markers(port);
    std::fs::write(marker("crash", port), "").unwrap();
    let sup = Arc::new(TtsSupervisor::new(test_config(fake_bin(), port, 1, 10)));
    let _idle = sup.spawn_idle_loop(Duration::from_millis(100));

    let (_url, guard) = sup.ensure_ready().await.expect("fake child should start");
    drop(guard); // let the idle loop stop the child

    let sigterm = marker("sigterm", port);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !sigterm.exists() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        sigterm.exists(),
        "a stop must deliver SIGTERM (marker {sigterm:?}) before any SIGKILL"
    );
    sup.shutdown().await;
    cleanup_markers(port);
}
