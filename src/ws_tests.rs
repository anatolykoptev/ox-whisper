//! The WebSocket path against a real socket: a client that overruns the buffer
//! cap, or whose audio cannot be decoded, must be told — not dropped silently.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message as ClientMsg;

use crate::config::Config;
use crate::handlers::AppState;
use crate::models::Models;

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Serves `/v1/listen` over an empty model set: no recognizer is loaded, so
/// every decode fails — which is what the tests need to see reported.
async fn serve(max_buffer_s: &'static str) -> std::net::SocketAddr {
    let config =
        Config::from_lookup(&move |k| (k == "WS_MAX_BUFFER_S").then(|| max_buffer_s.into()));
    let state = Arc::new(AppState {
        models: Models::empty(),
        config,
        shutdown: crate::server::ShutdownSignal::inert(),
    });
    let app = Router::new()
        .route("/v1/listen", get(super::ws_listen))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    addr
}

async fn connect(max_buffer_s: &'static str) -> Client {
    connect_with(max_buffer_s, "vad=false").await
}

async fn connect_with(max_buffer_s: &'static str, query: &str) -> Client {
    let addr = serve(max_buffer_s).await;
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/listen?{query}"))
        .await
        .unwrap();
    let meta = next_json(&mut ws).await.expect("metadata first");
    assert_eq!(meta["type"], "Metadata");
    ws
}

/// The next text frame as JSON; `None` once the connection is closed.
async fn next_json(ws: &mut Client) -> Option<serde_json::Value> {
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("no frame within 5 s");
        match frame {
            Some(Ok(ClientMsg::Text(t))) => return Some(serde_json::from_str(&t).unwrap()),
            Some(Ok(ClientMsg::Close(_))) | None | Some(Err(_)) => return None,
            Some(Ok(_)) => {}
        }
    }
}

/// What the next frame was, telling a clean Close (with its code) apart from
/// the connection simply disappearing (a reset or a dropped socket).
#[derive(Debug)]
enum Frame {
    Json(serde_json::Value),
    Close(Option<u16>),
    Gone,
}

async fn next_frame(ws: &mut Client) -> Frame {
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("no frame within 5 s");
        match frame {
            Some(Ok(ClientMsg::Text(t))) => return Frame::Json(serde_json::from_str(&t).unwrap()),
            Some(Ok(ClientMsg::Close(f))) => return Frame::Close(f.map(|f| u16::from(f.code))),
            None | Some(Err(_)) => return Frame::Gone,
            Some(Ok(_)) => {}
        }
    }
}

fn pcm_seconds(s: usize) -> ClientMsg {
    ClientMsg::Binary(vec![0u8; s * 16000 * 2].into())
}

#[tokio::test]
async fn a_client_that_overruns_the_buffer_gets_an_error_then_a_1009_close() {
    let mut ws = connect("1").await;
    ws.send(pcm_seconds(2)).await.unwrap();

    match next_frame(&mut ws).await {
        Frame::Json(msg) => {
            assert_eq!(msg["type"], "Error");
            assert!(
                msg["message"]
                    .as_str()
                    .unwrap()
                    .contains("audio buffer limit"),
                "{msg}"
            );
        }
        other => panic!("want the Error frame first, got {other:?}"),
    }
    // A Close frame with 1009 (message too big), not a bare disconnect: a
    // reset can swallow the Error frame above.
    match next_frame(&mut ws).await {
        Frame::Close(Some(1009)) => {}
        other => panic!("want Close(1009), got {other:?}"),
    }
}

/// An interim decode works on a copy of the buffer, so when it cannot run
/// nothing is lost and nothing is sent; the final decode still reports.
#[tokio::test]
async fn a_failed_interim_decode_is_skipped_but_a_failed_final_one_is_reported() {
    let mut ws = connect_with("30", "vad=false&interim_results=true").await;
    ws.send(pcm_seconds(3)).await.unwrap(); // past the 2 s interim interval

    // No Error arrives for the interim attempt.
    assert!(
        tokio::time::timeout(Duration::from_millis(800), ws.next())
            .await
            .is_err(),
        "an interim failure must not reach the client"
    );
    ws.send(ClientMsg::Text(r#"{"type":"Finalize"}"#.into()))
        .await
        .unwrap();
    match next_frame(&mut ws).await {
        Frame::Json(msg) => assert_eq!(msg["type"], "Error", "{msg}"),
        other => panic!("want the final Error, got {other:?}"),
    }
}

/// The audio is taken out of the buffer before it is decoded, so a decode that
/// cannot run (here: no model) has to reach the client, or that audio is just
/// gone. Also the control for the test above: this audio fits the cap.
#[tokio::test]
async fn a_failed_decode_is_reported_to_the_client() {
    let mut ws = connect("1").await;
    ws.send(ClientMsg::Binary(vec![0u8; 16000].into()))
        .await
        .unwrap();
    ws.send(ClientMsg::Text(r#"{"type":"Finalize"}"#.into()))
        .await
        .unwrap();

    let msg = next_json(&mut ws).await.expect("an error frame");
    assert_eq!(msg["type"], "Error");
    let text = msg["message"].as_str().unwrap();
    assert!(!text.contains("buffer limit"), "{msg}");
    assert!(text.contains("no recognizer available"), "{msg}");
}

/// An unsupported language is refused before the upgrade; a supported one,
/// and no language at all, upgrade (the positive controls).
#[tokio::test]
async fn an_unsupported_language_is_refused_before_the_upgrade() {
    let addr = serve("1").await;
    let url = |q: &str| format!("ws://{addr}/v1/listen?{q}");

    let err = tokio_tungstenite::connect_async(url("language=zh"))
        .await
        .expect_err("zh must not upgrade");
    match err {
        tokio_tungstenite::tungstenite::Error::Http(res) => assert_eq!(res.status(), 400),
        other => panic!("want an HTTP 400, got {other:?}"),
    }
    for ok in ["language=ru", "language=en-US", "vad=false"] {
        let (mut ws, _) = tokio_tungstenite::connect_async(url(ok)).await.unwrap();
        assert_eq!(
            next_json(&mut ws).await.unwrap()["type"],
            "Metadata",
            "{ok}"
        );
    }
}

/// Decoding assumes 16 kHz and nothing resamples, so any other rate is refused
/// before the upgrade; 16000, the default, upgrades (the positive control).
#[tokio::test]
async fn a_sample_rate_other_than_16k_is_refused_before_the_upgrade() {
    let addr = serve("1").await;
    let url = |q: &str| format!("ws://{addr}/v1/listen?{q}");

    for rate in ["8000", "44100", "0"] {
        let err = tokio_tungstenite::connect_async(url(&format!("sample_rate={rate}")))
            .await
            .expect_err("must not upgrade");
        match err {
            tokio_tungstenite::tungstenite::Error::Http(res) => {
                assert_eq!(res.status(), 400, "{rate}")
            }
            other => panic!("want an HTTP 400, got {other:?}"),
        }
    }
    let (mut ws, _) = tokio_tungstenite::connect_async(url("sample_rate=16000"))
        .await
        .unwrap();
    assert_eq!(next_json(&mut ws).await.unwrap()["type"], "Metadata");
}

/// The per-frame VAD pass covers the whole buffer: run on a tokio worker it
/// stalls every other task for as long as it takes (or waits for the shared
/// detector). A heartbeat on the same single-threaded runtime must keep
/// ticking while a pass waits on a detector that another thread holds.
#[tokio::test(flavor = "current_thread")]
async fn the_vad_pass_does_not_stall_the_runtime() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut config = Config::from_lookup(&|k| (k == "WS_MAX_BUFFER_S").then(|| "120".into()));
    config.vad_model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vad/silero_vad.onnx")
        .to_string_lossy()
        .into_owned();
    config.max_audio_duration_s = 120.0;
    let mut models = Models::empty();
    models.vad = crate::models::load_vad(&config);
    assert!(models.vad.is_some(), "VAD fixture loads");
    let state = Arc::new(AppState {
        models,
        config,
        shutdown: crate::server::ShutdownSignal::inert(),
    });
    let app = Router::new()
        .route("/v1/listen", get(super::ws_listen))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/listen"))
        .await
        .unwrap();
    assert_eq!(next_json(&mut ws).await.unwrap()["type"], "Metadata");

    let ticks = Arc::new(AtomicUsize::new(0));
    let t = ticks.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(1)).await;
            t.fetch_add(1, Ordering::SeqCst);
        }
    });
    tokio::task::yield_now().await;

    // Another thread holds the detector for 400 ms: the pass has to wait.
    let held = state.clone();
    let (locked_tx, locked_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _detector = crate::vad::lock_vad(held.models.vad.as_ref().unwrap());
        locked_tx.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(400));
    });
    locked_rx.recv().unwrap();

    // The frame is handled in order, so the answer to Finalize arrives only
    // after the VAD pass over the audio before it.
    ws.send(pcm_seconds(2)).await.unwrap();
    ws.send(ClientMsg::Text(r#"{"type":"Finalize"}"#.into()))
        .await
        .unwrap();
    let before = ticks.load(Ordering::SeqCst);
    let _ = next_json(&mut ws).await;
    let during = ticks.load(Ordering::SeqCst) - before;
    assert!(
        during >= 50,
        "the runtime stalled during the VAD pass: {during} ticks"
    );
}
