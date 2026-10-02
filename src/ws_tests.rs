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
async fn connect(max_buffer_s: &'static str) -> Client {
    let config =
        Config::from_lookup(&move |k| (k == "WS_MAX_BUFFER_S").then(|| max_buffer_s.into()));
    let state = Arc::new(AppState {
        models: Models::empty(),
        config,
    });
    let app = Router::new()
        .route("/v1/listen", get(super::ws_listen))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/listen?vad=false"))
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

fn pcm_seconds(s: usize) -> ClientMsg {
    ClientMsg::Binary(vec![0u8; s * 16000 * 2].into())
}

#[tokio::test]
async fn a_client_that_overruns_the_buffer_gets_an_error_and_is_closed() {
    let mut ws = connect("1").await;
    ws.send(pcm_seconds(2)).await.unwrap();

    let msg = next_json(&mut ws).await.expect("an error frame");
    assert_eq!(msg["type"], "Error");
    assert!(
        msg["message"]
            .as_str()
            .unwrap()
            .contains("audio buffer limit"),
        "{msg}"
    );
    assert!(next_json(&mut ws).await.is_none(), "connection stays open");
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
    assert!(text.contains("not supported or model not loaded"), "{msg}");
}
