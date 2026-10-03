use std::sync::Arc;

use axum::extract::ws::{CloseFrame, Message, WebSocket, close_code};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::response::{IntoResponse, Response};

use crate::chunking::sanitize_utf8;
use crate::handlers::AppState;
use crate::language;
use crate::models::PARAKEET_MODEL_NAME;
use crate::transcribe::{TranscribeError, plan_chunks, transcribe_chunks};
use crate::words::{WordTimestamp, compute_chunk_offsets};
use crate::ws_session::WsSession;
use crate::ws_types::{ClientMessage, ServerMessage, WsParams};

const INTERIM_INTERVAL_S: f32 = 2.0;

struct WsConnGuard {
    start: std::time::Instant,
}
impl Drop for WsConnGuard {
    fn drop(&mut self) {
        metrics::gauge!(crate::metrics::names::WS_ACTIVE).decrement(1.0);
        metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => "ws_listen", "status" => "ok")
            .increment(1);
        metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => "ws_listen")
            .record(self.start.elapsed().as_secs_f64());
    }
}

pub async fn ws_listen(
    State(state): State<Arc<AppState>>,
    Query(params): Query<WsParams>,
    ws: WebSocketUpgrade,
) -> Response {
    // Refused before the upgrade, so the client sees a 400 and not a socket
    // that opens and then produces text in the wrong language.
    if let Err(e) = language::resolve(&params.language) {
        return e.into_response();
    }
    // Decoding assumes 16 kHz and nothing resamples: audio sent at another
    // rate would be transcribed as garbage, so it is refused up front.
    if params.sample_rate != 16000 {
        let body = serde_json::json!({
            "error": {
                "message": format!(
                    "sample_rate {} is not supported; send 16000 Hz mono audio",
                    params.sample_rate
                ),
                "type": "invalid_request_error",
                "param": "sample_rate",
                "code": "invalid_sample_rate",
            }
        });
        return (axum::http::StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
    }
    // Counted before the upgrade completes, so shutdown never misses a session.
    let session = state.shutdown.session();
    ws.on_upgrade(move |socket| handle_ws(socket, state, params, session))
}

async fn handle_ws(
    mut socket: WebSocket,
    state: Arc<AppState>,
    params: WsParams,
    _session: crate::server::SessionGuard,
) {
    let start = std::time::Instant::now();
    metrics::gauge!(crate::metrics::names::WS_ACTIVE).increment(1.0);
    let _conn = WsConnGuard { start };

    let request_id = uuid::Uuid::new_v4().to_string();

    // Send metadata
    let meta = ServerMessage::Metadata {
        request_id: request_id.clone(),
        model: PARAKEET_MODEL_NAME.to_string(),
        channels: 1,
    };
    if send_msg(&mut socket, &meta).await.is_err() {
        return;
    }

    let mut session = WsSession::new(params.sample_rate, state.config.ws_max_buffer_samples());
    let mut shutdown = state.shutdown.clone();

    loop {
        // Upgraded connections are not drained by the HTTP server's graceful
        // shutdown: a session left open would pin it until the drain ran out.
        // Tell the client to reconnect instead (1001), at a message boundary.
        let received = tokio::select! {
            biased;
            _ = shutdown.fired() => {
                close_with(&mut socket, close_code::AWAY, "server shutting down").await;
                break;
            }
            received = socket.recv() => received,
        };
        let msg = match received {
            Some(Ok(msg)) => msg,
            Some(Err(e)) => {
                tracing::debug!("WS recv error: {}", e);
                break;
            }
            None => break,
        };

        match msg {
            Message::Binary(data) => {
                if let Err(full) = session.push_audio(&data, &params.encoding) {
                    metrics::counter!(crate::metrics::names::WS_BUFFER_LIMIT).increment(1);
                    let _ = send_msg(
                        &mut socket,
                        &ServerMessage::Error {
                            message: full.to_string(),
                        },
                    )
                    .await;
                    close_with(&mut socket, close_code::SIZE, "audio buffer limit reached").await;
                    break;
                }

                // VAD check if enabled
                if params.vad {
                    // A VAD pass over the whole buffer runs per frame: on the
                    // blocking pool, not on a tokio worker.
                    let st = state.clone();
                    let checked = tokio::task::spawn_blocking(move || {
                        let out = session.run_vad_check(&st.models, &st.config);
                        (session, out)
                    })
                    .await;
                    let (returned, (vad_msgs, speech_final)) = match checked {
                        Ok(done) => done,
                        Err(e) => {
                            // The session state went down with the task: tell
                            // the client, and close with 1011.
                            tracing::error!("WS VAD check failed: {e}");
                            let _ = send_msg(
                                &mut socket,
                                &ServerMessage::Error {
                                    message: "internal error".into(),
                                },
                            )
                            .await;
                            close_with(&mut socket, close_code::ERROR, "internal error").await;
                            break;
                        }
                    };
                    session = returned;
                    for m in vad_msgs {
                        if send_msg(&mut socket, &m).await.is_err() {
                            return;
                        }
                    }
                    if speech_final {
                        if let Some(msg) = do_transcribe(&state, &mut session, false).await {
                            if send_msg(&mut socket, &msg).await.is_err() {
                                return;
                            }
                        }
                        continue;
                    }
                }

                // Interim results
                if params.interim_results && session.should_emit_interim(INTERIM_INTERVAL_S) {
                    if let Some(msg) = do_transcribe_interim(&state, &mut session).await {
                        if send_msg(&mut socket, &msg).await.is_err() {
                            return;
                        }
                    }
                    session.mark_interim();
                }
            }
            Message::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                Ok(ClientMessage::Finalize) => {
                    if let Some(msg) = do_transcribe(&state, &mut session, true).await {
                        if send_msg(&mut socket, &msg).await.is_err() {
                            return;
                        }
                    }
                }
                Ok(ClientMessage::CloseStream) => {
                    if let Some(msg) = do_transcribe(&state, &mut session, true).await {
                        let _ = send_msg(&mut socket, &msg).await;
                    }
                    let _ = send_msg(&mut socket, &ServerMessage::CloseStream).await;
                    break;
                }
                Ok(ClientMessage::KeepAlive) => {}
                Err(e) => {
                    tracing::debug!("WS unknown text message: {}", e);
                }
            },
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// Closes the socket properly: a Close frame with `code` (1009 after a buffer
/// overrun, 1001 on shutdown), then a short bounded wait for the client's reply while
/// reading and discarding what it still sends. Dropping the socket with
/// unread inbound frames makes the kernel send a reset, and the client can
/// lose the Error frame that explains the close.
async fn close_with(socket: &mut WebSocket, code: u16, reason: &'static str) {
    let frame = CloseFrame {
        code,
        reason: reason.into(),
    };
    if socket.send(Message::Close(Some(frame))).await.is_err() {
        return;
    }
    let _ = tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while let Some(Ok(msg)) = socket.recv().await {
            if matches!(msg, Message::Close(_)) {
                break;
            }
        }
    })
    .await;
}

/// Transcribe the buffer and return a final Results message.
async fn do_transcribe(
    state: &Arc<AppState>,
    session: &mut WsSession,
    from_finalize: bool,
) -> Option<ServerMessage> {
    let samples = session.take_buffer();
    if samples.is_empty() {
        return None;
    }
    match transcribe_buffer(state, samples).await {
        Ok(Some((text, words))) => Some(session.store_final(text, words, from_finalize)),
        Ok(None) => None,
        Err(e) => Some(buffer_error(&e)),
    }
}

/// Transcribe a copy of the buffer for interim results (non-destructive peek).
async fn do_transcribe_interim(
    state: &Arc<AppState>,
    session: &mut WsSession,
) -> Option<ServerMessage> {
    let samples = session.peek_buffer();
    if samples.is_empty() {
        return None;
    }
    match transcribe_buffer(state, samples).await {
        Ok(Some((text, words))) => Some(session.interim_result(text, words)),
        Ok(None) => None,
        // An interim decode works on a peeked copy, so a failure loses nothing,
        // and the next interval tries again: with a busy slot an Error every
        // 2 s would be noise. Count it and skip.
        Err(e) => {
            tracing::debug!("WS interim decode skipped: {e}");
            metrics::counter!(crate::metrics::names::WS_INTERIM_SKIPPED).increment(1);
            None
        }
    }
}

/// A *final* decode that failed (no free recognizer, a failed reload) is told
/// to the client: its audio was taken out of the buffer, so silence would lose
/// it. Interim decodes, which work on a copy, skip instead.
fn buffer_error(e: &TranscribeError) -> ServerMessage {
    tracing::warn!("WS transcription failed: {e}");
    ServerMessage::Error {
        message: e.to_string(),
    }
}

/// Run transcription on samples via spawn_blocking. `Ok(None)` is audio that
/// decoded to no text; `Err` is a decode that could not run.
async fn transcribe_buffer(
    state: &Arc<AppState>,
    samples: Vec<f32>,
) -> Result<Option<(String, Vec<WordTimestamp>)>, TranscribeError> {
    let models = state.clone();

    tokio::task::spawn_blocking(move || {
        let config = &models.config;
        // Decode in bounded chunks (and bounded batches) like the batch
        // endpoint — never as one call, which on Parakeet's full-attention
        // encoder grows memory with the square of the buffer length and pins
        // the slot. The buffer itself is capped by WS_MAX_BUFFER_S.
        let chunks = plan_chunks(samples, config);
        let offsets = compute_chunk_offsets(&chunks, 16000);
        let (texts, words) = transcribe_chunks(
            &models.models,
            &chunks,
            &offsets,
            config.hallucination_threshold,
        )?;
        let text = sanitize_utf8(texts.join(" ").trim());
        if text.is_empty() {
            return Ok(None);
        }
        Ok(Some((text, words)))
    })
    .await
    .unwrap_or(Err(TranscribeError::NoRecognizer))
}

async fn send_msg(socket: &mut WebSocket, msg: &ServerMessage) -> Result<(), ()> {
    let json = serde_json::to_string(msg).map_err(|_| ())?;
    socket
        .send(Message::Text(json.into()))
        .await
        .map_err(|_| ())
}

#[cfg(test)]
#[path = "ws_tests.rs"]
mod tests;
