use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::response::Response;

use crate::chunking::sanitize_utf8;
use crate::handlers::AppState;
use crate::transcribe::{TranscribeError, maybe_punctuate, split_audio_chunks, transcribe_routed};
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
    ws.on_upgrade(move |socket| handle_ws(socket, state, params))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<AppState>, params: WsParams) {
    let start = std::time::Instant::now();
    metrics::gauge!(crate::metrics::names::WS_ACTIVE).increment(1.0);
    let _conn = WsConnGuard { start };

    let request_id = uuid::Uuid::new_v4().to_string();
    let model = if params.language == "ru" {
        "gigaam"
    } else {
        "moonshine-v2"
    };

    // Send metadata
    let meta = ServerMessage::Metadata {
        request_id: request_id.clone(),
        model: model.to_string(),
        channels: 1,
    };
    if send_msg(&mut socket, &meta).await.is_err() {
        return;
    }

    let mut session = WsSession::new(params.sample_rate, state.config.ws_max_buffer_samples());

    loop {
        let msg = match socket.recv().await {
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
                    break;
                }

                // VAD check if enabled
                if params.vad {
                    let (vad_msgs, speech_final) =
                        session.run_vad_check(&state.models, &state.config);
                    for m in vad_msgs {
                        if send_msg(&mut socket, &m).await.is_err() {
                            return;
                        }
                    }
                    if speech_final {
                        if let Some(msg) = do_transcribe(&state, &mut session, &params, false).await
                        {
                            if send_msg(&mut socket, &msg).await.is_err() {
                                return;
                            }
                        }
                        continue;
                    }
                }

                // Interim results
                if params.interim_results && session.should_emit_interim(INTERIM_INTERVAL_S) {
                    if let Some(msg) = do_transcribe_interim(&state, &mut session, &params).await {
                        if send_msg(&mut socket, &msg).await.is_err() {
                            return;
                        }
                    }
                    session.mark_interim();
                }
            }
            Message::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                Ok(ClientMessage::Finalize) => {
                    if let Some(msg) = do_transcribe(&state, &mut session, &params, true).await {
                        if send_msg(&mut socket, &msg).await.is_err() {
                            return;
                        }
                    }
                }
                Ok(ClientMessage::CloseStream) => {
                    if let Some(msg) = do_transcribe(&state, &mut session, &params, true).await {
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

/// Transcribe the buffer and return a final Results message.
async fn do_transcribe(
    state: &Arc<AppState>,
    session: &mut WsSession,
    params: &WsParams,
    from_finalize: bool,
) -> Option<ServerMessage> {
    let samples = session.take_buffer();
    if samples.is_empty() {
        return None;
    }
    match transcribe_buffer(state, samples, &params.language, params.punctuate).await {
        Ok(Some((text, words))) => Some(session.store_final(text, words, from_finalize)),
        Ok(None) => None,
        Err(e) => Some(buffer_error(&e)),
    }
}

/// Transcribe a copy of the buffer for interim results (non-destructive peek).
async fn do_transcribe_interim(
    state: &Arc<AppState>,
    session: &mut WsSession,
    params: &WsParams,
) -> Option<ServerMessage> {
    let samples = session.peek_buffer();
    if samples.is_empty() {
        return None;
    }
    match transcribe_buffer(state, samples, &params.language, params.punctuate).await {
        Ok(Some((text, words))) => Some(session.interim_result(text, words)),
        Ok(None) => None,
        Err(e) => Some(buffer_error(&e)),
    }
}

/// A decode that failed (no free recognizer, a failed reload) is told to the
/// client: the audio was taken from the buffer, so silence would lose it.
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
    language: &str,
    punctuate: bool,
) -> Result<Option<(String, Vec<WordTimestamp>)>, TranscribeError> {
    let models = state.clone();
    let lang = language.to_string();
    let punct = punctuate;

    tokio::task::spawn_blocking(move || {
        let config = &models.config;
        let engine = models.models.route(&lang, &config.parakeet_langs);
        // The session buffer is not capped, so decode it in bounded chunks
        // (and bounded batches) like the batch endpoint — never as one call,
        // which on Parakeet's full-attention encoder grows memory with the
        // square of the stream length and pins the slot.
        let chunks = split_audio_chunks(samples, config.max_chunk_s * 16000);
        let offsets = compute_chunk_offsets(&chunks, 16000);
        let (engine, texts, words) = transcribe_routed(
            &models.models,
            config,
            engine,
            &lang,
            &chunks,
            &offsets,
            config.hallucination_threshold,
        )?;
        let text = sanitize_utf8(texts.join(" ").trim());
        if text.is_empty() {
            return Ok(None);
        }
        let text = if punct {
            maybe_punctuate(&models.models, &text, &lang, engine, Some(true))
        } else {
            text
        };
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
