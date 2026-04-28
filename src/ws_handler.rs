use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::response::Response;

use crate::handlers::AppState;
use crate::transcribe::{compression_ratio, maybe_punctuate};
use crate::words::{WordTimestamp, estimate_words_from_text, extract_words_with_confidence};
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
    let model = if params.language == "ru" { "gigaam" } else { "moonshine-v2" };

    // Send metadata
    let meta = ServerMessage::Metadata {
        request_id: request_id.clone(),
        model: model.to_string(),
        channels: 1,
    };
    if send_msg(&mut socket, &meta).await.is_err() {
        return;
    }

    let mut session = WsSession::new(params.sample_rate);

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
                session.push_audio(&data, &params.encoding);

                // VAD check if enabled
                if params.vad {
                    let (vad_msgs, speech_final) =
                        session.run_vad_check(&state.models, &state.config);
                    for m in vad_msgs {
                        if send_msg(&mut socket, &m).await.is_err() { return; }
                    }
                    if speech_final {
                        if let Some(msg) = do_transcribe(&state, &mut session, &params, false).await {
                            if send_msg(&mut socket, &msg).await.is_err() { return; }
                        }
                        continue;
                    }
                }

                // Interim results
                if params.interim_results && session.should_emit_interim(INTERIM_INTERVAL_S) {
                    if let Some(msg) = do_transcribe_interim(&state, &mut session, &params).await {
                        if send_msg(&mut socket, &msg).await.is_err() { return; }
                    }
                    session.mark_interim();
                }
            }
            Message::Text(text) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Finalize) => {
                        if let Some(msg) = do_transcribe(&state, &mut session, &params, true).await {
                            if send_msg(&mut socket, &msg).await.is_err() { return; }
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
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// Transcribe the buffer and return a final Results message.
async fn do_transcribe(
    state: &Arc<AppState>, session: &mut WsSession, params: &WsParams, from_finalize: bool,
) -> Option<ServerMessage> {
    let samples = session.take_buffer();
    if samples.is_empty() { return None; }
    let (text, words) = transcribe_buffer(state, samples, &params.language, params.punctuate).await?;
    Some(session.store_final(text, words, from_finalize))
}

/// Transcribe a copy of the buffer for interim results (non-destructive peek).
async fn do_transcribe_interim(
    state: &Arc<AppState>, session: &mut WsSession, params: &WsParams,
) -> Option<ServerMessage> {
    let samples = session.peek_buffer();
    if samples.is_empty() { return None; }
    let (text, words) = transcribe_buffer(state, samples, &params.language, params.punctuate).await?;
    Some(session.interim_result(text, words))
}

/// Run transcription on samples via spawn_blocking.
async fn transcribe_buffer(
    state: &Arc<AppState>, samples: Vec<f32>, language: &str, punctuate: bool,
) -> Option<(String, Vec<WordTimestamp>)> {
    let models = state.clone();
    let lang = language.to_string();
    let punct = punctuate;

    tokio::task::spawn_blocking(move || {
        let threshold = models.config.hallucination_threshold;
        let (text, words) = if lang == "ru" {
            transcribe_with_pool_ru(&models.models, &samples, threshold)?
        } else {
            transcribe_with_pool_en(&models.models, &samples, &lang, threshold)?
        };
        let text = if punct {
            maybe_punctuate(&models.models, &text, &lang, Some(true))
        } else {
            text
        };
        Some((text, words))
    })
    .await
    .ok()?
}

fn transcribe_with_pool_en(
    models: &crate::models::Models, samples: &[f32], _language: &str, threshold: f64,
) -> Option<(String, Vec<WordTimestamp>)> {
    let pool = models.en.as_ref()?;
    let mut rec = pool.acquire()?;
    metrics::gauge!(crate::metrics::names::POOL_BUSY, "lang" => "en").increment(1.0);
    struct EnBusyGuard;
    impl Drop for EnBusyGuard {
        fn drop(&mut self) {
            metrics::gauge!(crate::metrics::names::POOL_BUSY, "lang" => "en").decrement(1.0);
        }
    }
    let _busy = EnBusyGuard;
    let result = rec.transcribe(16000, samples);
    let text = result.text.trim().to_string();
    if text.is_empty() || compression_ratio(&text) > threshold { return None; }
    let mut words = Vec::new();
    extract_words_with_confidence(&result.tokens, &result.timestamps, &result.log_probs, 0.0, &mut words);
    if words.is_empty() && !text.is_empty() {
        words = estimate_words_from_text(&text, samples.len() as f32 / 16000.0, 0.0);
    }
    Some((text, words))
}

fn transcribe_with_pool_ru(
    models: &crate::models::Models, samples: &[f32], threshold: f64,
) -> Option<(String, Vec<WordTimestamp>)> {
    let pool = models.ru.as_ref()?;
    let mut rec = pool.acquire()?;
    metrics::gauge!(crate::metrics::names::POOL_BUSY, "lang" => "ru").increment(1.0);
    struct RuBusyGuard;
    impl Drop for RuBusyGuard {
        fn drop(&mut self) {
            metrics::gauge!(crate::metrics::names::POOL_BUSY, "lang" => "ru").decrement(1.0);
        }
    }
    let _busy = RuBusyGuard;
    let result = rec.transcribe(16000, samples);
    let text = result.text.trim().to_string();
    if text.is_empty() || compression_ratio(&text) > threshold { return None; }
    let mut words = Vec::new();
    extract_words_with_confidence(&result.tokens, &result.timestamps, &result.log_probs, 0.0, &mut words);
    if words.is_empty() && !text.is_empty() {
        words = estimate_words_from_text(&text, samples.len() as f32 / 16000.0, 0.0);
    }
    Some((text, words))
}

async fn send_msg(socket: &mut WebSocket, msg: &ServerMessage) -> Result<(), ()> {
    let json = serde_json::to_string(msg).map_err(|_| ())?;
    socket.send(Message::Text(json.into())).await.map_err(|_| ())
}
