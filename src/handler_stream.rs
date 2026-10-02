use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::wrappers::ReceiverStream;

use crate::handlers::{AppState, observe, parse_upload};
use crate::language;
use crate::streaming;

pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let endpoint = "transcribe_stream";
    let start = std::time::Instant::now();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    let upload = match parse_upload(&mut multipart, &state.config.upload_dir).await {
        Ok(u) => u,
        Err(msg) => {
            metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => "err")
                .increment(1);
            metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
                .record(start.elapsed().as_secs_f64());
            let data = serde_json::json!({"type": "error", "message": msg});
            let _ = tx.send(Ok(Event::default().data(data.to_string()))).await;
            drop(tx);
            return Sse::new(ReceiverStream::new(rx)).into_response();
        }
    };

    // The stream has not started, so an unsupported language can still be a
    // plain 400 instead of an error event on a 200.
    if let Err(e) = language::resolve(&upload.language) {
        observe(endpoint, false, start);
        return e.into_response();
    }

    // Owned by the blocking job below: removed when it ends, on any path.
    let file = upload.file;
    let vad = upload.vad;

    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<streaming::StreamEvent>(32);

    // Forward chunk events to SSE
    let tx_fwd = tx.clone();
    let forwarder = tokio::spawn(async move {
        while let Some(evt) = chunk_rx.recv().await {
            let data = serde_json::json!({
                "type": "chunk",
                "index": evt.chunk_index,
                "total": evt.total_chunks,
                "text": evt.text,
            });
            let _ = tx_fwd
                .send(Ok(Event::default().data(data.to_string())))
                .await;
        }
    });

    // Run transcription, then send final event
    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            streaming::transcribe_streaming(
                &state.models,
                &state.config,
                file.path(),
                vad,
                chunk_tx,
            )
        })
        .await;

        let _ = forwarder.await;

        match result {
            Ok(Ok(r)) => {
                metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => "ok")
                    .increment(1);
                metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
                    .record(start.elapsed().as_secs_f64());
                let data = serde_json::json!({
                    "type": "done",
                    "text": r.text,
                    "duration_ms": r.duration_ms,
                    "speech_ms": r.speech_ms,
                });
                let _ = tx.send(Ok(Event::default().data(data.to_string()))).await;
            }
            Ok(Err(e)) => {
                metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => "err")
                    .increment(1);
                metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
                    .record(start.elapsed().as_secs_f64());
                let data = serde_json::json!({"type": "error", "message": e.to_string()});
                let _ = tx.send(Ok(Event::default().data(data.to_string()))).await;
            }
            Err(e) => {
                metrics::counter!(crate::metrics::names::REQUESTS_TOTAL, "endpoint" => endpoint, "status" => "err")
                    .increment(1);
                metrics::histogram!(crate::metrics::names::REQUEST_DURATION, "endpoint" => endpoint)
                    .record(start.elapsed().as_secs_f64());
                let data = serde_json::json!({"type": "error", "message": e.to_string()});
                let _ = tx.send(Ok(Event::default().data(data.to_string()))).await;
            }
        }
    });

    Sse::new(ReceiverStream::new(rx)).into_response()
}
