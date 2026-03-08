use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;

mod audio;
mod chunking;
mod config;
mod detect;
mod formats;
mod handler_openai;
mod handler_stream;
mod handlers;
mod models;
mod openai;
mod paragraphs;
mod pii;
mod pool;
mod punctuate;
mod recognizer;
mod smart_format;
mod spelling;
mod streaming;
mod transcribe;
mod vad;
mod words;

use crate::config::Config;
use crate::handlers::AppState;
use crate::models::Models;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();
    let port = config.port;
    let max_body_size = config.max_body_size_mb * 1024 * 1024;

    tracing::info!("Loading models...");
    let models = Models::load(&config);

    let state = Arc::new(AppState { models, config });

    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/transcribe", post(handlers::transcribe_json))
        .route("/transcribe/upload", post(handlers::transcribe_upload))
        .route("/transcribe/stream", post(handler_stream::transcribe_stream))
        .route("/v1/audio/transcriptions", post(handler_openai::transcriptions))
        .route("/v1/models", get(handler_openai::list_models))
        .layer(DefaultBodyLimit::max(max_body_size))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Starting ox-whisper on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
