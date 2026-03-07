pub mod audio;
pub mod chunking;
mod config;
pub mod models;
pub mod punctuate;
pub mod vad;

use axum::{Router, routing::get};
use serde::Serialize;
use tokio::net::TcpListener;

use crate::config::Config;

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    engine: &'static str,
    version: &'static str,
}

async fn health() -> axum::Json<HealthResponse> {
    axum::Json(HealthResponse {
        status: "starting",
        engine: "sherpa-onnx",
        version: "0.1.0",
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();

    let app = Router::new().route("/health", get(health));

    let addr = format!("0.0.0.0:{}", config.port);
    tracing::info!("Starting ox-whisper on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
