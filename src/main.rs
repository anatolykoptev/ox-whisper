use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;

#[cfg(test)]
mod api_tests;
mod audio;
mod chunking;
mod config;
mod formats;
mod handler_openai;
mod handlers;
mod language;
mod metrics;
mod models;
mod openai;
mod paragraphs;
mod pii;
mod pool;
mod server;
mod smart_format;
mod spelling;
mod tmpfile;
mod transcribe;
mod upload;
mod vad;
mod words;
mod ws_handler;
mod ws_session;
mod ws_types;

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

    let prom_handle = metrics::install_recorder();
    let prom_addr: std::net::SocketAddr = format!("0.0.0.0:{}", config.prom_port)
        .parse()
        .expect("invalid prom_port");
    tokio::spawn(metrics::serve(prom_handle, prom_addr));

    for var in config::removed_settings_present(&|k| std::env::var(k).ok()) {
        tracing::warn!("{var} is set but no longer used; it is ignored");
    }

    tracing::info!("Loading models...");
    let models = match Models::load(&config) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("{e}");
            std::process::exit(1);
        }
    };

    let drain = std::time::Duration::from_secs(config.shutdown_drain_s);
    let (trigger, shutdown) = server::ShutdownSignal::channel();
    let state = Arc::new(AppState {
        models,
        config,
        shutdown,
    });

    let app = Router::new()
        .route("/health", get(handlers::health))
        .route(
            "/v1/audio/transcriptions",
            post(handler_openai::transcriptions),
        )
        .route("/v1/models", get(handler_openai::list_models))
        .route("/v1/listen", get(ws_handler::ws_listen))
        .layer(DefaultBodyLimit::max(max_body_size))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Starting ox-whisper on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    if server::serve(listener, app, shutdown_signal(), trigger, drain).await?
        == server::Outcome::TimedOut
    {
        // Returning would drop the runtime, which waits for every decode still
        // running on the blocking pool: the hang the drain bound exists to end.
        std::process::exit(1);
    }

    Ok(())
}

/// Resolves on SIGINT or SIGTERM — `docker stop` sends SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => sig.recv().await,
            Err(_) => std::future::pending::<Option<()>>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<Option<()>>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
