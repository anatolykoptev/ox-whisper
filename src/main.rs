use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;

mod audio;
mod chunking;
mod config;
mod detect;
mod diarize;
mod formats;
mod handler_openai;
mod handler_stream;
mod handlers;
mod metrics;
mod models;
mod openai;
mod paragraphs;
mod pii;
mod pool;
mod punctuate;
mod recognizer;
mod routing;
mod smart_format;
mod spelling;
mod streaming;
mod tmpfile;
mod transcribe;
mod tts;
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

    if config.idle_evict_secs > 0 {
        tracing::info!(
            "Idle eviction enabled: recognizers will be evicted after {}s of inactivity",
            config.idle_evict_secs
        );
    }

    tracing::info!("Loading models...");
    let models = Models::load(&config);

    // TTS child supervision is lazy: the supervisor exists but the tts-server
    // child is not spawned until the first ensure_ready() call.
    let tts = if config.tts.enabled {
        let sup = Arc::new(crate::tts::TtsSupervisor::new(config.tts.clone()));
        match &config.tts.upstream_url {
            Some(url) => {
                tracing::info!("TTS enabled via external upstream {url} (no managed child)");
            }
            None => {
                if config.tts.model.is_empty() || config.tts.codec.is_empty() {
                    tracing::warn!(
                        "TTS_ENABLED but TTS_MODEL/TTS_CODEC unset — first TTS request will fail to spawn"
                    );
                }
                let bin = &config.tts.bin;
                let executable = std::fs::metadata(bin)
                    .map(|m| m.is_file() && has_exec_bit(&m))
                    .unwrap_or(false);
                if !executable {
                    tracing::warn!(
                        "TTS_ENABLED but TTS_BIN {bin} is missing or not executable — TTS requests will fail"
                    );
                }
                if config.tts.idle_stop_secs > 0 {
                    let tick = std::time::Duration::from_secs(config.tts.idle_stop_secs / 4)
                        .max(std::time::Duration::from_secs(1));
                    sup.spawn_idle_loop(tick);
                    tracing::info!(
                        idle_stop_secs = config.tts.idle_stop_secs,
                        ?tick,
                        "TTS idle-stop enabled"
                    );
                }
            }
        }
        Some(sup)
    } else {
        None
    };

    let state = Arc::new(AppState {
        models,
        config,
        tts,
    });
    let tts_sup = state.tts.clone();

    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/transcribe", post(handlers::transcribe_json))
        .route("/transcribe/upload", post(handlers::transcribe_upload))
        .route(
            "/transcribe/stream",
            post(handler_stream::transcribe_stream),
        )
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
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Stop the TTS child via SIGTERM so `docker stop` does not time out into
    // SIGKILL.
    if let Some(sup) = tts_sup {
        sup.shutdown().await;
    }

    Ok(())
}

/// File has at least one executable bit (always true on non-unix).
fn has_exec_bit(m: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        m.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        let _ = m;
        true
    }
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
