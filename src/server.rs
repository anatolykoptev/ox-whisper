//! The serve loop with a bounded graceful shutdown.
//!
//! axum's graceful shutdown stops accepting and waits for the connections in
//! flight, with no limit: one stuck request, or a client that keeps a
//! connection open, keeps the process alive until the supervisor kills it.
//! [`serve`] tells live WebSocket sessions to close, then gives everything
//! else `drain` to finish before returning anyway.

use std::future::{Future, IntoFuture};
use std::time::Duration;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::{oneshot, watch};

/// Handed to every WebSocket session; resolves once shutdown has begun.
#[derive(Clone)]
pub struct ShutdownSignal(watch::Receiver<bool>);

impl ShutdownSignal {
    /// A signal tied to the returned trigger: `send(true)` fires it.
    pub fn channel() -> (watch::Sender<bool>, Self) {
        let (tx, rx) = watch::channel(false);
        (tx, Self(rx))
    }

    /// A signal that never fires (tests).
    #[cfg(test)]
    pub fn inert() -> Self {
        // The sender is dropped at once: `fired` treats that as "never".
        Self::channel().1
    }

    /// Resolves when shutdown begins; pends forever if the trigger is gone.
    pub async fn fired(&mut self) {
        if self.0.wait_for(|begun| *begun).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

/// How [`serve`] ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every connection finished.
    Drained,
    /// The drain ran out with work in flight. Dropping the runtime now would
    /// wait for any decode still running on the blocking pool, so the caller
    /// should exit the process instead.
    TimedOut,
}

/// Serves `app` until `stop` resolves, then shuts down: `trigger` is fired so
/// WebSocket sessions close, new connections are refused, and in-flight
/// requests get `drain` to finish. Returns when the server has drained or the
/// drain ran out, whichever is first.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    stop: impl Future<Output = ()> + Send + 'static,
    trigger: watch::Sender<bool>,
    drain: Duration,
) -> std::io::Result<Outcome> {
    let (begun_tx, begun_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            stop.await;
            let _ = trigger.send(true);
            let _ = begun_tx.send(());
        })
        .into_future();
    let drain_timer = async {
        // Err: the server finished without ever being told to stop.
        if begun_rx.await.is_ok() {
            tokio::time::sleep(drain).await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        result = server => result.map(|()| Outcome::Drained),
        _ = drain_timer => {
            tracing::warn!(?drain, "shutdown drain timed out; exiting with work still in flight");
            Ok(Outcome::TimedOut)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    use axum::routing::get;
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    use tokio_tungstenite::tungstenite::Message as ClientMsg;

    use crate::config::Config;
    use crate::handlers::AppState;
    use crate::models::Models;

    struct Running {
        addr: std::net::SocketAddr,
        stop: oneshot::Sender<()>,
        server: tokio::task::JoinHandle<std::io::Result<Outcome>>,
    }

    /// `/v1/listen` plus a `/hang` route that never answers. `wired`: whether
    /// the sessions are told about the shutdown.
    async fn start(wired: bool, drain: Duration) -> Running {
        let (trigger, signal) = ShutdownSignal::channel();
        let state = Arc::new(AppState {
            models: Models::empty(),
            config: Config::from_lookup(&|_| None),
            shutdown: if wired {
                signal
            } else {
                ShutdownSignal::inert()
            },
        });
        let app = Router::new()
            .route("/v1/listen", get(crate::ws_handler::ws_listen))
            .route(
                "/hang",
                get(|| async { std::future::pending::<&'static str>().await }),
            )
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        let server = tokio::spawn(serve(
            listener,
            app,
            async move {
                let _ = stopped.await;
            },
            trigger,
            drain,
        ));
        Running { addr, stop, server }
    }

    async fn ws_client(
        addr: std::net::SocketAddr,
    ) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
    {
        let url = format!("ws://{addr}/v1/listen?vad=false");
        let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        // Metadata is the first frame: the session is live.
        let first = ws.next().await.unwrap().unwrap();
        assert!(matches!(first, ClientMsg::Text(_)), "{first:?}");
        ws
    }

    /// Premise, measured: the HTTP server's graceful shutdown does not wait
    /// for an upgraded connection, so `serve` returns while a session is still
    /// open and the client is simply cut off, with no Close frame. Telling the
    /// sessions is therefore the fix, not the drain.
    #[tokio::test]
    async fn graceful_shutdown_alone_cuts_a_live_session_off_without_a_close() {
        let run = start(false, Duration::from_secs(3600)).await;
        let mut ws = ws_client(run.addr).await;
        run.stop.send(()).unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(3), run.server)
            .await
            .expect("serve should not wait for the upgraded session")
            .unwrap()
            .unwrap();
        assert_eq!(outcome, Outcome::Drained);
        let frame = tokio::time::timeout(Duration::from_millis(500), ws.next()).await;
        assert!(
            !matches!(frame, Ok(Some(Ok(ClientMsg::Close(_))))),
            "unexpected clean close: {frame:?}"
        );
    }

    /// The fix for the above: the client gets a 1001 Close and the server
    /// returns promptly, long before the drain runs out.
    #[tokio::test]
    async fn shutdown_closes_live_sessions_with_going_away() {
        let run = start(true, Duration::from_secs(60)).await;
        let mut ws = ws_client(run.addr).await;
        let began = Instant::now();
        run.stop.send(()).unwrap();
        let frame = tokio::time::timeout(Duration::from_secs(3), ws.next())
            .await
            .expect("no frame after shutdown began");
        match frame {
            Some(Ok(ClientMsg::Close(Some(close)))) => {
                assert_eq!(u16::from(close.code), 1001, "{close:?}");
            }
            other => panic!("expected a Close(1001), got {other:?}"),
        }
        // Complete the close handshake so the connection ends.
        while let Ok(Some(Ok(_))) =
            tokio::time::timeout(Duration::from_millis(200), ws.next()).await
        {}
        drop(ws);
        let done = tokio::time::timeout(Duration::from_secs(3), run.server)
            .await
            .expect("serve did not return after the session closed");
        assert_eq!(done.unwrap().unwrap(), Outcome::Drained);
        assert!(began.elapsed() < Duration::from_secs(10));
    }

    /// A request that never finishes cannot hold the process past the drain.
    #[tokio::test]
    async fn the_drain_bounds_a_request_that_never_finishes() {
        let drain = Duration::from_millis(400);
        let run = start(true, drain).await;
        let mut conn = tokio::net::TcpStream::connect(run.addr).await.unwrap();
        conn.write_all(b"GET /hang HTTP/1.1\r\nHost: t\r\n\r\n")
            .await
            .unwrap();
        // Let the request reach its handler before shutdown begins.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let began = Instant::now();
        run.stop.send(()).unwrap();
        let done = tokio::time::timeout(Duration::from_secs(5), run.server)
            .await
            .expect("serve still running long after the drain");
        assert_eq!(done.unwrap().unwrap(), Outcome::TimedOut);
        assert!(
            began.elapsed() >= drain,
            "returned before the drain: {:?}",
            began.elapsed()
        );
    }
}
