//! The serve loop with a bounded graceful shutdown.
//!
//! axum's graceful shutdown stops accepting and waits for the HTTP requests in
//! flight, with no limit, and it does not wait for upgraded (WebSocket)
//! connections at all. [`serve`] therefore tells live sessions to close, then
//! waits for the requests *and* the sessions together, inside one `drain`
//! deadline. [`block_on_bounded`] bounds the last unbounded step: dropping a
//! tokio runtime waits for every `spawn_blocking` job, such as a decode whose
//! client has already left.

use std::future::{Future, IntoFuture};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::{oneshot, watch};

/// Handed to every WebSocket session: resolves once shutdown has begun, and
/// counts the session as live until its [`SessionGuard`] is dropped.
#[derive(Clone)]
pub struct ShutdownSignal {
    begun: watch::Receiver<bool>,
    live: Arc<watch::Sender<usize>>,
}

/// The server's end: fires the signal and watches the live-session count.
pub struct ShutdownTrigger {
    begun: watch::Sender<bool>,
    live: watch::Receiver<usize>,
}

/// Keeps a session counted as live; hold it for the session's whole life,
/// including any decode it is awaiting.
pub struct SessionGuard(Arc<watch::Sender<usize>>);

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.0.send_modify(|n| *n -= 1);
    }
}

impl ShutdownSignal {
    /// A connected pair: the trigger goes to [`serve`], the signal to the sessions.
    pub fn channel() -> (ShutdownTrigger, Self) {
        let (begun_tx, begun_rx) = watch::channel(false);
        let (live_tx, live_rx) = watch::channel(0usize);
        (
            ShutdownTrigger {
                begun: begun_tx,
                live: live_rx,
            },
            Self {
                begun: begun_rx,
                live: Arc::new(live_tx),
            },
        )
    }

    /// A signal that never fires (tests).
    #[cfg(test)]
    pub fn inert() -> Self {
        // The trigger is dropped at once: `fired` treats that as "never".
        Self::channel().1
    }

    /// Counts a new live session. Take it before the upgrade completes, so
    /// there is no window in which a session exists uncounted.
    pub fn session(&self) -> SessionGuard {
        self.live.send_modify(|n| *n += 1);
        SessionGuard(self.live.clone())
    }

    /// Resolves when shutdown begins; pends forever if the trigger is gone.
    pub async fn fired(&mut self) {
        if self.begun.wait_for(|begun| *begun).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

/// How [`serve`] ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every request and session finished.
    Drained,
    /// The drain ran out with work in flight.
    TimedOut,
}

/// Serves `app` until `stop` resolves, then shuts down: sessions are told to
/// close, new connections are refused, and requests and sessions get `drain`
/// (counted from `stop`) to finish. Returns when everything has finished or
/// the drain ran out, whichever is first.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    stop: impl Future<Output = ()> + Send + 'static,
    trigger: ShutdownTrigger,
    drain: Duration,
) -> std::io::Result<Outcome> {
    let ShutdownTrigger { begun, mut live } = trigger;
    let (begun_tx, begun_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            stop.await;
            let _ = begun.send(true);
            let _ = begun_tx.send(());
        })
        .into_future();
    let finished = async {
        server.await?;
        // Upgraded connections are not part of the HTTP drain: wait for the
        // sessions themselves (an idle one is still closing, one in a decode
        // finishes it first).
        let _ = live.wait_for(|n| *n == 0).await;
        Ok::<_, std::io::Error>(Outcome::Drained)
    };
    let drain_timer = async {
        // Err: the server finished without ever being told to stop.
        if begun_rx.await.is_ok() {
            tokio::time::sleep(drain).await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        result = finished => result,
        _ = drain_timer => {
            tracing::warn!(?drain, "shutdown drain timed out; exiting with work still in flight");
            Ok(Outcome::TimedOut)
        }
    }
}

/// Runs `fut` to completion on `rt`, then tears the runtime down within
/// `teardown`. A plain drop of the runtime waits for every blocking job with
/// no limit, which includes a decode whose client left; past `teardown` those
/// jobs are abandoned and die with the process.
pub fn block_on_bounded<F: Future>(
    rt: tokio::runtime::Runtime,
    fut: F,
    teardown: Duration,
) -> F::Output {
    let out = rt.block_on(fut);
    rt.shutdown_timeout(teardown);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use axum::routing::get;
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    use tokio::sync::Notify;
    use tokio_tungstenite::tungstenite::Message as ClientMsg;

    use crate::config::Config;
    use crate::handlers::AppState;
    use crate::models::Models;

    struct Running {
        addr: std::net::SocketAddr,
        stop: oneshot::Sender<()>,
        server: tokio::task::JoinHandle<std::io::Result<Outcome>>,
        /// Notified when the `/hang` handler has been entered.
        entered: Arc<Notify>,
    }

    /// `/v1/listen`, a `/hang` route that never answers, and `/busy-session`,
    /// an upgraded connection that stays busy for 700 ms (a decode in flight)
    /// and ignores the shutdown signal. `wired`: whether `/v1/listen`
    /// sessions are told about the shutdown.
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
        let entered = Arc::new(Notify::new());
        let hang_entered = entered.clone();
        let busy_state = state.clone();
        let app = Router::new()
            .route("/v1/listen", get(crate::ws_handler::ws_listen))
            .route(
                "/hang",
                get(move || {
                    let entered = hang_entered.clone();
                    async move {
                        entered.notify_one();
                        std::future::pending::<&'static str>().await
                    }
                }),
            )
            .route(
                "/busy-session",
                get(move |ws: axum::extract::WebSocketUpgrade| {
                    let guard = busy_state.shutdown.session();
                    async move {
                        ws.on_upgrade(move |_socket| async move {
                            let _guard = guard;
                            tokio::time::sleep(Duration::from_millis(700)).await;
                        })
                    }
                }),
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
        Running {
            addr,
            stop,
            server,
            entered,
        }
    }

    type Client = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    async fn ws_client(addr: std::net::SocketAddr) -> Client {
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
    /// sessions, and waiting for them, is therefore the fix.
    #[tokio::test]
    async fn graceful_shutdown_alone_does_not_wait_for_an_upgraded_connection() {
        // No ShutdownSignal session guard is taken here: this is plain axum.
        let app = Router::new().route(
            "/ws",
            get(|ws: axum::extract::WebSocketUpgrade| async move {
                ws.on_upgrade(|_s| async { tokio::time::sleep(Duration::from_secs(3600)).await })
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = stopped.await;
                })
                .await
        });
        let (_ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .unwrap();
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), server)
            .await
            .expect("axum waited for the upgraded connection")
            .unwrap()
            .unwrap();
    }

    /// The client gets a 1001 Close, and `serve` returns well before the drain.
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
        // Complete the close handshake so the session ends.
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

    /// `serve` does not return while a session is still closing: a client that
    /// never answers the Close keeps the session in its bounded close wait
    /// (1 s), and returning earlier would let `main` drop the runtime under it.
    #[tokio::test]
    async fn serve_waits_for_an_idle_session_to_finish_closing() {
        let run = start(true, Duration::from_secs(30)).await;
        // Held, never polled: no Close reply goes back.
        let _ws = ws_client(run.addr).await;
        let began = Instant::now();
        run.stop.send(()).unwrap();
        let done = tokio::time::timeout(Duration::from_secs(10), run.server)
            .await
            .expect("serve did not return");
        assert_eq!(done.unwrap().unwrap(), Outcome::Drained);
        assert!(
            began.elapsed() >= Duration::from_millis(900),
            "returned while the session was still closing: {:?}",
            began.elapsed()
        );
    }

    /// An upgraded session that is busy (a decode in flight) is waited for,
    /// within the drain.
    #[tokio::test]
    async fn serve_waits_for_a_busy_session_within_the_drain() {
        let run = start(true, Duration::from_secs(30)).await;
        let (_ws, _) = tokio_tungstenite::connect_async(format!("ws://{}/busy-session", run.addr))
            .await
            .unwrap();
        let began = Instant::now();
        run.stop.send(()).unwrap();
        let done = tokio::time::timeout(Duration::from_secs(10), run.server)
            .await
            .expect("serve did not return");
        assert_eq!(done.unwrap().unwrap(), Outcome::Drained);
        assert!(
            began.elapsed() >= Duration::from_millis(500),
            "returned with the session still busy: {:?}",
            began.elapsed()
        );
    }

    /// A busy session that outlasts the drain is cut off at the drain.
    #[tokio::test]
    async fn the_drain_bounds_a_busy_session() {
        let run = start(true, Duration::from_millis(200)).await;
        let (_ws, _) = tokio_tungstenite::connect_async(format!("ws://{}/busy-session", run.addr))
            .await
            .unwrap();
        run.stop.send(()).unwrap();
        let done = tokio::time::timeout(Duration::from_secs(5), run.server)
            .await
            .expect("serve did not return");
        assert_eq!(done.unwrap().unwrap(), Outcome::TimedOut);
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
        // The handler has been entered: shutdown begins with a request in flight.
        run.entered.notified().await;
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

    /// The real teardown path: the server drained cleanly, but a blocking job
    /// (a decode whose client left) is still running. Dropping the runtime
    /// would wait for it; `block_on_bounded` must not.
    #[test]
    fn teardown_does_not_wait_for_a_blocking_job() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let began = Instant::now();
        let outcome = block_on_bounded(
            rt,
            async {
                let run = start(true, Duration::from_secs(5)).await;
                // An abandoned decode: nothing awaits this handle.
                tokio::task::spawn_blocking(|| std::thread::sleep(Duration::from_secs(4)));
                run.stop.send(()).unwrap();
                run.server.await.unwrap().unwrap()
            },
            Duration::from_millis(300),
        );
        assert_eq!(outcome, Outcome::Drained);
        assert!(
            began.elapsed() < Duration::from_secs(2),
            "teardown waited for the blocking job: {:?}",
            began.elapsed()
        );
    }
}
