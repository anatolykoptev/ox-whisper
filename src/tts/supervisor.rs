//! Supervisor for the external `tts-server` child process.
//!
//! The TTS engine is an upstream `tts-server` binary (qwentts.cpp) run as a
//! child process inside the same container, listening on 127.0.0.1. It has no
//! auth; ox-whisper is its only client. The supervisor owns the child's
//! lifecycle:
//!
//! - lazy start: nothing spawns at boot; the first [`ensure_ready`] starts the
//!   child and waits for `GET /health` to answer 200
//! - single-flight: each start runs in a spawned leader task that owns every
//!   transition of its generation (drain, backoff, spawn, probe, commit);
//!   callers only subscribe to its shared outcome, so a dropped caller can
//!   never abort a start and concurrent callers share one child
//! - idle stop: a background loop stops the child after `idle_stop_secs` with
//!   no in-flight [`TtsGuard`] (0 = never)
//! - crash handling: an unrequested exit flips state to `Crashed`; the next
//!   `ensure_ready` restarts with backoff (1 s doubling to 30 s, reset after
//!   60 s of being healthy)
//! - shutdown: [`shutdown`] stops the running child with SIGTERM and keeps any
//!   start that has not spawned yet from spawning one
//! - OOM priority: on Linux the child gets `oom_score_adj=1000` and nice 10 so
//!   the kernel OOM killer picks the TTS engine before the STT server in the
//!   shared container cgroup
//!
//! [`ensure_ready`]: TtsSupervisor::ensure_ready
//! [`shutdown`]: TtsSupervisor::shutdown

use std::io;
use std::process::Stdio;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use tokio::process::Child;
use tokio::sync::watch;
use tokio::time::Instant;

use crate::config::TtsConfig;
use crate::metrics::names;

/// Lifecycle state of the managed TTS child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsState {
    /// No child is managed because TTS is off (no supervisor exists).
    Disabled,
    /// An external `TTS_UPSTREAM_URL` is configured — no child is managed.
    External,
    /// No child process is running.
    Stopped,
    /// Child spawned, waiting for `/health` to answer 200.
    Starting,
    /// Child answers `GET /health` with 200.
    Ready,
    /// Child exited without a stop request; restart is allowed after backoff.
    Crashed,
}

/// Errors returned by [`TtsSupervisor::ensure_ready`]. Cloneable so every
/// waiter of one start generation observes the same error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TtsError {
    /// `TTS_ENABLED` is false — the supervisor is not supposed to run a child.
    #[error("tts is disabled")]
    Disabled,
    /// The `tts-server` binary could not be spawned.
    #[error("tts child spawn failed: {0}")]
    Spawn(String),
    /// The child exited before answering `/health` with 200.
    #[error("tts child exited before becoming healthy: {0}")]
    StartupExit(String),
    /// `/health` did not answer 200 within `startup_timeout_secs`.
    #[error("tts child did not answer /health within {0}s")]
    StartupTimeout(u64),
    /// `TTS_PORT` is already bound by another process — `/health` answers
    /// would be a foreign listener's, never ours.
    #[error("tts port {0} is already bound by another process")]
    PortBusy(u16),
    /// The previous child ignored its stop request past the drain deadline.
    #[error("previous child still running")]
    DrainTimeout,
    /// The leader task owning the start died before publishing an outcome.
    #[error("tts start leader exited without an outcome")]
    StartAborted,
    /// [`TtsSupervisor::shutdown`] was called — no child starts any more.
    #[error("tts is shutting down")]
    ShuttingDown,
}

/// Why a running child was asked to stop. Feeds `stops_total{reason}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    Idle,
    Shutdown,
    /// The child never answered `/health` within `startup_timeout_secs`.
    StartupTimeout,
}

impl StopReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Shutdown => "shutdown",
            Self::StartupTimeout => "startup_timeout",
        }
    }
}

struct Inner {
    state: TtsState,
    /// Monotonic id per spawned child; stale monitor reports are ignored.
    generation: u64,
    /// Wakes the child monitor to request a stop.
    stop_tx: Option<watch::Sender<()>>,
    /// Turns `true` when this generation's monitor has reaped the child. A
    /// fresh channel per spawn so a stale value cannot release a waiter
    /// early; a watch (not a Notify) so every waiter wakes, including one
    /// that subscribes after the reap.
    child_gone: Option<watch::Receiver<bool>>,
    /// Set when a stop was requested — distinguishes stops from crashes.
    stop_reason: Option<StopReason>,
    /// In-flight proxied requests — idle time only counts while zero. Kept
    /// under this mutex so a new request cannot slip between the idle loop's
    /// check and its stop decision.
    in_flight: usize,
    /// Last point in time the child had activity (ready transition or guard drop).
    last_activity: Instant,
    /// When the current child reached `Ready` — decides the backoff reset.
    ready_since: Option<Instant>,
    /// Current restart backoff (doubled per consecutive early crash).
    backoff: Duration,
    /// Earliest instant a restart is allowed after a crash.
    retry_after: Option<Instant>,
    /// Receiver of the in-flight start generation's shared outcome. `Some`
    /// iff a leader task currently owns a start; callers join by cloning it.
    start_rx: Option<watch::Receiver<StartOutcome>>,
    /// Set by `shutdown` — no new generation starts and a leader that has not
    /// spawned yet gives up instead of spawning.
    shutting_down: bool,
}

impl Inner {
    fn new() -> Self {
        Self {
            state: TtsState::Stopped,
            generation: 0,
            stop_tx: None,
            child_gone: None,
            stop_reason: None,
            in_flight: 0,
            last_activity: Instant::now(),
            ready_since: None,
            backoff: Duration::ZERO,
            retry_after: None,
            start_rx: None,
            shutting_down: false,
        }
    }
}

/// The single outcome of one start generation — `None` until the leader
/// publishes. Shared with every waiter through `Inner::start_rx`.
type StartOutcome = Option<Result<String, TtsError>>;

/// Process supervisor for the TTS child. One per process; lives in `AppState`.
pub struct TtsSupervisor {
    cfg: TtsConfig,
    inner: Mutex<Inner>,
    http: Client<HttpConnector, Empty<Bytes>>,
}

const BACKOFF_INITIAL: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(30);
const BACKOFF_RESET_AFTER: Duration = Duration::from_secs(60);
const HEALTH_POLL: Duration = Duration::from_millis(100);
/// Bound on a single `/health` probe — a listener that accepts but never
/// answers must not park the start loop.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// Grace period between SIGTERM and SIGKILL when stopping the child.
const TERM_GRACE: Duration = Duration::from_secs(3);
const STOP_WAIT: Duration = Duration::from_secs(5);
/// Upper bound on [`TtsSupervisor::shutdown`] waiting for the reap.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(10);

impl TtsSupervisor {
    pub fn new(cfg: TtsConfig) -> Self {
        let http = Client::builder(TokioExecutor::new()).build(HttpConnector::new());
        // Pre-register every counter and label value at 0 so `increase`-style
        // alerts observe the first event instead of a missing series.
        metrics::gauge!(names::TTS_CHILD_UP).set(0);
        metrics::counter!(names::TTS_CHILD_STARTS).increment(0);
        metrics::counter!(names::TTS_CHILD_RESTARTS).increment(0);
        for reason in ["idle", "shutdown", "startup_timeout"] {
            metrics::counter!(names::TTS_CHILD_STOPS, "reason" => reason).increment(0);
        }
        for reason in ["spawn", "exit", "timeout", "port_busy", "drain"] {
            metrics::counter!(names::TTS_CHILD_START_FAILURES, "reason" => reason).increment(0);
        }
        Self {
            cfg,
            inner: Mutex::new(Inner::new()),
            http,
        }
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Current lifecycle state. `External` when an upstream URL is
    /// configured (no child is ever managed).
    pub fn state(&self) -> TtsState {
        if self.cfg.upstream_url.is_some() {
            TtsState::External
        } else {
            self.inner().state
        }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.cfg.port)
    }

    /// RAII guard marking one in-flight TTS request. While at least one guard
    /// is alive the idle loop will not stop the child; dropping the last guard
    /// restarts the idle clock.
    pub fn guard(self: &Arc<Self>) -> TtsGuard {
        let mut inner = self.inner();
        inner.in_flight += 1;
        inner.last_activity = Instant::now();
        TtsGuard {
            sup: Arc::clone(self),
        }
    }

    /// Ensure a TTS endpoint is reachable and return its base URL plus a
    /// request guard that keeps the child alive.
    ///
    /// With `TTS_UPSTREAM_URL` set this just returns that URL and never
    /// spawns. Otherwise each start runs in a spawned leader task that owns
    /// every transition of its generation — drain of the previous child,
    /// crash backoff, the port check, spawn and the `/health` probe loop —
    /// and publishes one shared outcome on `Inner::start_rx`. Callers only
    /// subscribe: a dropped caller can never abort a start, and every waiter
    /// of a generation observes the same `Ok(url)` or the same error. The
    /// returned guard is taken under the same lock that observed `Ready`, so
    /// an idle stop can never slip between readiness and the guard.
    pub async fn ensure_ready(self: &Arc<Self>) -> Result<(String, TtsGuard), TtsError> {
        if !self.cfg.enabled {
            return Err(TtsError::Disabled);
        }
        if let Some(url) = &self.cfg.upstream_url {
            return Ok((url.clone(), self.guard()));
        }
        loop {
            // Join the in-flight generation or launch a new leader task for
            // it. `start_rx.is_some()` is the single-flight token: exactly one
            // leader exists per generation and it is never cancelled here.
            let mut rx = {
                let mut inner = self.inner();
                if inner.shutting_down {
                    return Err(TtsError::ShuttingDown);
                }
                if let Some(ready) = Self::ready_locked(&mut inner, self) {
                    return Ok(ready);
                }
                match inner.start_rx.clone() {
                    Some(rx) => rx,
                    None => {
                        let (tx, rx) = watch::channel::<StartOutcome>(None);
                        inner.start_rx = Some(rx.clone());
                        let sup = Arc::clone(self);
                        tokio::spawn(async move {
                            // Clears `start_rx` on ANY exit — published
                            // outcome, failure or panic — so later callers can
                            // start a new generation and a dead leader cannot
                            // wedge the supervisor in `Starting`.
                            let cleanup = StartCleanup {
                                sup: Arc::clone(&sup),
                            };
                            let outcome = sup.start_run().await;
                            // Retire the generation before publishing: a caller
                            // arriving after the publish must launch the next
                            // generation, not rejoin this finished one.
                            drop(cleanup);
                            let _ = tx.send(Some(outcome));
                        });
                        rx
                    }
                }
            };

            // Wait for the leader to publish this generation's outcome. The
            // sender lives in the leader task; if it dies without publishing,
            // `changed` errors and the waiter bails out with StartAborted.
            let outcome = loop {
                if let Some(out) = &*rx.borrow() {
                    break out.clone();
                }
                if rx.changed().await.is_err() {
                    break Err(TtsError::StartAborted);
                }
            };
            match outcome {
                Err(e) => return Err(e),
                Ok(_) => {
                    // Re-observe `Ready` under the lock that takes the guard:
                    // the leader published success but an idle stop may have
                    // raced us — then the child is on its way down and its URL
                    // must not be handed out. Re-loop to join the next start.
                    let mut inner = self.inner();
                    match Self::ready_locked(&mut inner, self) {
                        Some(ready) => return Ok(ready),
                        None => continue,
                    }
                }
            }
        }
    }

    /// If the child is `Ready`, take a request guard under the same lock and
    /// return the base URL with it. `None` when not ready.
    fn ready_locked(inner: &mut Inner, sup: &Arc<Self>) -> Option<(String, TtsGuard)> {
        if inner.state != TtsState::Ready {
            return None;
        }
        inner.in_flight += 1;
        inner.last_activity = Instant::now();
        Some((
            sup.base_url(),
            TtsGuard {
                sup: Arc::clone(sup),
            },
        ))
    }

    /// The start leader's body: owns every transition of one start
    /// generation — drain, backoff, port check, spawn, probe, commit. Runs
    /// detached inside the spawned task, so it is never cancelled by callers.
    /// On success the child is `Ready`; on failure the state is `Stopped` or
    /// `Crashed` and `retry_after` defers the next generation.
    async fn start_run(self: &Arc<Self>) -> Result<String, TtsError> {
        // A previous child may still be reaping — wait for it before binding
        // the port again. A wedged child fails the whole generation.
        if let Err(e) = self.drain_child().await {
            self.bump_backoff();
            return Err(self.fail_start("drain", e));
        }

        // Crash backoff: the previous failure defers this start.
        let delay = self
            .inner()
            .retry_after
            .map(|t| t.saturating_duration_since(Instant::now()))
            .unwrap_or_default();
        if !delay.is_zero() {
            tracing::info!(?delay, "tts child start deferred by backoff");
            tokio::time::sleep(delay).await;
        }

        // The port must be free before spawn — otherwise `/health` answers
        // could come from a foreign listener and get committed as ours.
        if std::net::TcpListener::bind(("127.0.0.1", self.cfg.port)).is_err() {
            self.bump_backoff();
            return Err(self.fail_start("port_busy", TtsError::PortBusy(self.cfg.port)));
        }

        let generation = {
            let mut inner = self.inner();
            // `shutdown` may have run while this leader drained or slept in
            // backoff: it found no child to stop, so nothing may spawn now.
            if inner.shutting_down {
                return Err(TtsError::ShuttingDown);
            }
            inner.state = TtsState::Starting;
            inner.generation += 1;
            // A fresh generation must not inherit a stop request recorded for
            // the previous child.
            inner.stop_reason = None;
            inner.generation
        };

        let (stop_tx, stop_rx) = watch::channel(());
        let child = match self.spawn_child() {
            Ok(child) => child,
            Err(e) => {
                self.inner().state = TtsState::Stopped;
                self.bump_backoff();
                return Err(self.fail_start("spawn", e));
            }
        };
        let (gone_tx, gone_rx) = watch::channel(false);
        {
            let mut inner = self.inner();
            inner.stop_tx = Some(stop_tx);
            inner.child_gone = Some(gone_rx);
            if inner.shutting_down {
                // `shutdown` ran between the generation commit and the spawn,
                // when there was no stop sender yet — stop the child now.
                Self::request_stop_locked(&mut inner, StopReason::Shutdown);
            }
        }
        tokio::spawn(Self::monitor(
            Arc::downgrade(self),
            generation,
            child,
            stop_rx,
            gone_tx,
        ));

        match self.wait_ready(generation).await {
            Ok(()) => {
                let mut inner = self.inner();
                if inner.state == TtsState::Starting && inner.generation == generation {
                    inner.state = TtsState::Ready;
                    let now = Instant::now();
                    inner.ready_since = Some(now);
                    inner.last_activity = now;
                    metrics::gauge!(names::TTS_CHILD_UP).set(1);
                    tracing::info!(port = self.cfg.port, "tts child is ready");
                    Ok(self.base_url())
                } else if inner.shutting_down {
                    Err(TtsError::ShuttingDown)
                } else {
                    // The monitor beat us to a transition (e.g. the child died
                    // right after a successful probe).
                    Err(self.fail_start(
                        "exit",
                        TtsError::StartupExit("child exited during startup".to_string()),
                    ))
                }
            }
            Err(TtsError::StartupTimeout(secs)) => {
                let mut inner = self.inner();
                if inner.state == TtsState::Starting && inner.generation == generation {
                    // Abort the spawn — request a stop so the child cannot
                    // linger without supervision. `on_child_exit` counts the
                    // stop as `startup_timeout` and applies the crash backoff.
                    inner.state = TtsState::Stopped;
                    inner.stop_reason = Some(StopReason::StartupTimeout);
                    if let Some(tx) = inner.stop_tx.take() {
                        let _ = tx.send(());
                    }
                }
                Err(self.fail_start("timeout", TtsError::StartupTimeout(secs)))
            }
            Err(_) if self.inner().shutting_down => {
                // `shutdown` stopped the child mid-start — not a failed start.
                Err(TtsError::ShuttingDown)
            }
            Err(e) => {
                // The child exited mid-start: `on_child_exit` already recorded
                // the `Crashed` state and the backoff.
                Err(self.fail_start("exit", e))
            }
        }
    }

    /// Wait for the previous child to be reaped before a new spawn binds the
    /// port, requesting its stop if the monitor has not been asked yet.
    /// Bounded by `STOP_WAIT` — a wedged child must not hang the start.
    async fn drain_child(&self) -> Result<(), TtsError> {
        let deadline = Instant::now() + STOP_WAIT;
        loop {
            let pending = {
                let mut inner = self.inner();
                match inner.child_gone.clone() {
                    Some(gone) => {
                        let tx = inner.stop_tx.take();
                        if tx.is_some() && inner.stop_reason.is_none() {
                            inner.stop_reason = Some(StopReason::Shutdown);
                        }
                        Some((tx, gone))
                    }
                    None => None,
                }
            }; // MutexGuard drops here — before any await.
            match pending {
                Some((tx, mut gone)) => {
                    if let Some(tx) = tx {
                        let _ = tx.send(());
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(TtsError::DrainTimeout);
                    }
                    let _ = tokio::time::timeout(
                        deadline.saturating_duration_since(now),
                        gone.wait_for(|reaped| *reaped),
                    )
                    .await;
                }
                None => return Ok(()),
            }
        }
    }

    /// Move the crash backoff one step up and defer the next start.
    fn bump_backoff(&self) {
        let mut inner = self.inner();
        inner.backoff = next_backoff(inner.backoff, None);
        inner.retry_after = Some(Instant::now() + inner.backoff);
    }

    /// Record one start failure — the labelled counter plus an error log —
    /// and return the error unchanged for publishing.
    fn fail_start(&self, reason: &'static str, err: TtsError) -> TtsError {
        metrics::counter!(names::TTS_CHILD_START_FAILURES, "reason" => reason).increment(1);
        tracing::error!(reason, error = %err, "tts child start failed");
        err
    }

    /// Stop the running child and keep any later or not-yet-spawned start
    /// from spawning one, then wait until the child is reaped (bounded by
    /// `SHUTDOWN_WAIT` so a wedged child cannot hang the exit). Called from
    /// `main`'s graceful-shutdown path so `docker stop` stops the child via
    /// SIGTERM instead of timing out into SIGKILL.
    pub async fn shutdown(&self) {
        let gone = {
            let mut inner = self.inner();
            inner.shutting_down = true;
            Self::request_stop_locked(&mut inner, StopReason::Shutdown);
            inner.child_gone.clone()
        };
        if let Some(mut gone) = gone {
            let _ = tokio::time::timeout(SHUTDOWN_WAIT, gone.wait_for(|reaped| *reaped)).await;
        }
    }

    /// Request a stop of the current child, if any is running.
    fn request_stop_locked(inner: &mut Inner, reason: StopReason) {
        if inner.stop_tx.is_none() {
            return;
        }
        if inner.stop_reason.is_none() {
            inner.stop_reason = Some(reason);
        }
        if matches!(inner.state, TtsState::Ready | TtsState::Starting) {
            inner.state = TtsState::Stopped;
        }
        if let Some(tx) = &inner.stop_tx {
            let _ = tx.send(());
        }
    }

    /// Spawn the tts-server binary with its arguments, env and Linux
    /// OOM/priority tweaks, then wire stdout/stderr into tracing.
    fn spawn_child(&self) -> Result<Child, TtsError> {
        let mut cmd = std::process::Command::new(&self.cfg.bin);
        cmd.arg("--model")
            .arg(&self.cfg.model)
            .arg("--codec")
            .arg(&self.cfg.codec)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(self.cfg.port.to_string())
            .arg("--max-batch")
            .arg(self.cfg.max_batch.to_string())
            .env("QT_N_THREADS", self.cfg.threads.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(child_pre_exec);
        }
        let mut child = tokio::process::Command::from(cmd)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| TtsError::Spawn(e.to_string()))?;
        metrics::counter!(names::TTS_CHILD_STARTS).increment(1);
        tracing::info!(bin = %self.cfg.bin, port = self.cfg.port, "tts child spawned");
        // Pipe the child's output into `tts_child` tracing. Detached: the
        // tasks end by themselves when the pipes close at child exit.
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(forward_output(stdout, "stdout"));
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(forward_output(stderr, "stderr"));
        }
        Ok(child)
    }

    /// Owns one child: waits for exit, or kills it on a stop request / on
    /// supervisor drop (stop sender closed), then reports the transition.
    async fn monitor(
        sup: Weak<TtsSupervisor>,
        generation: u64,
        mut child: Child,
        mut stop_rx: watch::Receiver<()>,
        gone: watch::Sender<bool>,
    ) {
        let result = tokio::select! {
            res = child.wait() => res,
            _ = stop_rx.changed() => {
                // Stop requested, or the sender was dropped with the
                // supervisor. SIGTERM first — tts-server traps it into a
                // graceful svr.stop() — SIGKILL only after TERM_GRACE.
                stop_child(&mut child).await
            }
        };
        if let Some(sup) = sup.upgrade() {
            sup.on_child_exit(generation, result);
        }
        // Release every waiter on this generation's reap (a draining leader,
        // `shutdown`). `send_replace` stores the value even with no receiver
        // left, so a waiter that subscribes late still sees it.
        gone.send_replace(true);
    }

    /// Records a child exit. `stop_reason` decides stop vs crash.
    fn on_child_exit(&self, generation: u64, result: io::Result<std::process::ExitStatus>) {
        let mut inner = self.inner();
        if inner.generation != generation {
            return;
        }
        inner.stop_tx = None;
        inner.child_gone = None;
        let healthy_for = inner.ready_since.take().map(|t| t.elapsed());
        match inner.stop_reason.take() {
            Some(reason) => {
                inner.state = TtsState::Stopped;
                metrics::gauge!(names::TTS_CHILD_UP).set(0);
                metrics::counter!(names::TTS_CHILD_STOPS, "reason" => reason.as_str()).increment(1);
                if reason == StopReason::StartupTimeout {
                    // A child that never became healthy counts as a failed
                    // start: defer the next one like a crash would.
                    inner.backoff = next_backoff(inner.backoff, None);
                    inner.retry_after = Some(Instant::now() + inner.backoff);
                } else if healthy_for.is_some_and(|d| d >= BACKOFF_RESET_AFTER) {
                    // A long-healthy child proves the backoff cause is gone —
                    // decay it so the next start is not deferred by a stale
                    // failure.
                    inner.backoff = Duration::ZERO;
                    inner.retry_after = None;
                }
                tracing::info!(reason = reason.as_str(), ?result, "tts child stopped");
            }
            None => {
                inner.state = TtsState::Crashed;
                metrics::gauge!(names::TTS_CHILD_UP).set(0);
                // Only a child that had been Ready counts as a crash; a death
                // during startup is the leader's `start_failures{exit}`.
                if healthy_for.is_some() {
                    metrics::counter!(names::TTS_CHILD_RESTARTS).increment(1);
                }
                inner.backoff = next_backoff(inner.backoff, healthy_for);
                inner.retry_after = Some(Instant::now() + inner.backoff);
                tracing::warn!(?result, backoff = ?inner.backoff, "tts child exited unexpectedly");
            }
        }
    }

    /// Poll `GET /health` until 200, the child exits, or the startup timeout.
    /// Each probe is bounded by `min(remaining, PROBE_TIMEOUT)` — a listener
    /// that accepts but never answers must not park the start loop forever.
    async fn wait_ready(&self, generation: u64) -> Result<(), TtsError> {
        let timeout = Duration::from_secs(self.cfg.startup_timeout_secs);
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            let remaining = deadline.saturating_duration_since(now);
            if remaining.is_zero() {
                return Err(TtsError::StartupTimeout(self.cfg.startup_timeout_secs));
            }
            if tokio::time::timeout(remaining.min(PROBE_TIMEOUT), self.probe_health())
                .await
                .unwrap_or(false)
            {
                return Ok(());
            }
            {
                let inner = self.inner();
                match inner.state {
                    TtsState::Starting if inner.generation == generation => {}
                    TtsState::Crashed => {
                        return Err(TtsError::StartupExit("child exited".to_string()));
                    }
                    other => {
                        return Err(TtsError::StartupExit(format!("unexpected state {other:?}")));
                    }
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(TtsError::StartupTimeout(self.cfg.startup_timeout_secs));
            }
            tokio::time::sleep_until((now + HEALTH_POLL).min(deadline)).await;
        }
    }

    /// One `GET /health` probe — true iff the child answers 200.
    async fn probe_health(&self) -> bool {
        let uri = match format!("{}/health", self.base_url()).parse::<hyper::Uri>() {
            Ok(uri) => uri,
            Err(_) => return false,
        };
        match self.http.get(uri).await {
            Ok(res) => {
                let ok = res.status() == hyper::StatusCode::OK;
                // Drain the body so the connection stays reusable.
                let _ = res.into_body().collect().await;
                ok
            }
            Err(_) => false,
        }
    }

    /// One idle-loop tick: stop the child iff it is `Ready`, past the idle
    /// threshold and no in-flight guard is alive.
    fn idle_tick(&self) {
        let idle_secs = self.cfg.idle_stop_secs;
        if idle_secs == 0 || self.cfg.upstream_url.is_some() {
            return;
        }
        let mut inner = self.inner();
        if inner.in_flight > 0 {
            return;
        }
        if inner.state == TtsState::Ready
            && inner.last_activity.elapsed() >= Duration::from_secs(idle_secs)
        {
            inner.state = TtsState::Stopped;
            inner.stop_reason = Some(StopReason::Idle);
            if let Some(tx) = inner.stop_tx.take() {
                let _ = tx.send(());
            }
            tracing::info!(idle_secs, "stopping idle tts child");
        }
    }

    /// Background loop calling `idle_tick` every `tick`. Same shape as
    /// `EvictablePool::spawn_eviction_loop`: skip the immediate first tick and
    /// contain panics so the loop cannot die silently.
    pub fn spawn_idle_loop(self: &Arc<Self>, tick: Duration) -> tokio::task::JoinHandle<()> {
        let sup = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            interval.tick().await;
            loop {
                interval.tick().await;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    sup.idle_tick();
                }));
                if let Err(e) = result {
                    let msg = e
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| e.downcast_ref::<String>().map(|s| s.as_str()))
                        .unwrap_or("unknown panic");
                    tracing::error!("tts idle loop panicked: {msg}");
                }
            }
        })
    }
}

/// RAII marker for one in-flight TTS request. Created via
/// [`TtsSupervisor::guard`]; dropping it releases the slot and refreshes the
/// idle clock.
pub struct TtsGuard {
    sup: Arc<TtsSupervisor>,
}

impl std::fmt::Debug for TtsGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsGuard").finish_non_exhaustive()
    }
}

impl Drop for TtsGuard {
    fn drop(&mut self) {
        let mut inner = self.sup.inner();
        inner.in_flight = inner.in_flight.saturating_sub(1);
        inner.last_activity = Instant::now();
    }
}

/// Forward one child output stream into `tracing` under the `tts_child`
/// target. Lines log at INFO, raised to WARN only for lines reporting an
/// error (`FATAL` or `ERROR`).
async fn forward_output<R: tokio::io::AsyncRead + Unpin>(reader: R, stream: &'static str) {
    use tokio::io::AsyncBufReadExt;
    let mut lines = tokio::io::BufReader::new(reader).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                if line.contains("FATAL") || line.contains("ERROR") {
                    tracing::warn!(target: "tts_child", stream, "{line}");
                } else {
                    tracing::info!(target: "tts_child", stream, "{line}");
                }
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(target: "tts_child", stream, "output read error: {e}");
                break;
            }
        }
    }
}

/// Terminate a child: SIGTERM first so tts-server can shut down gracefully
/// (`svr.stop()`), then SIGKILL after `TERM_GRACE`. Always leaves the child
/// reaped — never a zombie.
async fn stop_child(child: &mut Child) -> io::Result<std::process::ExitStatus> {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
    }
    match tokio::time::timeout(TERM_GRACE, child.wait()).await {
        Ok(res) => res,
        Err(_) => {
            let _ = child.start_kill();
            child.wait().await
        }
    }
}

/// Crash backoff step: 1 s doubling to 30 s. `healthy_for` at least
/// `BACKOFF_RESET_AFTER` proves the failure cause is gone and resets it.
fn next_backoff(current: Duration, healthy_for: Option<Duration>) -> Duration {
    if current.is_zero() || healthy_for.is_some_and(|d| d >= BACKOFF_RESET_AFTER) {
        BACKOFF_INITIAL
    } else {
        (current * 2).min(BACKOFF_MAX)
    }
}

/// Clears `Inner::start_rx` when the start leader exits for any reason —
/// published outcome, failure or panic — so later callers can launch a new
/// generation. If the leader died mid-start it also requests the child's
/// stop so nothing lingers unsupervised.
struct StartCleanup {
    sup: Arc<TtsSupervisor>,
}

impl Drop for StartCleanup {
    fn drop(&mut self) {
        let mut inner = self.sup.inner();
        inner.start_rx = None;
        if inner.state == TtsState::Starting {
            inner.state = TtsState::Stopped;
            if inner.stop_reason.is_none() {
                inner.stop_reason = Some(StopReason::Shutdown);
            }
            if let Some(tx) = inner.stop_tx.take() {
                let _ = tx.send(());
            }
        }
    }
}

/// Runs post-fork in the child before exec. Allocation and locks are unsafe
/// there, so this is raw libc only: raise `oom_score_adj` to 1000 so the kernel
/// OOM killer picks the TTS child before the STT server in the shared cgroup,
/// and lower CPU priority to nice 10. Failures warn on stderr (piped into
/// `tts_child` tracing) and never abort the spawn.
#[cfg(target_os = "linux")]
fn child_pre_exec() -> io::Result<()> {
    const OOM_WARN: &[u8] = b"tts pre_exec: failed to write /proc/self/oom_score_adj\n";
    const PRIO_WARN: &[u8] = b"tts pre_exec: setpriority(10) failed\n";
    unsafe fn warn(msg: &[u8]) {
        unsafe {
            libc::write(libc::STDERR_FILENO, msg.as_ptr().cast(), msg.len());
        }
    }
    unsafe {
        let fd = libc::open(
            c"/proc/self/oom_score_adj".as_ptr(),
            libc::O_WRONLY | libc::O_CLOEXEC,
        );
        if fd >= 0 {
            const ADJ: &[u8] = b"1000";
            let written = libc::write(fd, ADJ.as_ptr().cast(), ADJ.len());
            libc::close(fd);
            if written != ADJ.len() as isize {
                warn(OOM_WARN);
            }
        } else {
            warn(OOM_WARN);
        }
        if libc::setpriority(libc::PRIO_PROCESS, 0, 10) != 0 {
            warn(PRIO_WARN);
        }
    }
    Ok(())
}
