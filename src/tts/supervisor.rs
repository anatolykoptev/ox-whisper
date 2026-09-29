//! Supervisor for the external `tts-server` child process.
//!
//! The TTS engine is an upstream `tts-server` binary (qwentts.cpp) run as a
//! child process inside the same container, listening on 127.0.0.1. It has no
//! auth; ox-whisper is its only client. The supervisor owns the child's
//! lifecycle:
//!
//! - lazy start: nothing spawns at boot; the first [`ensure_ready`] starts the
//!   child and waits for `GET /health` to answer 200
//! - single-flight: concurrent callers share one start, never two children
//! - idle stop: a background loop stops the child after `idle_stop_secs` with
//!   no in-flight [`TtsGuard`] (0 = never)
//! - crash handling: an unrequested exit flips state to `Crashed`; the next
//!   `ensure_ready` restarts with backoff (1 s doubling to 30 s, reset after
//!   60 s of being healthy)
//! - OOM priority: on Linux the child gets `oom_score_adj=1000` and nice 10 so
//!   the kernel OOM killer picks the TTS engine before the STT server in the
//!   shared container cgroup
//!
//! [`ensure_ready`]: TtsSupervisor::ensure_ready

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
use tokio::sync::{Notify, watch};
use tokio::time::Instant;

use crate::config::TtsConfig;
use crate::metrics::names;

/// Lifecycle state of the managed TTS child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsState {
    /// No child is managed (external `TTS_UPSTREAM_URL` configured, or TTS off).
    Disabled,
    /// No child process is running.
    Stopped,
    /// Child spawned, waiting for `/health` to answer 200.
    Starting,
    /// Child answers `GET /health` with 200.
    Ready,
    /// Child exited without a stop request; restart is allowed after backoff.
    Crashed,
}

/// Errors returned by [`TtsSupervisor::ensure_ready`].
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    /// `TTS_ENABLED` is false — the supervisor is not supposed to run a child.
    #[error("tts is disabled")]
    Disabled,
    /// The `tts-server` binary could not be spawned.
    #[error("tts child spawn failed: {0}")]
    Spawn(#[source] io::Error),
    /// The child exited before answering `/health` with 200.
    #[error("tts child exited before becoming healthy: {0}")]
    StartupExit(String),
    /// `/health` did not answer 200 within `startup_timeout_secs`.
    #[error("tts child did not answer /health within {0}s")]
    StartupTimeout(u64),
}

/// Why a running child was asked to stop. Feeds `stops_total{reason}`.
#[derive(Debug, Clone, Copy)]
enum StopReason {
    Idle,
    Shutdown,
}

impl StopReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Shutdown => "shutdown",
        }
    }
}

struct Inner {
    state: TtsState,
    /// Monotonic id per spawned child; stale monitor reports are ignored.
    generation: u64,
    /// Wakes the child monitor to request a stop.
    stop_tx: Option<watch::Sender<()>>,
    /// Fires when this generation's monitor has reaped the child. A fresh Arc
    /// per spawn so a stale notification cannot release a waiter early.
    child_gone: Option<Arc<Notify>>,
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
        }
    }
}

/// Process supervisor for the TTS child. One per process; lives in `AppState`.
pub struct TtsSupervisor {
    cfg: TtsConfig,
    inner: Mutex<Inner>,
    /// Single-flight: serializes start attempts so one start is shared.
    start_lock: tokio::sync::Mutex<()>,
    http: Client<HttpConnector, Empty<Bytes>>,
}

const BACKOFF_INITIAL: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(30);
const BACKOFF_RESET_AFTER: Duration = Duration::from_secs(60);
const HEALTH_POLL: Duration = Duration::from_millis(100);
const STOP_WAIT: Duration = Duration::from_secs(5);

impl TtsSupervisor {
    pub fn new(cfg: TtsConfig) -> Self {
        let http = Client::builder(TokioExecutor::new()).build(HttpConnector::new());
        metrics::gauge!(names::TTS_CHILD_UP).set(0);
        Self {
            cfg,
            inner: Mutex::new(Inner::new()),
            start_lock: tokio::sync::Mutex::new(()),
            http,
        }
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Current lifecycle state. `Disabled` when an external upstream is
    /// configured (no child is ever managed).
    pub fn state(&self) -> TtsState {
        if self.cfg.upstream_url.is_some() {
            TtsState::Disabled
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

    /// Ensure a TTS endpoint is reachable and return its base URL.
    ///
    /// With `TTS_UPSTREAM_URL` set this just returns that URL and never spawns.
    /// Otherwise starts the child if needed and waits for `/health` to answer
    /// 200 (bounded by `startup_timeout_secs`). Concurrent callers share one
    /// start via `start_lock`; a crashed child is restarted after backoff.
    pub async fn ensure_ready(self: &Arc<Self>) -> Result<String, TtsError> {
        if !self.cfg.enabled {
            return Err(TtsError::Disabled);
        }
        if let Some(url) = &self.cfg.upstream_url {
            return Ok(url.clone());
        }
        {
            let mut inner = self.inner();
            if inner.state == TtsState::Ready {
                inner.last_activity = Instant::now();
                return Ok(self.base_url());
            }
        }

        let _permit = self.start_lock.lock().await;

        // If a child is still up or a stop is in flight, wait for it to be
        // fully reaped before spawning a new one on the same port. `child_gone`
        // is cleared by `on_child_exit` once the monitor has reaped; request a
        // stop first when the monitor has not been asked yet. Bounded by a
        // deadline — a wedged child must not hang the caller forever.
        let drain_deadline = Instant::now() + STOP_WAIT;
        loop {
            let pending = {
                let mut inner = self.inner();
                match inner.state {
                    TtsState::Disabled => return Err(TtsError::Disabled),
                    TtsState::Ready => {
                        inner.last_activity = Instant::now();
                        return Ok(self.base_url());
                    }
                    _ => {}
                }
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
                Some((tx, gone)) => {
                    if let Some(tx) = tx {
                        let _ = tx.send(());
                    }
                    let now = Instant::now();
                    if now >= drain_deadline {
                        break;
                    }
                    let _ = tokio::time::timeout(
                        drain_deadline.saturating_duration_since(now),
                        gone.notified(),
                    )
                    .await;
                }
                None => break,
            }
        }

        // Crash backoff: the previous unrequested exit defers the next start.
        let delay = self
            .inner()
            .retry_after
            .map(|t| t.saturating_duration_since(Instant::now()))
            .unwrap_or_default();
        if !delay.is_zero() {
            tracing::info!(?delay, "tts child restart deferred by crash backoff");
            tokio::time::sleep(delay).await;
        }

        let generation = {
            let mut inner = self.inner();
            inner.state = TtsState::Starting;
            inner.generation += 1;
            // Fresh generation must not inherit a stop request recorded for
            // the previous child (drain deadline can leave one behind).
            inner.stop_reason = None;
            inner.generation
        };

        let (stop_tx, stop_rx) = watch::channel(());
        let child = match self.spawn_child() {
            Ok(child) => child,
            Err(e) => {
                self.inner().state = TtsState::Stopped;
                return Err(e);
            }
        };
        let gone = Arc::new(Notify::new());
        {
            let mut inner = self.inner();
            inner.stop_tx = Some(stop_tx);
            inner.child_gone = Some(Arc::clone(&gone));
        }
        tokio::spawn(Self::monitor(
            Arc::downgrade(self),
            generation,
            child,
            stop_rx,
            gone,
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
                } else {
                    // The monitor beat us to a transition (e.g. the child died
                    // right after a successful probe).
                    Err(TtsError::StartupExit(
                        "child exited during startup".to_string(),
                    ))
                }
            }
            Err(e) => {
                let mut inner = self.inner();
                if inner.state == TtsState::Starting && inner.generation == generation {
                    // Timeout: abort the spawn — request a shutdown stop so the
                    // child cannot linger without supervision.
                    inner.state = TtsState::Stopped;
                    inner.stop_reason = Some(StopReason::Shutdown);
                    if let Some(tx) = inner.stop_tx.take() {
                        let _ = tx.send(());
                    }
                }
                Err(e)
            }
        }
    }

    /// Ask for the running child to be stopped (e.g. server shutdown).
    /// Non-blocking: the monitor kills and reaps the child asynchronously.
    pub fn shutdown(&self) {
        self.request_stop(StopReason::Shutdown);
    }

    /// Request a stop of the current child, if any is running.
    fn request_stop(&self, reason: StopReason) {
        let mut inner = self.inner();
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
            .map_err(TtsError::Spawn)?;
        metrics::counter!(names::TTS_CHILD_STARTS).increment(1);
        tracing::info!(bin = %self.cfg.bin, port = self.cfg.port, "tts child spawned");
        // Pipe the child's output into `tts_child` tracing. Detached: the
        // tasks end by themselves when the pipes close at child exit.
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(forward_output(stdout, false));
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(forward_output(stderr, true));
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
        gone: Arc<Notify>,
    ) {
        let result = tokio::select! {
            res = child.wait() => res,
            _ = stop_rx.changed() => {
                // Stop requested, or the sender was dropped with the
                // supervisor. Kill and reap — never leave a zombie.
                let _ = child.start_kill();
                child.wait().await
            }
        };
        if let Some(sup) = sup.upgrade() {
            sup.on_child_exit(generation, result);
        }
        // Release any ensure_ready waiter draining this generation. Only fires
        // on this generation's Notify — stale permits cannot leak forward.
        gone.notify_one();
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
                tracing::info!(reason = reason.as_str(), ?result, "tts child stopped");
            }
            None => {
                inner.state = TtsState::Crashed;
                metrics::gauge!(names::TTS_CHILD_UP).set(0);
                metrics::counter!(names::TTS_CHILD_RESTARTS).increment(1);
                let backoff = if inner.backoff.is_zero()
                    || healthy_for.is_some_and(|d| d >= BACKOFF_RESET_AFTER)
                {
                    BACKOFF_INITIAL
                } else {
                    (inner.backoff * 2).min(BACKOFF_MAX)
                };
                inner.backoff = backoff;
                inner.retry_after = Some(Instant::now() + backoff);
                tracing::warn!(?result, ?backoff, "tts child exited unexpectedly");
            }
        }
    }

    /// Poll `GET /health` until 200, the child exits, or the startup timeout.
    async fn wait_ready(&self, generation: u64) -> Result<(), TtsError> {
        let timeout = Duration::from_secs(self.cfg.startup_timeout_secs);
        let deadline = Instant::now() + timeout;
        loop {
            if self.probe_health().await {
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

impl Drop for TtsGuard {
    fn drop(&mut self) {
        let mut inner = self.sup.inner();
        inner.in_flight = inner.in_flight.saturating_sub(1);
        inner.last_activity = Instant::now();
    }
}

/// Forward one child output stream into `tracing` under the `tts_child`
/// target; stderr is logged at warn level, stdout at info.
async fn forward_output<R: tokio::io::AsyncRead + Unpin>(reader: R, is_err: bool) {
    use tokio::io::AsyncBufReadExt;
    let mut lines = tokio::io::BufReader::new(reader).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                if is_err {
                    tracing::warn!(target: "tts_child", "{line}");
                } else {
                    tracing::info!(target: "tts_child", "{line}");
                }
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(target: "tts_child", "output read error: {e}");
                break;
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
