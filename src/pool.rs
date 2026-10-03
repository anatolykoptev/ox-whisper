use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How long [`EvictablePool::acquire`] waits for a busy slot by default.
pub const DEFAULT_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(30);

// ── EvictablePool ─────────────────────────────────────────────────────────────

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Errors returned by [`EvictablePool::acquire`].
#[derive(Debug, thiserror::Error)]
pub enum AcquireError {
    /// All slots stayed in use for the whole acquire timeout.
    #[error("all pool slots are busy")]
    AllBusy,
    /// Factory returned an error when reinitialising an evicted slot.
    #[error("pool slot reinit failed: {0}")]
    ReinitFailed(#[from] anyhow::Error),
}

/// A single slot in an EvictablePool.
///
/// State machine:
/// - `item = Some(T)` + `busy = false`  → idle, ready to acquire
/// - `item = None`   + `busy = false`   → evicted, will be reinit on next acquire
/// - `item = None`   + `busy = true`    → held by a guard (in use)
struct EvictableSlot<T> {
    item: Mutex<Option<T>>,
    busy: std::sync::atomic::AtomicBool,
    last_used: AtomicU64,
    /// The last attempt to rebuild this evicted slot failed. Per slot, not per
    /// pool: one slot's success must not hide another's failure, and the
    /// background retry needs to know which slots to try.
    reload_failed: std::sync::atomic::AtomicBool,
}

/// Pool with opt-in idle eviction. Items are lazily re-created via `factory`
/// when a slot is evicted and then acquired again.
///
/// When `idle_secs == 0` eviction is **disabled** and the pool behaves like
/// a simple pre-allocated pool.
pub struct EvictablePool<T> {
    slots: Vec<Arc<EvictableSlot<T>>>,
    factory: Arc<dyn Fn() -> Result<T, anyhow::Error> + Send + Sync>,
    idle_secs: u64,
    /// Release generation + condvar: a guard drop bumps the counter and wakes
    /// waiters. The counter closes the race between a failed try and the wait.
    released: Arc<(Mutex<u64>, Condvar)>,
    acquire_timeout: Duration,
}

impl<T: Send + 'static> EvictablePool<T> {
    /// Create a pool with `size` slots pre-filled via `factory`.
    /// `idle_secs == 0` disables eviction.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(
        size: usize,
        idle_secs: u64,
        factory: Arc<dyn Fn() -> Result<T, anyhow::Error> + Send + Sync>,
    ) -> Self {
        let now = unix_now_secs();
        let slots = (0..size)
            .map(|_| {
                // Unwrap is safe at startup — failure here is a configuration error.
                let item = (factory)().expect("pool factory failed during initialisation");
                Arc::new(EvictableSlot {
                    item: Mutex::new(Some(item)),
                    busy: std::sync::atomic::AtomicBool::new(false),
                    last_used: AtomicU64::new(now),
                    reload_failed: std::sync::atomic::AtomicBool::new(false),
                })
            })
            .collect();
        Self {
            slots,
            factory,
            idle_secs,
            released: Arc::new((Mutex::new(0), Condvar::new())),
            acquire_timeout: DEFAULT_ACQUIRE_TIMEOUT,
        }
    }

    /// Create a pool pre-filled with already-constructed items.
    ///
    /// `factory` is used only for lazy re-init after eviction.
    /// `idle_secs == 0` disables eviction.
    pub fn from_items(
        items: Vec<T>,
        idle_secs: u64,
        factory: Arc<dyn Fn() -> Result<T, anyhow::Error> + Send + Sync>,
    ) -> Self {
        let now = unix_now_secs();
        let slots = items
            .into_iter()
            .map(|item| {
                Arc::new(EvictableSlot {
                    item: Mutex::new(Some(item)),
                    busy: std::sync::atomic::AtomicBool::new(false),
                    last_used: AtomicU64::new(now),
                    reload_failed: std::sync::atomic::AtomicBool::new(false),
                })
            })
            .collect();
        Self {
            slots,
            factory,
            idle_secs,
            released: Arc::new((Mutex::new(0), Condvar::new())),
            acquire_timeout: DEFAULT_ACQUIRE_TIMEOUT,
        }
    }

    /// False when every slot's last rebuild failed: no request can be served
    /// until one succeeds, and `/health` reports it. Derived from slot state, so
    /// it cannot outlive the failures it describes: a slot that is loaded, in
    /// use, or merely evicted (it reloads on demand) counts as able to serve,
    /// and the eviction loop retries the failed ones on its own.
    pub fn is_healthy(&self) -> bool {
        self.slots.is_empty()
            || self
                .slots
                .iter()
                .any(|s| !s.reload_failed.load(Ordering::Acquire))
    }

    /// Sets how long [`Self::acquire`] waits for a busy slot.
    pub fn with_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }

    /// Acquire a slot, waiting up to the acquire timeout while all are busy.
    /// Blocks the calling thread: call it from blocking code, not an async task.
    ///
    /// Returns `Err(AcquireError::AllBusy)` if no slot freed up in time.
    /// Returns `Err(AcquireError::ReinitFailed)` as [`Self::try_acquire`] does.
    pub fn acquire(&self) -> Result<EvictableGuard<T>, AcquireError> {
        let start = Instant::now();
        let deadline = start + self.acquire_timeout;
        let (lock, cvar) = &*self.released;
        loop {
            let mut seen = *lock.lock().unwrap_or_else(|p| p.into_inner());
            match self.try_acquire() {
                Err(AcquireError::AllBusy) => {}
                // One slot failed to reload but another is in use: it will be
                // released (and `/health` says the pool can serve), so keep
                // waiting for it rather than failing the request now.
                Err(AcquireError::ReinitFailed(_)) if self.any_busy() => {
                    // The failed attempt notified waiters, this thread
                    // included: take the generation again so that wake-up
                    // does not turn the wait into a retry loop. A release
                    // landing before the re-read leaves a slot idle, which
                    // the check below catches.
                    seen = *lock.lock().unwrap_or_else(|p| p.into_inner());
                    if self.has_idle_item() {
                        continue;
                    }
                }
                other => {
                    if other.is_ok() && start.elapsed() > Duration::from_millis(1) {
                        metrics::histogram!(crate::metrics::names::POOL_ACQUIRE_WAIT)
                            .record(start.elapsed().as_secs_f64());
                    }
                    return other;
                }
            }
            let mut generation = lock.lock().unwrap_or_else(|p| p.into_inner());
            while *generation == seen {
                let now = Instant::now();
                if now >= deadline {
                    metrics::counter!(crate::metrics::names::POOL_ACQUIRE_TIMEOUTS).increment(1);
                    tracing::warn!("pool acquire timed out after {:?}", self.acquire_timeout);
                    return Err(AcquireError::AllBusy);
                }
                generation = cvar
                    .wait_timeout(generation, deadline - now)
                    .unwrap_or_else(|p| p.into_inner())
                    .0;
            }
        }
    }

    /// Acquire an idle slot without waiting. Re-initializes evicted slots via
    /// factory (cold start).
    ///
    /// Returns `Err(AcquireError::AllBusy)` if all slots are in use.
    /// Returns `Err(AcquireError::ReinitFailed)` if an evicted slot's factory call fails
    /// and no other slot could serve; the failed slot is left as `None` (not permanently
    /// dead — the next acquire and the eviction loop retry it).
    pub fn try_acquire(&self) -> Result<EvictableGuard<T>, AcquireError> {
        let now = unix_now_secs();
        let mut reinit_error = None;
        for slot in &self.slots {
            // Skip slots that are already in use.
            if slot.busy.load(Ordering::Acquire) {
                continue;
            }

            // ── M4: factory called OUTSIDE the mutex ─────────────────────────
            // Take lock, re-check busy, and claim the slot before releasing the
            // mutex. Setting busy=true while the lock is held prevents another
            // acquire from racing on the same evicted slot and avoids evict_idle
            // sniping the item between the unlock and the factory call.
            let mut guard = match slot.item.lock() {
                Ok(g) => g,
                Err(poisoned) => {
                    tracing::warn!("pool mutex poisoned — recovering inner value");
                    metrics::counter!(crate::metrics::names::POOL_MUTEX_POISONED).increment(1);
                    poisoned.into_inner()
                }
            };
            if slot.busy.load(Ordering::Acquire) {
                continue;
            }

            if guard.is_none() {
                // Claim this slot for reinit and release the lock before the
                // (potentially slow) factory call.
                slot.busy.store(true, Ordering::Release);
                drop(guard);

                let new_item = match self.reinit(slot) {
                    Ok(item) => item,
                    Err(e) => {
                        // Leave slot as None; clear busy so next acquire can retry.
                        slot.busy.store(false, Ordering::Release);
                        notify_released(&self.released);
                        // Another slot may still serve: health says the pool
                        // can serve while any slot is not failing.
                        reinit_error = Some(e);
                        continue;
                    }
                };

                slot.last_used.store(now, Ordering::Relaxed);
                slot.reload_failed.store(false, Ordering::Release);
                return Ok(EvictableGuard {
                    slot: Arc::clone(slot),
                    item: Some(new_item),
                    released: Arc::clone(&self.released),
                });
            }

            // Slot has an item — claim it under lock.
            // ── B1: set busy BEFORE releasing MutexGuard ─────────────────────
            // This prevents a race window between mutex unlock and busy.store.
            slot.busy.store(true, Ordering::Release);
            slot.last_used.store(now, Ordering::Relaxed);
            let item = guard
                .take()
                .expect("BUG: slot item missing after lock acquired");
            // guard drops here (MutexGuard released) — busy is already true.
            return Ok(EvictableGuard {
                slot: Arc::clone(slot),
                item: Some(item),
                released: Arc::clone(&self.released),
            });
        }
        match reinit_error {
            Some(e) => Err(AcquireError::ReinitFailed(e)),
            None => Err(AcquireError::AllBusy),
        }
    }

    /// Rebuilds an evicted slot through the factory, outside the slot mutex.
    /// The caller has claimed the slot (`busy`) and holds no lock. Records the
    /// failure on the slot (`reload_failed`); the caller clears it once the
    /// rebuilt item is stored, so a slot never reads healthy while empty.
    fn reinit(&self, slot: &EvictableSlot<T>) -> Result<T, anyhow::Error> {
        metrics::counter!(crate::metrics::names::POOL_COLD_STARTS).increment(1);
        tracing::info!("pool cold start: reinitializing evicted slot");
        match (self.factory)() {
            Ok(item) => Ok(item),
            Err(e) => {
                tracing::error!("pool slot reinit failed: {e}");
                metrics::counter!(crate::metrics::names::POOL_REINIT_FAILURES).increment(1);
                slot.reload_failed.store(true, Ordering::Release);
                Err(e)
            }
        }
    }

    /// Retries every slot whose last rebuild failed, without waiting for a
    /// request: a front end that stops sending traffic while `/health` is 503
    /// would otherwise never give the pool the chance to recover. Blocks for
    /// the duration of a model load: call it from blocking code. Returns how
    /// many slots recovered.
    pub fn retry_failed(&self) -> usize {
        let mut recovered = 0;
        for slot in &self.slots {
            if !slot.reload_failed.load(Ordering::Acquire) {
                continue;
            }
            // Claim like `try_acquire` does, so a request cannot rebuild the
            // same slot at the same time.
            let guard = match slot.item.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if slot.busy.load(Ordering::Acquire) || guard.is_some() {
                continue;
            }
            slot.busy.store(true, Ordering::Release);
            drop(guard);

            let rebuilt = self.reinit(slot);
            let mut guard = match slot.item.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let Ok(item) = rebuilt {
                *guard = Some(item);
                // Item first, flag second: healthy implies a loaded slot.
                slot.reload_failed.store(false, Ordering::Release);
                // Fresh stamp: the next idle tick must not evict it again.
                slot.last_used.store(unix_now_secs(), Ordering::Relaxed);
                recovered += 1;
            }
            slot.busy.store(false, Ordering::Release);
            drop(guard);
            notify_released(&self.released);
        }
        recovered
    }

    /// Evict slots idle longer than `threshold_secs`. Returns count evicted.
    /// When `self.idle_secs == 0`, does nothing and returns 0.
    pub fn evict_idle(&self, threshold_secs: u64) -> usize {
        if self.idle_secs == 0 {
            return 0;
        }
        let now = unix_now_secs();
        let mut count = 0;
        for slot in &self.slots {
            // Never evict a busy slot.
            if slot.busy.load(Ordering::Acquire) {
                continue;
            }
            let age = now.saturating_sub(slot.last_used.load(Ordering::Relaxed));
            if age >= threshold_secs
                && let Ok(mut guard) = slot.item.lock()
                && guard.is_some()
                && !slot.busy.load(Ordering::Acquire)
            {
                *guard = None;
                count += 1;
                metrics::counter!(crate::metrics::names::POOL_EVICTIONS).increment(1);
                tracing::info!("pool evicted idle slot (age={}s)", age);
            }
        }
        count
    }

    /// Spawn a background tokio task that calls `evict_idle` every `tick` interval.
    ///
    /// The returned `JoinHandle` can be aborted to stop the loop.
    /// When `self.idle_secs == 0`, the loop still spawns but `evict_idle` is a
    /// no-op — callers should gate on `idle_secs > 0` before calling this.
    pub fn spawn_eviction_loop(
        self: &std::sync::Arc<Self>,
        tick: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        let pool = std::sync::Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            interval.tick().await; // skip the immediate first tick
            loop {
                interval.tick().await;
                let threshold = pool.idle_secs;
                // Catch panics in evict_idle to avoid silently killing the loop.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pool.evict_idle(threshold);
                }));
                // Slots whose rebuild failed are retried here, each tick, off
                // the async worker: a rebuild loads a multi-GB model.
                if pool.has_failed_slots() {
                    let p = std::sync::Arc::clone(&pool);
                    let _ = tokio::task::spawn_blocking(move || p.retry_failed()).await;
                }
                if let Err(e) = result {
                    let msg = e
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| e.downcast_ref::<String>().map(|s| s.as_str()))
                        .unwrap_or("unknown panic");
                    tracing::error!("eviction loop panicked: {msg}");
                    metrics::counter!(crate::metrics::names::POOL_EVICTION_LOOP_PANICS)
                        .increment(1);
                }
            }
        })
    }

    /// A loaded slot nobody is using.
    fn has_idle_item(&self) -> bool {
        self.slots.iter().any(|s| {
            !s.busy.load(Ordering::Acquire)
                && s.item.lock().unwrap_or_else(|p| p.into_inner()).is_some()
        })
    }

    fn any_busy(&self) -> bool {
        self.slots.iter().any(|s| s.busy.load(Ordering::Acquire))
    }

    fn has_failed_slots(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.reload_failed.load(Ordering::Acquire))
    }

    /// Test helper: a pool of `size` evicted slots (nothing loaded yet).
    #[cfg(test)]
    pub fn evicted(
        size: usize,
        idle_secs: u64,
        factory: Arc<dyn Fn() -> Result<T, anyhow::Error> + Send + Sync>,
    ) -> Self {
        let now = unix_now_secs();
        let slots = (0..size)
            .map(|_| {
                Arc::new(EvictableSlot {
                    item: Mutex::new(None),
                    busy: std::sync::atomic::AtomicBool::new(false),
                    last_used: AtomicU64::new(now),
                    reload_failed: std::sync::atomic::AtomicBool::new(false),
                })
            })
            .collect();
        Self {
            slots,
            factory,
            idle_secs,
            released: Arc::new((Mutex::new(0), Condvar::new())),
            acquire_timeout: DEFAULT_ACQUIRE_TIMEOUT,
        }
    }

    /// Test helper: push all slots' last_used `secs` seconds into the past.
    #[cfg(test)]
    pub fn force_last_used_ago(&self, secs: u64) {
        let past = unix_now_secs().saturating_sub(secs);
        for slot in &self.slots {
            slot.last_used.store(past, Ordering::Relaxed);
        }
    }
}

/// Wakes acquirers waiting for a slot.
fn notify_released(released: &(Mutex<u64>, Condvar)) {
    let (lock, cvar) = released;
    let mut generation = lock.lock().unwrap_or_else(|p| p.into_inner());
    *generation = generation.wrapping_add(1);
    cvar.notify_all();
}

/// RAII guard that returns the item to its slot on drop and refreshes `last_used`.
pub struct EvictableGuard<T> {
    slot: Arc<EvictableSlot<T>>,
    item: Option<T>,
    released: Arc<(Mutex<u64>, Condvar)>,
}

impl<T> std::ops::Deref for EvictableGuard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.item.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for EvictableGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.item.as_mut().unwrap()
    }
}

impl<T> Drop for EvictableGuard<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let now = unix_now_secs();
            if let Ok(mut guard) = self.slot.item.lock() {
                self.slot.last_used.store(now, Ordering::Relaxed);
                *guard = Some(item);
                // ── B1: busy.store INSIDE the MutexGuard scope ───────────────
                // Publishing busy=false while still holding the lock eliminates the
                // race window that existed when busy.store fired after unlock.
                self.slot.busy.store(false, Ordering::Release);
            } else {
                // Mutex poisoned — we can't return the item safely, but at least
                // clear busy so the slot isn't permanently stuck.
                self.slot.busy.store(false, Ordering::Release);
                tracing::error!("pool mutex poisoned on guard drop — item lost");
            }
            notify_released(&self.released);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AOrdering};

    fn make_pool(size: usize, idle_secs: u64) -> EvictablePool<u32> {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        EvictablePool::new(
            size,
            idle_secs,
            Arc::new(move || {
                c.fetch_add(1, AOrdering::SeqCst);
                Ok(42u32)
            }),
        )
    }

    // ── 1. eviction_disabled_by_default ─────────────────────────────────────
    #[test]
    fn eviction_disabled_by_default() {
        let pool = make_pool(2, 0);
        {
            let _guard = pool.acquire().expect("should acquire");
        }
        // eviction disabled → evict_idle returns 0 regardless of threshold
        assert_eq!(pool.evict_idle(1), 0, "eviction disabled => no evictions");
    }

    // ── 2. eviction_after_idle_threshold ────────────────────────────────────
    #[test]
    fn eviction_after_idle_threshold() {
        let pool = make_pool(2, 1);
        {
            let _guard = pool.acquire().expect("should acquire");
        }
        pool.force_last_used_ago(2);

        let evicted = pool.evict_idle(1);
        assert_eq!(
            evicted, 2,
            "both idle slots should be evicted after threshold"
        );
    }

    // ── 3. lazy_reinit_after_eviction ───────────────────────────────────────
    #[test]
    fn lazy_reinit_after_eviction() {
        let pool = make_pool(1, 1);
        {
            let _guard = pool.acquire().expect("acquire 1");
        }
        pool.force_last_used_ago(5);
        let evicted = pool.evict_idle(1);
        assert_eq!(evicted, 1, "one slot evicted");

        let guard = pool.acquire().expect("reinit acquire");
        assert_eq!(*guard, 42u32, "reinit should produce factory value");
    }

    // ── 4. last_used_updated_on_acquire ─────────────────────────────────────
    #[test]
    fn last_used_updated_on_acquire() {
        let pool = make_pool(2, 10);
        pool.force_last_used_ago(100);

        {
            let _guard = pool.acquire().expect("acquire");
        }

        // One slot was recently used, the other still has old timestamp.
        // evict with threshold=10: the one acquired recently should survive,
        // the untouched one should be evicted.
        let evicted = pool.evict_idle(10);
        assert_eq!(
            evicted, 1,
            "only stale slot evicted; recently used slot survives"
        );
    }

    // ── 5. metric_eviction_counter_increments ───────────────────────────────
    #[test]
    fn metric_eviction_counter_increments() {
        let pool = make_pool(3, 1);
        {
            let _g1 = pool.acquire().expect("acquire 1");
            let _g2 = pool.acquire().expect("acquire 2");
            let _g3 = pool.acquire().expect("acquire 3");
        }
        pool.force_last_used_ago(5);
        let evicted = pool.evict_idle(1);
        assert_eq!(evicted, 3, "all 3 slots evicted");
    }

    // ── 6. eviction_loop_runs_and_evicts ────────────────────────────────────
    /// Verify that `spawn_eviction_loop` actually calls `evict_idle` periodically.
    /// Uses an Arc<EvictablePool<u32>> with idle_secs=1, forces last_used 5s into
    /// the past, then lets the loop fire and checks that a slot is evicted.
    #[tokio::test]
    async fn eviction_loop_runs_and_evicts() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicUsize, Ordering as AOrdering2};

        // Pool with 1 slot, idle_secs=1 so eviction is enabled.
        let evict_counter = StdArc::new(AtomicUsize::new(0));
        let ec = evict_counter.clone();
        let pool = StdArc::new(EvictablePool::new(
            1,
            1,
            StdArc::new(move || {
                ec.fetch_add(1, AOrdering2::SeqCst);
                Ok(99u32)
            }),
        ));

        // Acquire + release to mark it idle, then wind back clock.
        {
            let _guard = pool.acquire().expect("acquire for loop test");
        }
        pool.force_last_used_ago(5);

        // Spawn loop with 200ms tick — fast enough for test, but won't fire
        // spuriously in normal runs.
        let handle = pool.spawn_eviction_loop(std::time::Duration::from_millis(200));

        // Wait 500ms — loop should fire at least twice in that window.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        handle.abort();

        // The slot should have been evicted (set to None).
        let guard = pool.slots[0].item.lock().unwrap();
        assert!(guard.is_none(), "slot should be evicted after loop ran");
    }

    // ── 7. eviction_loop_no_eviction_when_idle_secs_zero ────────────────────
    /// When idle_secs==0, eviction is disabled; loop must not evict.
    #[tokio::test]
    async fn eviction_loop_no_eviction_when_idle_secs_zero() {
        use std::sync::Arc as StdArc;

        let pool = StdArc::new(make_pool(1, 0)); // idle_secs=0 → disabled
        {
            let _guard = pool.acquire().expect("acquire");
        }
        pool.force_last_used_ago(9999);

        let handle = pool.spawn_eviction_loop(std::time::Duration::from_millis(100));
        tokio::time::sleep(std::time::Duration::from_millis(350)).await;
        handle.abort();

        // Slot should still be Some — no eviction.
        let guard = pool.slots[0].item.lock().unwrap();
        assert!(
            guard.is_some(),
            "idle_secs=0 → no eviction even with stale last_used"
        );
    }

    /// A pool whose evicted slot cannot be rebuilt must say so, and recover
    /// the moment a rebuild succeeds.
    #[test]
    fn a_failing_reinit_marks_the_pool_unhealthy_until_one_succeeds() {
        use std::sync::atomic::AtomicBool;
        let broken = Arc::new(AtomicBool::new(true));
        let b = broken.clone();
        let pool = EvictablePool::from_items(
            vec![0u32],
            1,
            Arc::new(move || {
                if b.load(Ordering::SeqCst) {
                    Err(anyhow::anyhow!("out of memory"))
                } else {
                    Ok(7u32)
                }
            }),
        );
        assert!(pool.is_healthy(), "a fresh pool is healthy");

        pool.force_last_used_ago(10);
        pool.evict_idle(1);
        assert!(pool.is_healthy(), "an evicted slot reloads on demand");
        assert!(pool.acquire().is_err());
        assert!(!pool.is_healthy(), "a failed reinit must be visible");

        broken.store(false, Ordering::SeqCst);
        drop(pool.acquire().expect("reinit succeeds now"));
        assert!(pool.is_healthy(), "a successful reinit clears it");
    }

    /// With two slots, one failing rebuild must not hide that the other can
    /// serve, and one success must not hide that the other still fails.
    #[test]
    fn health_follows_every_slot_not_the_last_reinit() {
        use std::sync::atomic::AtomicUsize;
        // First rebuild fails, later ones succeed.
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let pool = EvictablePool::evicted(
            2,
            1,
            Arc::new(move || {
                if c.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(anyhow::anyhow!("out of memory"))
                } else {
                    Ok(1u32)
                }
            }),
        );
        // Slot 0 fails, the request falls through to slot 1 and is served.
        let guard = pool.try_acquire().expect("slot 1 serves the request");
        assert!(pool.is_healthy(), "slot 1 is fine");
        assert!(pool.slots[0].reload_failed.load(Ordering::SeqCst));
        drop(guard);
        // Fail every rebuild now, evict, and exhaust both slots.
        let pool =
            EvictablePool::<u32>::evicted(2, 1, Arc::new(|| Err(anyhow::anyhow!("no memory"))));
        assert!(pool.try_acquire().is_err());
        assert!(!pool.is_healthy(), "every slot failed");
    }

    /// The latch of #65: every slot failed, nothing sends a request, and the
    /// pool must still recover once a rebuild can succeed. Driven through the
    /// eviction loop alone — no `acquire` after the failure.
    #[tokio::test]
    async fn the_eviction_loop_retries_a_failed_reload_without_a_request() {
        use std::sync::atomic::AtomicBool;
        let broken = Arc::new(AtomicBool::new(true));
        let b = broken.clone();
        let pool = Arc::new(EvictablePool::evicted(
            1,
            1,
            Arc::new(move || {
                if b.load(Ordering::SeqCst) {
                    Err(anyhow::anyhow!("out of memory"))
                } else {
                    Ok(5u32)
                }
            }),
        ));
        assert!(pool.try_acquire().is_err());
        assert!(!pool.is_healthy());

        broken.store(false, Ordering::SeqCst);
        let handle = pool.spawn_eviction_loop(Duration::from_millis(50));
        let deadline = Instant::now() + Duration::from_secs(5);
        while !pool.is_healthy() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        handle.abort();
        assert!(pool.is_healthy(), "no request came, yet the pool recovered");
        assert_eq!(
            *pool.slots[0].item.lock().unwrap(),
            Some(5),
            "the rebuilt item is resident"
        );
    }

    /// A failed reload on one slot must not fail a request while another slot
    /// is merely busy: the request waits for it.
    #[test]
    fn acquire_waits_for_a_busy_slot_when_another_slot_cannot_reload() {
        use std::sync::atomic::AtomicUsize;
        // First rebuild succeeds (slot 0, held below), every later one fails.
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let pool = Arc::new(
            EvictablePool::evicted(
                2,
                1,
                Arc::new(move || {
                    if c.fetch_add(1, Ordering::SeqCst) == 0 {
                        Ok(1u32)
                    } else {
                        Err(anyhow::anyhow!("out of memory"))
                    }
                }),
            )
            .with_acquire_timeout(Duration::from_secs(5)),
        );
        let held = pool.try_acquire().expect("slot 0 loads");
        let p = pool.clone();
        let waiter = std::thread::spawn(move || p.acquire().map(|g| *g));
        std::thread::sleep(Duration::from_millis(200));
        assert!(!waiter.is_finished(), "failed instead of waiting");
        assert!(
            calls.load(Ordering::SeqCst) <= 3,
            "the waiter retried the reload in a loop: {} attempts",
            calls.load(Ordering::SeqCst)
        );
        drop(held);
        assert_eq!(waiter.join().unwrap().expect("served by slot 0"), 1);
    }

    // ── B2. factory_error_returns_err_and_slot_stays_alive ──────────────────
    /// When factory returns Err, acquire() must return Err (not panic),
    /// log the error, and leave the slot as None so next acquire retries.
    #[test]
    fn factory_error_returns_err_and_slot_stays_alive() {
        use std::sync::atomic::{AtomicUsize, Ordering as AO};
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        // Factory fails on first reinit call, succeeds on second.
        // Use from_items so pool starts pre-filled (factory not called at init).
        let pool = Arc::new(EvictablePool::from_items(
            vec![0u32],
            1,
            Arc::new(move || {
                let n = cc.fetch_add(1, AO::SeqCst);
                if n == 0 {
                    Err(anyhow::anyhow!("factory fail"))
                } else {
                    Ok(99u32)
                }
            }),
        ));

        // Force evict the pre-filled slot so next acquire calls factory.
        pool.force_last_used_ago(10);
        pool.evict_idle(1);
        assert!(
            pool.slots[0].item.lock().unwrap().is_none(),
            "slot must be evicted"
        );

        // First acquire: factory fails → Err returned.
        let result = pool.acquire();
        assert!(
            result.is_err(),
            "acquire must return Err when factory fails"
        );

        // Slot must still be None — not permanently dead; busy must be false.
        assert!(
            pool.slots[0].item.lock().unwrap().is_none(),
            "slot stays None after reinit failure"
        );
        assert!(
            !pool.slots[0].busy.load(Ordering::Acquire),
            "busy must be cleared after reinit failure"
        );

        // Second acquire: factory succeeds → Ok returned.
        let guard = pool
            .acquire()
            .expect("second acquire must succeed after factory recovers");
        assert_eq!(*guard, 99u32);
    }

    // ── B1. drop_ordering_no_race ────────────────────────────────────────────
    /// Stress test: N concurrent acquire+drop cycles.
    /// Must not panic or permanently lose items.
    #[test]
    fn drop_ordering_no_race() {
        use std::sync::Arc as StdArc;

        let pool = StdArc::new(EvictablePool::new(
            2,
            0, // eviction disabled; we test drop ordering directly
            StdArc::new(|| Ok(42u32)),
        ));

        let mut handles = Vec::new();

        for _ in 0..8 {
            let p = pool.clone();
            let handle = std::thread::spawn(move || {
                for _ in 0..1000 {
                    // AllBusy is fine; panic or item loss is not.
                    if let Ok(guard) = p.acquire() {
                        drop(guard);
                    }
                }
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().expect("thread must not panic");
        }

        // After all threads done, both slots must be returned (Some).
        for slot in &pool.slots {
            assert!(
                slot.item.lock().unwrap().is_some(),
                "slot item must be returned after all guards dropped"
            );
            assert!(
                !slot.busy.load(AOrdering::Acquire),
                "busy must be false after all guards dropped"
            );
        }
    }

    // ── M4. factory_called_outside_lock ─────────────────────────────────────
    /// A slow factory must not block another acquire from checking the slot.
    /// We verify that two concurrent acquires on a 2-slot pool can both proceed
    /// even when one slot is reinitializing (factory takes time).
    #[test]
    fn factory_called_outside_lock() {
        use std::sync::Arc as StdArc;
        use std::time::Duration;

        // Factory signals it's running then sleeps.
        let pool = StdArc::new(EvictablePool::new(
            2,
            1,
            StdArc::new(|| {
                std::thread::sleep(Duration::from_millis(50));
                Ok(77u32)
            }),
        ));

        // Evict both slots.
        pool.force_last_used_ago(10);
        pool.evict_idle(1);
        for slot in &pool.slots {
            assert!(slot.item.lock().unwrap().is_none());
        }

        // Two threads: one acquires (triggers slow factory), other tries simultaneously.
        let p1 = pool.clone();
        let p2 = pool.clone();
        let t1 = std::thread::spawn(move || p1.acquire());
        std::thread::sleep(Duration::from_millis(5)); // let t1 start factory
        let t2 = std::thread::spawn(move || p2.acquire());

        let r1 = t1.join().expect("t1 no panic");
        let r2 = t2.join().expect("t2 no panic");

        // At least one must succeed; both should not deadlock.
        let successes = r1.is_ok() as usize + r2.is_ok() as usize;
        assert!(
            successes >= 1,
            "at least one acquire must succeed with 2-slot pool"
        );
    }

    // ── M1. tick_interval_is_quarter_of_idle_secs ───────────────────────────
    /// Verify the tick calculation: tick = max(idle_secs/4, 5s).
    #[test]
    fn tick_interval_quarter_of_idle_secs() {
        use std::time::Duration;
        // This is a pure logic test of the formula used in Models::load.
        let compute_tick = |idle_secs: u64| -> Duration {
            let quarter = Duration::from_secs(idle_secs / 4);
            quarter.max(Duration::from_secs(5))
        };

        assert_eq!(compute_tick(120), Duration::from_secs(30));
        assert_eq!(compute_tick(40), Duration::from_secs(10));
        assert_eq!(
            compute_tick(16),
            Duration::from_secs(5),
            "minimum 5s applies"
        );
        assert_eq!(
            compute_tick(4),
            Duration::from_secs(5),
            "aggressive threshold: 1s → 5s minimum"
        );
    }

    // ── bounded wait ────────────────────────────────────────────────────────
    /// With every slot busy, acquire waits for a release instead of failing.
    #[test]
    fn acquire_waits_for_a_released_slot() {
        let pool = Arc::new(make_pool(1, 0).with_acquire_timeout(Duration::from_secs(5)));
        let held = pool.acquire().expect("first acquire");
        let p = pool.clone();
        let waiter = std::thread::spawn(move || p.acquire().map(|g| *g));
        std::thread::sleep(Duration::from_millis(200));
        drop(held);
        assert_eq!(waiter.join().unwrap().expect("second acquire waits"), 42);
    }

    /// The wait is bounded: a slot that never frees up gives AllBusy.
    #[test]
    fn acquire_gives_up_after_the_timeout() {
        let pool = make_pool(1, 0).with_acquire_timeout(Duration::from_millis(150));
        let _held = pool.acquire().expect("first acquire");
        let t = Instant::now();
        assert!(matches!(pool.acquire(), Err(AcquireError::AllBusy)));
        assert!(t.elapsed() >= Duration::from_millis(150));
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
        assert!(matches!(pool.try_acquire(), Err(AcquireError::AllBusy)));
    }
}
