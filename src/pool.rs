use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ── EvictablePool ─────────────────────────────────────────────────────────────

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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
}

/// Pool with opt-in idle eviction. Items are lazily re-created via `factory`
/// when a slot is evicted and then acquired again.
///
/// When `idle_secs == 0` eviction is **disabled** and the pool behaves like
/// a simple pre-allocated pool.
pub struct EvictablePool<T> {
    slots: Vec<Arc<EvictableSlot<T>>>,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    idle_secs: u64,
}

impl<T: Send + 'static> EvictablePool<T> {
    /// Create a pool with `size` slots pre-filled via `factory`.
    /// `idle_secs == 0` disables eviction.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(
        size: usize,
        idle_secs: u64,
        factory: Arc<dyn Fn() -> T + Send + Sync>,
    ) -> Self {
        let now = unix_now_secs();
        let slots = (0..size)
            .map(|_| {
                Arc::new(EvictableSlot {
                    item: Mutex::new(Some((factory)())),
                    busy: std::sync::atomic::AtomicBool::new(false),
                    last_used: AtomicU64::new(now),
                })
            })
            .collect();
        Self { slots, factory, idle_secs }
    }

    /// Create a pool pre-filled with already-constructed items.
    ///
    /// `factory` is used only for lazy re-init after eviction.
    /// `idle_secs == 0` disables eviction.
    pub fn from_items(
        items: Vec<T>,
        idle_secs: u64,
        factory: Arc<dyn Fn() -> T + Send + Sync>,
    ) -> Self {
        let now = unix_now_secs();
        let slots = items
            .into_iter()
            .map(|item| {
                Arc::new(EvictableSlot {
                    item: Mutex::new(Some(item)),
                    busy: std::sync::atomic::AtomicBool::new(false),
                    last_used: AtomicU64::new(now),
                })
            })
            .collect();
        Self { slots, factory, idle_secs }
    }

    /// Acquire an idle slot. Re-initializes evicted slots via factory (cold start).
    /// Returns `None` if all slots are busy.
    pub fn acquire(&self) -> Option<EvictableGuard<T>> {
        let now = unix_now_secs();
        for slot in &self.slots {
            // Skip slots that are already in use.
            if slot.busy.load(Ordering::Acquire) {
                continue;
            }
            let mut guard = match slot.item.lock() {
                Ok(g) => g,
                Err(_) => continue,
            };
            // Still busy check inside lock to avoid TOCTOU.
            if slot.busy.load(Ordering::Acquire) {
                continue;
            }
            // Lazy reinit if evicted.
            if guard.is_none() {
                metrics::counter!(crate::metrics::names::POOL_COLD_STARTS).increment(1);
                tracing::info!("pool cold start: reinitializing evicted slot");
                *guard = Some((self.factory)());
            }
            slot.busy.store(true, Ordering::Release);
            slot.last_used.store(now, Ordering::Relaxed);
            let item = guard.take().unwrap();
            return Some(EvictableGuard { slot: Arc::clone(slot), item: Some(item) });
        }
        None
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
                pool.evict_idle(threshold);
            }
        })
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

/// RAII guard that returns the item to its slot on drop and refreshes `last_used`.
pub struct EvictableGuard<T> {
    slot: Arc<EvictableSlot<T>>,
    item: Option<T>,
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
            }
            self.slot.busy.store(false, Ordering::Release);
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
                42u32
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
        assert_eq!(evicted, 2, "both idle slots should be evicted after threshold");
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
        assert_eq!(evicted, 1, "only stale slot evicted; recently used slot survives");
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
        use std::sync::atomic::{AtomicUsize, Ordering as AOrdering2};
        use std::sync::Arc as StdArc;

        // Pool with 1 slot, idle_secs=1 so eviction is enabled.
        let evict_counter = StdArc::new(AtomicUsize::new(0));
        let ec = evict_counter.clone();
        let pool = StdArc::new(EvictablePool::new(
            1,
            1,
            StdArc::new(move || {
                ec.fetch_add(1, AOrdering2::SeqCst);
                99u32
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
        assert!(guard.is_some(), "idle_secs=0 → no eviction even with stale last_used");
    }
}
