# ox-whisper Improvements Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix VAD boundary artifacts, add SSE streaming endpoint for real-time feedback, add recognizer pool for concurrent requests.

**Architecture:** Three independent improvements to the existing ox-whisper Rust service. VAD padding fixes edge artifacts. SSE streaming sends partial results during chunked transcription. Recognizer pool eliminates the Mutex bottleneck for concurrent requests.

**Tech Stack:** Rust, axum (SSE via `axum::response::Sse`), sherpa-onnx offline recognizers, tokio

---

### Task 1: VAD Boundary Padding

**Problem:** VAD segments cut speech at exact boundaries, causing artifacts like "дру" at the start of segments where the model misinterprets the cut audio.

**Files:**
- Modify: `src/vad.rs:48-76`

**Step 1: Write the fix — add padding around VAD segments**

In `apply_vad()`, after collecting segments and before grouping into chunks, pad each segment with silence or adjacent audio to avoid hard cuts. Add 50ms (800 samples at 16kHz) of silence padding before each segment.

```rust
// In apply_vad(), after draining segments and before grouping:
const PAD_SAMPLES: usize = 800; // 50ms at 16kHz

// Pad segment starts with preceding audio (or silence)
for segment in &mut segments {
    let mut padded = vec![0.0f32; PAD_SAMPLES];
    padded.extend_from_slice(&segment.samples);
    segment.samples = padded;
}
```

Wait — `segment.samples` is from `SpeechSegment` which is a sherpa type. We can't mutate it. Instead, we should build our own padded sample vectors during the grouping phase.

The actual approach: when building chunks from segments, prepend silence padding:

In `src/vad.rs`, change the segment grouping loop:

```rust
const PAD_SAMPLES: usize = 800; // 50ms pre-padding

for segment in segments {
    let mut seg_samples: Vec<f32> = Vec::with_capacity(PAD_SAMPLES + segment.samples.len());
    seg_samples.extend(std::iter::repeat(0.0f32).take(PAD_SAMPLES));
    seg_samples.extend_from_slice(&segment.samples);
    let seg_slice = &seg_samples[..];
    // ... rest of force-split logic uses seg_slice
}
```

But the current code uses `&segment.samples[..]` as a slice reference. We need to restructure slightly to own the padded data.

**Step 2: Run tests**

```bash
cd /home/krolik/src/ox-whisper && cargo test
```

**Step 3: Test with RU audio**

```bash
curl -s -X POST http://127.0.0.1:8092/transcribe/upload \
  -F "file=@/home/krolik/tools/moonshine-service/models/ru/test_wavs/0.wav" \
  -F "language=ru" | python3 -m json.tool
```

Expected: no "дру" artifact.

**Step 4: Commit**

```bash
git add src/vad.rs
git commit -m "fix: add 50ms silence padding to VAD segments to reduce boundary artifacts"
```

---

### Task 2: SSE Streaming Endpoint

**Problem:** Long audio transcription (24s+) takes 2-3 seconds. Clients get no feedback until complete. Add an SSE endpoint that streams partial results as each chunk is transcribed.

**Files:**
- Modify: `src/handlers.rs` — add `/transcribe/stream` SSE endpoint
- Modify: `src/transcribe.rs` — add `transcribe_streaming()` that yields results per chunk
- Modify: `src/main.rs` — register new route
- Modify: `Cargo.toml` — add `tokio-stream` dependency

**Step 1: Add tokio-stream dependency**

```bash
cd /home/krolik/src/ox-whisper && cargo add tokio-stream
```

**Step 2: Add streaming transcription function**

In `src/transcribe.rs`, add a function that returns results per chunk instead of joining:

```rust
pub struct ChunkResult {
    pub text: String,
    pub chunk_index: usize,
    pub total_chunks: usize,
}

pub fn transcribe_chunks_iter(
    models: &Models,
    config: &Config,
    audio_path: &Path,
    language: &str,
    vad_override: Option<bool>,
) -> Result<(Vec<ChunkResult>, f64), TranscribeError> {
    // Same pipeline as transcribe() but returns per-chunk results
    // WAV conversion, loading, VAD, chunking...
    // Then iterate chunks, returning each result
}
```

**Step 3: Add SSE handler**

In `src/handlers.rs`:

```rust
use axum::response::sse::{Event, Sse};
use tokio_stream::StreamExt;

pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    // Parse upload, then spawn_blocking for transcription
    // Send SSE events for each chunk: { "chunk": N, "total": M, "text": "..." }
    // Final event: { "done": true, "text": "full text", "duration_ms": ... }
}
```

**Step 4: Register route**

In `src/main.rs`:

```rust
.route("/transcribe/stream", post(handlers::transcribe_stream))
```

**Step 5: Test SSE endpoint**

```bash
curl -N -X POST http://127.0.0.1:8092/transcribe/stream \
  -F "file=@/tmp/test_long_en.wav" -F "language=en"
```

Expected: multiple SSE events, then final result.

**Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/handlers.rs src/transcribe.rs src/main.rs
git commit -m "feat: add SSE streaming endpoint /transcribe/stream"
```

---

### Task 3: Recognizer Pool for Concurrent Requests

**Problem:** `Mutex<MoonshineRecognizer>` allows only one concurrent transcription. Multiple simultaneous requests queue behind the mutex. Replace with a pool of recognizers.

**Files:**
- Create: `src/pool.rs` — generic recognizer pool
- Modify: `src/models.rs` — use pool instead of Mutex
- Modify: `src/transcribe.rs` — acquire from pool instead of locking mutex
- Modify: `src/main.rs` — add `mod pool`
- Modify: `src/config.rs` — add `pool_size` config

**Step 1: Create recognizer pool**

`src/pool.rs` — a simple channel-based pool:

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;
use std::sync::Mutex;

pub struct Pool<T> {
    items: Mutex<Vec<T>>,
    semaphore: Semaphore,
}

impl<T> Pool<T> {
    pub fn new(items: Vec<T>) -> Self {
        let len = items.len();
        Self {
            items: Mutex::new(items),
            semaphore: Semaphore::new(len),
        }
    }

    pub fn acquire(&self) -> Option<PoolGuard<'_, T>> {
        let permit = self.semaphore.try_acquire().ok()?;
        let item = self.items.lock().ok()?.pop()?;
        Some(PoolGuard { pool: self, item: Some(item), _permit: permit })
    }
}

pub struct PoolGuard<'a, T> {
    pool: &'a Pool<T>,
    item: Option<T>,
    _permit: tokio::sync::SemaphorePermit<'a>,
}

impl<T> std::ops::DerefMut for PoolGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.item.as_mut().unwrap()
    }
}

impl<T> std::ops::Deref for PoolGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { self.item.as_ref().unwrap() }
}

impl<T> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            if let Ok(mut items) = self.pool.items.lock() {
                items.push(item);
            }
        }
    }
}
```

**Step 2: Add pool_size config**

In `src/config.rs`, add:

```rust
pub pool_size: usize,
// ...
pool_size: env::var("POOL_SIZE")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(2),
```

**Step 3: Update Models to use Pool**

In `src/models.rs`, change:

```rust
pub en: Option<Pool<MoonshineRecognizer>>,  // was Mutex
pub ru: Option<Pool<TransducerRecognizer>>, // was Mutex
// vad and punct stay as Mutex (lightweight, fast)
```

Load `pool_size` recognizer instances:

```rust
fn load_moonshine_pool(config: &Config) -> Option<Pool<MoonshineRecognizer>> {
    let mut recognizers = Vec::new();
    for i in 0..config.pool_size {
        match MoonshineRecognizer::new(moonshine_cfg.clone()) {
            Ok(r) => recognizers.push(r),
            Err(e) => { tracing::error!("Pool {}: {}", i, e); break; }
        }
    }
    if recognizers.is_empty() { None } else { Some(Pool::new(recognizers)) }
}
```

**Step 4: Update transcribe.rs**

Change mutex locking to pool acquire:

```rust
fn transcribe_en(models: &Models, chunks: &[Vec<f32>], language: &str) -> Result<Vec<String>, TranscribeError> {
    let pool = models.en.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable(language.to_string()))?;
    let mut rec = pool.acquire()
        .ok_or(TranscribeError::NoRecognizer)?;
    // ... rest same
}
```

**Step 5: Test concurrent requests**

```bash
# Send 3 requests simultaneously
for i in 1 2 3; do
  curl -s -X POST http://127.0.0.1:8092/transcribe/upload \
    -F "file=@/tmp/test_en.wav" -F "language=en" &
done
wait
```

**Step 6: Commit**

```bash
git add src/pool.rs src/models.rs src/transcribe.rs src/config.rs src/main.rs
git commit -m "feat: add recognizer pool for concurrent transcription requests"
```
