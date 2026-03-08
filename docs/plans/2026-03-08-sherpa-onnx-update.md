# Sherpa-onnx v1.12.9 → v1.12.28 + GigaAM v3 Architecture Update

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update sherpa-onnx to v1.12.28, switch RU from GigaAM v2 CTC to GigaAM v3 transducer
with built-in punctuation, expose new capabilities (word timestamps, confidence scores) for RU.

**Architecture:** Replace sherpa-onnx source, rebuild C API shared libraries, update sherpa-rs
Rust wrappers to expose new result fields (timestamps, ys_log_probs), make RU transcription
produce punctuated text natively (no separate punct model), add word timestamps for RU.

**Tech Stack:** Rust 1.88, sherpa-onnx (C++/CMake), bindgen, aarch64-linux

---

## New Capabilities to Leverage

GigaAM v3 transducer (`sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16`):
- **Built-in punctuation**: `punct` variant produces punctuated & normalized text directly
- **WER 8.4%** average (vs ~12% v2 CTC, vs 25.1% Whisper)
- **30% better** on callcenter, background music, natural speech, disordered speech

sherpa-onnx v1.12.28 `SherpaOnnxOfflineRecognizerResult` new fields:
- `ys_log_probs` (float*) — per-token log probability → confidence filtering (works for ALL offline transducers incl. GigaAM)
- `durations` (float*) — per-token duration in seconds (TDT models only, NOT GigaAM)
- `timestamps` (float*) — per-token absolute time (already existed, use for word timestamps)
- `segment_timestamps/durations/texts` — segment-level output (Whisper-style)
- `json` — structured JSON with all result data
- `emotion`, `event` — for future capabilities

GigaAM v3 technical details:
- feature_dim = **64** mel filters (not 80) — sherpa-onnx reads from ONNX metadata automatically
- Punct variant vocab_size = **513** tokens (non-punct variant = 1025)
- encoder dim = 768, decoder = LSTM (float32), joiner = float32
- sample_rate = 16000 Hz
- fbank: dither=0, preemph=0, hann window, low_freq=0, high_freq=8000

---

### Task 1: Update sherpa-onnx source

**Files:**
- Replace: `vendor/sherpa-rs-sys/sherpa-onnx/` (entire directory, gitignored)

**Step 1: Remove old sherpa-onnx source**

```bash
rm -rf vendor/sherpa-rs-sys/sherpa-onnx
```

**Step 2: Clone v1.12.28**

```bash
git clone --depth 1 --branch v1.12.28 \
  https://github.com/k2-fsa/sherpa-onnx.git \
  vendor/sherpa-rs-sys/sherpa-onnx
rm -rf vendor/sherpa-rs-sys/sherpa-onnx/.git
```

**Step 3: Verify version**

```bash
grep "SHERPA_ONNX_VERSION" vendor/sherpa-rs-sys/sherpa-onnx/CMakeLists.txt
```
Expected: `set(SHERPA_ONNX_VERSION "1.12.28")`

**Step 4: Apply session.cc optimizations**

First verify the sed patterns still match in v1.12.28:

```bash
grep -n "SetInterOpNumThreads(num_threads)" vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc
grep -n "// sess_opts.SetGraphOptimizationLevel" vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc
grep -n "ORT_ENABLE_EXTENDED" vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc
```

If all three match, apply:

```bash
sed -i \
  -e 's/SetInterOpNumThreads(num_threads)/SetInterOpNumThreads(1)/' \
  -e 's|// sess_opts.SetGraphOptimizationLevel|sess_opts.SetGraphOptimizationLevel|' \
  -e 's/ORT_ENABLE_EXTENDED/ORT_ENABLE_ALL/' \
  vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc
```

If patterns changed, adapt accordingly and update the Dockerfile sed commands too.

---

### Task 2: Build new sherpa-onnx shared libraries

**Files:**
- Output: `vendor/sherpa-onnx/lib/libsherpa-onnx-c-api.so`
- Output: `vendor/sherpa-onnx/lib/libsherpa-onnx-cxx-api.so`

The pre-built `libonnxruntime.so` (v1.23.2) stays as-is.

**Step 1: Configure cmake build**

```bash
cd vendor/sherpa-rs-sys/sherpa-onnx
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNXRUNTIME_LIB_DIR=/home/krolik/src/ox-whisper/vendor/sherpa-onnx/lib \
  -DSHERPA_ONNXRUNTIME_INCLUDE_DIR=/home/krolik/src/ox-whisper/vendor/sherpa-rs-sys/sherpa-onnx/cmake/
```

Note: We point `SHERPA_ONNXRUNTIME_LIB_DIR` to our existing pre-built ORT v1.23.2.
If cmake auto-downloads ORT, let it — then just copy the built sherpa libs.

**Step 2: Build**

```bash
cmake --build . --config Release -j4
```

**Step 3: Copy built libraries**

```bash
cp lib/libsherpa-onnx-c-api.so /home/krolik/src/ox-whisper/vendor/sherpa-onnx/lib/
cp lib/libsherpa-onnx-cxx-api.so /home/krolik/src/ox-whisper/vendor/sherpa-onnx/lib/
```

**Step 4: Verify ABI compatibility**

```bash
readelf -V vendor/sherpa-onnx/lib/libsherpa-onnx-c-api.so | grep VERS
```

---

### Task 3: Update sherpa-rs to expose new result fields

**Files:**
- Modify: `vendor/sherpa-rs/src/lib.rs` — `OfflineRecognizerResult` struct
- Modify: `vendor/sherpa-rs/src/transducer.rs` — return `OfflineRecognizerResult` instead of `String`

**Step 1: Update `OfflineRecognizerResult` in `vendor/sherpa-rs/src/lib.rs`**

Add new fields to the Rust struct to match the v1.12.28 C API:

```rust
#[derive(Debug, Clone)]
pub struct OfflineRecognizerResult {
    pub lang: String,
    pub text: String,
    pub timestamps: Vec<f32>,
    pub tokens: Vec<String>,
    // NEW: per-token durations (seconds) — for word-level timestamps
    pub durations: Vec<f32>,
    // NEW: per-token log probabilities — for confidence scoring
    pub log_probs: Vec<f32>,
}
```

Update `OfflineRecognizerResult::new()` to read the new fields from the C struct:

```rust
impl OfflineRecognizerResult {
    fn new(result: &sherpa_rs_sys::SherpaOnnxOfflineRecognizerResult) -> Self {
        let lang = unsafe { cstr_to_string(result.lang) };
        let text = unsafe { cstr_to_string(result.text) };
        let count = result.count.try_into().unwrap();

        let timestamps = if result.timestamps.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(result.timestamps, count).to_vec() }
        };

        let durations = if result.durations.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(result.durations, count).to_vec() }
        };

        let log_probs = if result.ys_log_probs.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(result.ys_log_probs, count).to_vec() }
        };

        let mut tokens = Vec::with_capacity(count);
        let mut next_token = result.tokens;
        for _ in 0..count {
            let token = unsafe { CStr::from_ptr(next_token) };
            tokens.push(token.to_string_lossy().into_owned());
            next_token = next_token
                .wrapping_byte_offset(token.to_bytes_with_nul().len().try_into().unwrap());
        }

        Self { lang, text, timestamps, tokens, durations, log_probs }
    }
}
```

**Step 2: Update `TransducerRecognizer::transcribe()` in `vendor/sherpa-rs/src/transducer.rs`**

Change return type from `String` to `OfflineRecognizerResult`:

```rust
pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> super::OfflineRecognizerResult {
    unsafe {
        let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
        sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
            stream, sample_rate as i32, samples.as_ptr(),
            samples.len().try_into().unwrap(),
        );
        sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
        let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
        let raw_result = result_ptr.read();
        let result = super::OfflineRecognizerResult::new(&raw_result);
        sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
        sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
        result
    }
}
```

Add `transcribe_batch()` to `TransducerRecognizer` (same pattern as `NemoCtcRecognizer`):

```rust
pub fn transcribe_batch(
    &mut self, sample_rate: u32, chunks: &[&[f32]],
) -> Vec<super::OfflineRecognizerResult> {
    if chunks.is_empty() { return Vec::new(); }
    unsafe {
        let mut streams = Vec::with_capacity(chunks.len());
        for samples in chunks {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream, sample_rate as i32, samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            streams.push(stream);
        }
        let mut stream_ptrs: Vec<_> = streams.iter().map(|s| *s as *const _).collect();
        sherpa_rs_sys::SherpaOnnxDecodeMultipleOfflineStreams(
            self.recognizer, stream_ptrs.as_mut_ptr(), streams.len() as i32,
        );
        let mut results = Vec::with_capacity(streams.len());
        for stream in &streams {
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(*stream);
            let raw = result_ptr.read();
            results.push(super::OfflineRecognizerResult::new(&raw));
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
        }
        for stream in streams {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
        }
        results
    }
}
```

---

### Task 4: Update RuRecognizer to use unified result type

**Files:**
- Modify: `src/recognizer.rs`

Now that both NemoCtc and Transducer return `OfflineRecognizerResult`, unify the API:

```rust
use sherpa_rs::OfflineRecognizerResult;
use sherpa_rs::nemo_ctc::NemoCtcRecognizer;
use sherpa_rs::transducer::TransducerRecognizer;

pub enum RuRecognizer {
    Transducer(TransducerRecognizer),
    NemoCtc(NemoCtcRecognizer),
}

impl RuRecognizer {
    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> OfflineRecognizerResult {
        match self {
            Self::Transducer(r) => r.transcribe(sample_rate, samples),
            Self::NemoCtc(r) => r.transcribe(sample_rate, samples),
        }
    }

    pub fn transcribe_batch(&mut self, sample_rate: u32, chunks: &[&[f32]]) -> Vec<OfflineRecognizerResult> {
        match self {
            Self::NemoCtc(r) => r.transcribe_batch(sample_rate, chunks),
            Self::Transducer(r) => r.transcribe_batch(sample_rate, chunks),
        }
    }

    /// Returns true if the model produces punctuated text natively.
    pub fn has_builtin_punct(&self) -> bool {
        matches!(self, Self::Transducer(_))
    }
}
```

Key change: `transcribe()` now returns `OfflineRecognizerResult` (not `String`), giving
access to `durations`, `log_probs`, `timestamps`, `tokens` for RU.

---

### Task 5: Add word timestamps and confidence for RU transcription

**Files:**
- Modify: `src/transcribe.rs` — `transcribe_ru()` to extract word timestamps

**Important:** `durations` field is TDT-only (Parakeet), NOT available for GigaAM v3.
Use `timestamps` field (absolute per-token times) which works for all transducers.
The existing `extract_words()` in `words.rs` already handles timestamp-based extraction.

**Step 1: Update `transcribe_ru()` in `src/transcribe.rs`**

Return word timestamps from RU transcription using the existing `extract_words()`:

```rust
fn transcribe_ru(
    models: &Models, chunks: &[Vec<f32>], chunk_offsets: &[f64],
) -> Result<(Vec<String>, Vec<WordTimestamp>), TranscribeError> {
    let pool = models.ru.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable("ru".to_string()))?;
    let mut rec = pool.acquire().ok_or(TranscribeError::NoRecognizer)?;
    let chunk_refs: Vec<&[f32]> = chunks.iter().map(|c| c.as_slice()).collect();
    let results = rec.transcribe_batch(16000, &chunk_refs);

    let mut texts = Vec::new();
    let mut words = Vec::new();
    for (i, r) in results.into_iter().enumerate() {
        let t = r.text.trim().to_string();
        if t.is_empty() || compression_ratio(&t) > 2.4 { continue; }
        let offset = chunk_offsets.get(i).copied().unwrap_or(0.0) as f32;

        // Use timestamps for word extraction (works for both CTC and transducer)
        if !r.timestamps.is_empty() {
            extract_words(&r.tokens, &r.timestamps, offset, &mut words);
        }

        texts.push(t);
    }
    Ok((texts, words))
}
```

**Step 3: Update `do_transcribe()` to pass chunk_offsets to RU and skip punct for RU**

```rust
// In do_transcribe():
let chunk_offsets = compute_chunk_offsets(&audio_chunks, 16000);
let (texts, words) = match language {
    "ru" => transcribe_ru(models, &audio_chunks, &chunk_offsets)?,
    _ => transcribe_en(models, &audio_chunks, language, &chunk_offsets)?,
};

// ...
// Skip external punctuation if model has built-in punct (GigaAM v3 transducer)
let skip_punct = language == "ru" && models.ru.as_ref()
    .and_then(|p| p.acquire())
    .map(|r| r.has_builtin_punct())
    .unwrap_or(false);
let text = if skip_punct {
    text
} else {
    maybe_punctuate(models, &text, language, punctuate_override)
};
```

---

### Task 6: Add confidence scores to API response

**Files:**
- Modify: `src/words.rs` — add `confidence` to `WordTimestamp`
- Modify: `src/handlers.rs` — add `confidence` to `TranscribeResponse`

**Step 1: Extend `WordTimestamp` with confidence**

```rust
#[derive(Debug, Clone, serde::Serialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f32,
    pub end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}
```

**Step 2: Populate confidence from `log_probs`**

In both `extract_words()` and `extract_words_from_durations()`, when building a word,
average the log_probs of its constituent tokens and convert to probability:

Add `log_probs: &[f32]` parameter. For each word, compute:
`confidence = Some((sum_log_probs / num_tokens).exp())`

Only populate if `log_probs` is non-empty, otherwise `confidence = None`.

**Step 3: Update `TranscribeResponse`**

Add average confidence to the response:

```rust
#[derive(Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<String>,
    pub duration_ms: f64,
    #[serde(skip_serializing_if = "is_zero")]
    pub speech_ms: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<crate::words::WordTimestamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,  // NEW: average confidence across all words
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
```

---

### Task 7: Update model loading and health endpoint

**Files:**
- Modify: `src/models.rs` — detect GigaAM v3 punct capability
- Modify: `src/handlers.rs` — update health response with model info

**Step 1: Update `load_ru()` detection logic in `src/models.rs`**

Current logic: if `model.int8.onnx` exists and no `encoder.int8.onnx` → NeMo CTC.
If `encoder.int8.onnx` exists → Transducer.

This stays the same. The `detect_transducer_type()` already handles `nemo_transducer`
detection. No changes needed here — model loading already works.

**Step 2: Update health endpoint in `src/handlers.rs`**

Replace hardcoded `"gigaam-v2-ctc-int8"` with dynamic model info:

```rust
let ru_model_name = if let Some(ref pool) = state.models.ru {
    if let Some(r) = pool.acquire() {
        if r.has_builtin_punct() { "gigaam-v3-rnnt-punct" } else { "gigaam-v2-ctc" }
    } else { "unknown" }
} else { "not-loaded" };

languages.insert("ru", LanguageInfo {
    model: ru_model_name,
    ready: state.models.ru.is_some(),
});
```

Note: `LanguageInfo.model` needs to change from `&'static str` to `String` (or use
leaked strings for the dynamic case).

---

### Task 8: Update streaming transcription for RU

**Files:**
- Modify: `src/streaming.rs` — `transcribe_ru_streaming()` to use new result type

Update `transcribe_ru_streaming()` to work with `OfflineRecognizerResult`:

```rust
fn transcribe_ru_streaming(
    models: &Models, chunks: &[Vec<f32>], total: usize,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<Vec<String>, TranscribeError> {
    let pool = models.ru.as_ref()
        .ok_or_else(|| TranscribeError::LanguageNotAvailable("ru".to_string()))?;
    let mut rec = pool.acquire().ok_or(TranscribeError::NoRecognizer)?;
    let mut texts = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let result = rec.transcribe(16000, chunk);
        let text = result.text.trim().to_string();
        if !text.is_empty() && compression_ratio(&text) <= 2.4 {
            let _ = tx.blocking_send(StreamEvent {
                chunk_index: i, total_chunks: total, text: text.clone(),
            });
            texts.push(text);
        }
    }
    Ok(texts)
}
```

---

### Task 9: Build and regression test

**Step 1: Build**

```bash
SHERPA_LIB_PATH=/home/krolik/src/ox-whisper/vendor/sherpa-onnx \
  cargo build --release 2>&1
```

Fix any compilation errors (likely: `OfflineRecognizerResult` field changes propagating).

**Step 2: Test with existing models (GigaAM v2 CTC)**

```bash
LD_LIBRARY_PATH=vendor/sherpa-onnx/lib \
MOONSHINE_PORT=8093 \
MOONSHINE_MODELS_DIR=/home/krolik/tools/moonshine-service/models/en \
ZIPFORMER_RU_DIR=/home/krolik/tools/moonshine-service/models/ru \
SILERO_VAD_MODEL=/home/krolik/tools/moonshine-service/models/vad/silero_vad.onnx \
PUNCT_MODEL=/home/krolik/tools/moonshine-service/models/punct-en/model.int8.onnx \
PUNCT_VOCAB=/home/krolik/tools/moonshine-service/models/punct-en/bpe.vocab \
./target/release/ox-whisper
```

```bash
# Test EN
time curl -s -X POST http://localhost:8093/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path":"/tmp/test_5s.wav","language":"en"}'

# Test RU (GigaAM v2 CTC — should still work as fallback)
time curl -s -X POST http://localhost:8093/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path":"/tmp/test_5s.wav","language":"ru"}'
```

---

### Task 10: Test GigaAM v3 model

**Step 1: Download GigaAM v3**

```bash
mkdir -p /tmp/gigaam-v3 && cd /tmp/gigaam-v3
for f in encoder.int8.onnx decoder.onnx joiner.onnx tokens.txt; do
  curl -sL -O "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16/resolve/main/$f"
done
```

**Step 2: Patch decoder metadata** (if still needed with v1.12.28)

The punct variant has vocab_size=513 (non-punct has 1025). Try WITHOUT patching first —
v1.12.28 may read metadata from model automatically. If it fails:

```python
import onnx
m = onnx.load('/tmp/gigaam-v3/decoder.onnx')
m.metadata_props.append(onnx.StringStringEntryProto(key='context_size', value='1'))
m.metadata_props.append(onnx.StringStringEntryProto(key='vocab_size', value='513'))
onnx.save(m, '/tmp/gigaam-v3/decoder.onnx')
```

**Step 3: Start server with GigaAM v3**

```bash
LD_LIBRARY_PATH=vendor/sherpa-onnx/lib \
MOONSHINE_PORT=8093 \
MOONSHINE_MODELS_DIR=/home/krolik/tools/moonshine-service/models/en \
ZIPFORMER_RU_DIR=/tmp/gigaam-v3 \
SILERO_VAD_MODEL=/home/krolik/tools/moonshine-service/models/vad/silero_vad.onnx \
PUNCT_MODEL=/home/krolik/tools/moonshine-service/models/punct-en/model.int8.onnx \
PUNCT_VOCAB=/home/krolik/tools/moonshine-service/models/punct-en/bpe.vocab \
./target/release/ox-whisper
```

**Step 4: Test and benchmark**

```bash
# Verify punctuation is in the output (no separate punct model for RU)
time curl -s -X POST http://localhost:8093/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path":"/tmp/test_5s.wav","language":"ru"}' | python3 -m json.tool

# Check word timestamps are populated
# Check health endpoint shows gigaam-v3-rnnt-punct
curl -s http://localhost:8093/health | python3 -m json.tool
```

Compare speed and quality with GigaAM v2 CTC.

---

### Task 11: Update Dockerfile

**Files:**
- Modify: `Dockerfile`

**Step 1: Verify Dockerfile sed patterns**

Check that the session.cc `sed` commands in Dockerfile still apply to v1.12.28.
If patterns changed in Task 1 Step 4, update the Dockerfile to match.

**Step 2: Docker build test**

```bash
docker build -t ox-whisper-test .
```

---

### Task 12: Commit, push, update docs

**Step 1: Update experiments.md**

Add GigaAM v3 test results with updated sherpa-onnx.

**Step 2: Commit**

```bash
git add vendor/sherpa-rs/ src/ docs/ Dockerfile
git commit -m "feat: update sherpa-onnx v1.12.9 → v1.12.28

- GigaAM v3 transducer with built-in punctuation for RU
- Word timestamps for RU via token durations
- Confidence scores (log_probs) in API response
- Batch decode for transducer model
- Skip separate punct model when using punct-capable transducer
- Update health endpoint with dynamic model info"
git push origin master
```

Do NOT deploy (standing user instruction).

---

### Architecture Summary

```
Before (v1.12.9 + GigaAM v2 CTC):
  Audio → VAD → CTC decode (batch) → join text → external punct model → response
  - No word timestamps for RU
  - No confidence scores
  - Separate punct model required

After (v1.12.28 + GigaAM v3 transducer):
  Audio → VAD → transducer decode (batch) → punctuated text + timestamps + log_probs → response
  - Word timestamps from token timestamps (same as EN Moonshine)
  - Per-word confidence from ys_log_probs (transducer-only)
  - Built-in punctuation (no separate model for RU)
  - GigaAM v2 CTC kept as automatic fallback
  - feature_dim=64 (read from ONNX metadata, not hardcoded)
```

### Risks

1. **sherpa-onnx cmake may try to download its own ORT** — need to point it to our existing
   v1.23.2 or let it download and just use the built sherpa libs.
2. **Struct layout changes** — bindgen handles this automatically, but sherpa-rs Rust wrappers
   reference fields that may have moved. The explicit field reads in Task 3 mitigate this.
3. **session.cc sed patterns may differ** — v1.12.28 may have changed the default session
   options code, breaking our sed patches.
4. **GigaAM v3 may still need decoder metadata patching** — the newer sherpa-onnx might
   read metadata differently.
5. **Transducer may be slower than CTC** — transducer does autoregressive decoding vs
   single-pass CTC. Benchmark in Task 10 will reveal this.
6. **`has_builtin_punct()` detection** — currently hardcoded to `Transducer` variant. If a
   non-punct transducer is used, this would incorrectly skip punctuation. Could improve
   by checking model name or metadata, but YAGNI for now.
