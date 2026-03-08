# ox-whisper Performance Experiments

## 2026-03-08: sherpa-onnx v1.12.28 Update + GigaAM v3 Architecture

### Goal

Update sherpa-onnx from v1.12.9 to v1.12.28 to unlock GigaAM v3 support, add per-word
confidence scores (`ys_log_probs`), and prepare architecture for GigaAM v3 punct model.

### Changes

1. **sherpa-onnx v1.12.9 → v1.12.28**: Replaced vendored source, rebuilt shared libraries.
   New C API fields: `durations`, `ys_log_probs`, `segment_timestamps/durations/texts`.
   New model backends: `wenet_ctc`, `omnilingual`, `medasr`, `funasr_nano`, `fire_red_asr_ctc`.

2. **Per-word confidence scores**: `ys_log_probs` → `log_probs` in `OfflineRecognizerResult` →
   `extract_words_with_confidence()` → `exp(avg_log_prob)` per word → `confidence` field in API.
   Works for all offline transducers including current v2 Zipformer.

3. **Transducer `transcribe()` → `OfflineRecognizerResult`**: Previously returned `String`,
   now returns full result with tokens, timestamps, log_probs. Enables word timestamps for
   transducer models (not just NeMo CTC).

4. **Built-in punctuation detection**: `has_builtin_punct()` on `RuRecognizer::Transducer`
   skips external punctuation model for GigaAM v3 (which has native punctuation).

5. **Health endpoint**: Dynamically reports `"gigaam-v3-rnnt-punct"` vs `"gigaam-v2-ctc"`
   based on loaded model type.

### Results

- EN Moonshine: No regression, transcription works
- RU v2 Transducer: Word timestamps + confidence scores (0.68–0.96 per word)
- Session.cc optimizations: Same sed patterns work on v1.12.28 (unchanged from v1.12.9)
- GigaAM v3 model not yet tested (requires model download)

### GigaAM v3 Model Info

- Punct variant: `csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16`
  - vocab_size = 513 tokens, feature_dim = 64 mel filters
  - WER 8.4% average (vs ~12% v2 CTC, vs 25.1% Whisper)
  - Built-in punctuation (no external punct model needed)
- Non-punct variant: vocab_size = 1025, same architecture

---

## 2026-03-08: GigaAM v3 Transducer — FAILED (pre-update attempt)

### Goal

Test GigaAM v3 RNNT (transducer with built-in punctuation) as a potential upgrade from GigaAM v2 CTC.

Model: `csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16`
- encoder.int8.onnx: 215MB (vs 226MB v2 CTC)
- decoder.onnx: 4.4MB, joiner.onnx: 2.6MB
- 1025 tokens (vs 3422 nodes in v2 CTC)
- model_type: `EncDecRNNTBPEModel` (NeMo transducer, NOT Zipformer)

### Changes Made

1. **Auto-detect NeMo transducer** (`models.rs`): `detect_transducer_type()` scans encoder
   ONNX metadata for `EncDecRNNTBPEModel` or `is_giga_am` markers, sets `model_type`
   to `"nemo_transducer"` instead of `"transducer"`.

2. **Flexible model file resolution** (`models.rs`): `find_model_file()` prefers `.int8.onnx`
   but falls back to `.onnx` — GigaAM v3 only ships non-quantized decoder/joiner.

3. **Decoder metadata patching**: GigaAM v3 decoder.onnx lacks `vocab_size` and `context_size`
   metadata that sherpa-onnx requires. Patched via onnx Python lib:
   `context_size=1, vocab_size=1025` (derived from decoder input shape and joiner output dim).

### Results

- **Model loading**: OK — detected as NeMo transducer, loaded successfully
- **Warmup (1s silence)**: OK — no crash
- **Actual transcription**: CRASH — `ConvInteger node '/pre_encode/conv/conv.0/Conv_quant':
  Invalid input shape: {0}`
- The error occurs in the encoder's quantized convolution, suggesting sherpa-onnx's feature
  extraction produces incompatible input dimensions for GigaAM v3's encoder architecture.

### Root Cause

Our sherpa-onnx vendor is **v1.12.9** (September 2024). GigaAM v3 models were released
**December 2025** for sherpa-onnx v1.12.28. The older version likely lacks proper feature
extraction and padding logic for GigaAM v3's `EncDecRNNTBPEModel` encoder.

### Path Forward

To use GigaAM v3, need to update the sherpa-onnx vendor submodule from v1.12.9 to v1.12.28+.
This requires:
- Updating `vendor/sherpa-rs-sys/sherpa-onnx/` submodule
- Rebuilding pre-built libraries (`libsherpa-onnx-c-api.so`, `libsherpa-onnx-cxx-api.so`)
- Potentially updating `sherpa-rs-sys` and `sherpa-rs` Rust bindings
- Docker rebuild

The `detect_transducer_type()` and `find_model_file()` helpers are kept — they're useful
regardless and make the model loading more robust.

---

## 2026-03-07: ONNX Runtime Optimizations

### Context

ox-whisper uses sherpa-onnx (via sherpa-rs Rust bindings) for speech-to-text inference.
Two models: English (Moonshine v2 Base, 68MB) and Russian (GigaAM v2 CTC int8, 226MB, 3422 nodes).

Pre-built sherpa-onnx ships with `libonnxruntime.so` v1.23.2 from csukuangfj/onnxruntime-libs.

### Optimization 1: ORT_ENABLE_ALL Graph Optimization

**What:** Changed `SetGraphOptimizationLevel` from `ORT_ENABLE_EXTENDED` (commented out by default)
to `ORT_ENABLE_ALL` in `vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc`.

**How:** Applied via `sed` in Dockerfile (vendor is gitignored):
```dockerfile
RUN sed -i \
    -e 's|// sess_opts.SetGraphOptimizationLevel|sess_opts.SetGraphOptimizationLevel|' \
    -e 's/ORT_ENABLE_EXTENDED/ORT_ENABLE_ALL/' \
    vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc
```

**Result:**
- EN Moonshine: **~4x speedup** (2.5-3.7s -> 0.6-0.8s for 5s audio)
- RU GigaAM: ~10-15% improvement

ORT_ENABLE_ALL enables constant folding, kernel fusion, and layout optimizations.
The dramatic EN improvement is likely due to Moonshine's architecture being highly amenable to fusion.

### Optimization 2: Inter-op Thread Reduction

**What:** Changed `SetInterOpNumThreads(num_threads)` to `SetInterOpNumThreads(1)`.

**Rationale:** Inter-op parallelism adds overhead for single-stream models (CTC, transducer).
The parallelism between operators is minimal for sequential inference — one thread eliminates
context-switching overhead.

**Result:** Small but consistent improvement across both models.

### Optimization 3: Batch Decode for VAD Chunks (RU)

**What:** Added `SherpaOnnxDecodeMultipleOfflineStreams` support to NeMo CTC recognizer.
When VAD splits audio into chunks, all chunks are decoded in a single batch call instead of
sequential loop.

**Implementation:** `vendor/sherpa-rs/src/nemo_ctc.rs` — `transcribe_batch()` method.
`src/recognizer.rs` — `RuRecognizer::transcribe_batch()` with Transducer fallback.
`src/transcribe.rs` — `transcribe_ru()` uses batch decoding.

**Result:** ~20-30% speedup for multi-chunk RU audio (most noticeable on longer files).

### Optimization 4: XNNPACK Execution Provider — FAILED

**Goal:** Enable XNNPACK EP for ARM NEON-optimized inference on aarch64.

**Approach:** Build onnxruntime from source with `-Donnxruntime_USE_XNNPACK=ON`, replace
the pre-built `libonnxruntime.so`.

**Attempts:**

1. **ORT v1.17.1 build** — Two build issues fixed:
   - Eigen hash mismatch (GitLab changed the archive hash for the same commit)
   - `_FORTIFY_SOURCE` redefinition with `-Werror` (build.py adds `-D_FORTIFY_SOURCE=2`
     but GCC 13 already defines it; fixed by replacing with `-U_FORTIFY_SOURCE`)
   - Build succeeded but **ABI mismatch**: sherpa-onnx expects `VERS_1.23.2`, not `VERS_1.17.1`

2. **ORT v1.23.2 build** — Same FORTIFY fix applied, build succeeded (26MB vs 30MB original).
   - Version symbols match (`VERS_1.23.2`)
   - EN Moonshine: **works** (256ms for 5s audio)
   - RU GigaAM CTC: **segfault** on first transcribe request

**Root cause:** `libsherpa-onnx-c-api.so` was compiled against csukuangfj's specific ORT build.
While version symbols match, internal data structures differ when XNNPACK EP is compiled in
(additional EP registration changes class layouts). The sherpa-onnx C API binary is not
ABI-compatible with a differently-configured ORT of the same version.

**Why not worth pursuing further:**
- Full fix requires rebuilding entire chain: ORT -> sherpa-onnx -> sherpa-rs-sys -> ox-whisper
- XNNPACK only covers basic ops: MatMul, Softmax, Conv, Relu, Sigmoid, Gemm
- GigaAM's heavy ops are NOT supported by XNNPACK:
  - `DynamicQuantizeMatMul`, `DynamicQuantizeLinear`
  - `FusedMatMul`, `MatMulIntegerToFloat`, `ConvInteger`
  - `LayerNormalization`, `SkipLayerNormalization`
  - `QuickGelu`
- Expected speedup even with full integration: **5-15% at best**
- Cost: hours of build pipeline work for marginal gain

### Summary

| Optimization | EN Moonshine | RU GigaAM | Status |
|---|---|---|---|
| ORT_ENABLE_ALL | **~4x** (2.5s->0.6s) | ~10-15% | Deployed |
| inter_op=1 | Small improvement | Small improvement | Deployed |
| Batch decode | N/A | ~20-30% | Deployed |
| XNNPACK EP | Works but N/A | Segfault | Abandoned |
| **Combined** | **2.5-3.7s -> 0.6-0.8s** | **3.7-4.6s -> 2.1-3.5s** | |

### GigaAM ONNX Operator Analysis

33 unique operators in GigaAM v2 CTC int8:
```
Add, Cast, Concat, ConvInteger, Div, DynamicQuantizeLinear,
DynamicQuantizeMatMul, Equal, Expand, Floor, FusedMatMul, Gather,
LayerNormalization, Less, LogSoftmax, MatMul, MatMulIntegerToFloat,
Mul, Neg, Not, QuickGelu, Range, Relu, Reshape, Shape, Sigmoid,
SkipLayerNormalization, Slice, Softmax, Split, Transpose, Unsqueeze, Where
```

XNNPACK-supported subset: MatMul, Softmax, Relu, Sigmoid, Conv (via ConvInteger partial).
Everything else falls back to CPU provider.

### Lessons Learned

1. **Pre-built sherpa-onnx binaries are ABI-locked** to a specific ORT build configuration.
   Swapping `libonnxruntime.so` alone doesn't work even with matching version symbols.
2. **ORT_ENABLE_ALL is the single biggest optimization** — and it was commented out by default
   in sherpa-onnx's `session.cc`. Always check default session options.
3. **XNNPACK is not a silver bullet for quantized models** — most quantization-specific ops
   (DynamicQuantize*, MatMulInteger*) are not in XNNPACK's kernel registry.
4. **GitLab archive hashes are unstable** — the same commit can produce different zip hashes
   over time, breaking reproducible builds.
