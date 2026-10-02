# ox-whisper Benchmarks

## 2026-10: FLEURS accuracy, Parakeet TDT 0.6B v3 vs the old models

Server: ARM64 (Oracle Cloud A1.Flex), 4 vCPU, 24 GB RAM, CPU-only, a shared box under some load.

**Method.**
- **Clips:** 100 Russian and 100 English FLEURS test utterances, the same set for every row; 16 kHz.
- **Request:** each clip goes through `/v1/audio/transcriptions` in a separate container, with VAD on as in production. Rows marked `ru` send every clip with `language=ru`.
- **WER:** normalised (lowercase, punctuation stripped, `ё`→`е`). "Case kept" scores capitalisation as well.
- **Empty:** a 200 response whose text is empty. English clips were sent twice.
- **Reference:** NVIDIA Parakeet q8_0 through whisper.cpp (ox-say), one-shot on the same clips: RU 5.58%, EN 5.50%.

| Configuration | RU WER (case kept) | EN WER (case kept) | Empty EN / 200 | RTF | RSS idle / peak |
|---|---|---|---|---|---|
| Zipformer-RU + Moonshine v2, before #54 | 17.39% (25.42%) | 29.88% (32.77%) | 14 | 0.03 | 0.8 / 1.1 GB |
| Zipformer-RU + Moonshine v2, after #54 | 13.89% (22.25%) | 15.47% (19.73%) | 0 | 0.03 | 0.8 / 1.1 GB |
| **Parakeet fp32, `ru`** | **7.65% (9.46%)** | **9.73% (11.92%)** | **0** | 0.14–0.27 ¹ | 3.2 / 3.5 GB |
| Parakeet fp32, VAD off | 5.52% (7.16%) | 5.74% (7.42%) | 0 | — | — |
| Parakeet int8, VAD off | 9.68% (11.15%) | 9.73% (11.78%) | 0 | 0.07 | 2.5 GB at pool 2 |
| Parakeet fp16 (third-party export) | 7.65% | 9.93% | 2 | 0.21 | 3.1 / 4.1 GB |
| GigaAM v3 CTC punct, VAD off (RU only) | 4.54% (6.34%) | — | — | 0.05 | 1.25 GB |

¹ Measured on this build under varying load on a shared host: 0.14 in the #52 run, 0.18 (EN) and 0.27 (RU) in the post-deploy probe. Upstream sherpa-onnx (pip) ran the same model at 0.17–0.18 on an idle host.

**5-minute concatenated files, VAD on.**

| Configuration | RU WER | EN WER | Peak RSS |
|---|---|---|---|
| Zipformer-RU + Moonshine v2, before #54 | 19.3% | 97.3% | ~1.6 GB |
| Zipformer-RU + Moonshine v2, after #54 | 14.1% | 94.9% | 2.1 GB |
| Parakeet fp32 | 18.2% | 37.6% | 4.3 GB |

The peak RSS of the two old-model rows is the process high-water mark under the 1.5 GB and 2.5 GB container limits of those runs (the first was swapping), so neither is an uncapped peak.

**Findings.**
- **The empty results came from the shared Silero VAD.** It carried state from one request to the next, and its 0.5 s `max_speech_duration` shredded speech into short segments. Both are fixed in #54. The Moonshine-only empties on 15–17 s single chunks, noted in the v0.3.0 section below, are a separate limit.
- **Parakeet with `language=ru` transcribes English speech as English.** The language parameter only picks the route.
- **The int8 export is about 4 points worse than fp32.** The same int8 file shows the same errors in upstream sherpa-onnx 1.13.8, so the export is at fault. fp32 is within 0.25 points of the whisper.cpp q8_0 reference.
- **fp16 saves nothing on this CPU.** onnxruntime upcasts it, so it needs the same RAM, has a higher peak, and runs about 50% slower.
- **A reload after idle eviction raised Parakeet RSS from 3.2 to 5.5 GB.** Run it with `PARAKEET_IDLE_EVICT_SECS=0`.
- **The VAD path still costs 2–4 points against VAD off** (#53).

## v0.3.0 (2026-03)

Server: ARM64 (Oracle Cloud A1.Flex), 4 vCPU, 24 GB RAM, CPU-only inference.
Docker image: 760 MB. Pool size: 2 recognizers per model.

### Latency (median of 3 runs)

| File | Lang | Audio | Latency | RTF | Notes |
|------|------|-------|---------|-----|-------|
| 0.wav | EN | 6.6s | 651ms | 0.10x | Moonshine v2, 16kHz |
| 1.wav | EN | 16.7s | 766ms | 0.05x | Empty result (model limit) |
| 8k.wav | EN | 4.8s | 315ms | 0.07x | 8kHz → resampled to 16kHz |
| 0.wav | RU | 9.2s | 699ms | 0.08x | Zipformer transducer |
| 1.wav | RU | 7.1s | 565ms | 0.08x | Zipformer transducer |
| example.wav | RU | 11.3s | 712ms | 0.06x | Zipformer transducer |

**Average RTF: 0.07x** (14x faster than real-time on CPU).

### WER (Word Error Rate)

| File | Reference | Hypothesis | WER |
|------|-----------|------------|-----|
| 0.wav (EN) | "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels" | "After early nightfall,, the yellow lamps would light up here and there the squalid quarter of the brothels." | **0%** |
| 8k.wav (EN) | "yet these thoughts affected hester prynne less with hope than apprehension" | "But, these thoughts are better just to train less,, but help than apprehension." | 72.7% |

- **EN 16kHz: WER 0%** on clean speech (Moonshine v2 base)
- **EN 8kHz: WER 73%** — expected degradation, model trained on 16kHz
- **EN 16.7s: empty** — Moonshine returns empty for some longer chunks (known limitation)
- **RU: no reference transcriptions** available for WER measurement

### Throughput

Sequential processing, single connection:
- **10 requests (6.6s EN audio each) in 7.1s**
- **1.4 req/s**
- **9.3x realtime throughput** (66s audio processed in 7.1s)

### Auto-Detection Overhead

| Mode | Latency |
|------|---------|
| Explicit `language=en` | ~820ms |
| Auto-detect | ~790ms |
| **Overhead** | **~0ms** (within noise) |

Language detection runs on first 3s of audio — negligible compared to full transcription.

### Known Limitations

1. **Moonshine v2 long audio**: Returns empty for some 15-17s single chunks. Workaround: VAD splits into shorter segments.
2. **8kHz audio**: Significant WER degradation. Input is resampled to 16kHz but information is lost.
3. **RU benchmarks**: Need CommonVoice/Golos test set for proper WER measurement.

### Comparison Context

| Metric | ox-whisper (CPU) | Deepgram Nova-3 (GPU) | AssemblyAI Best (GPU) |
|--------|-----------------|----------------------|----------------------|
| RTF | 0.07x | ~0.003x | ~0.01x |
| WER EN (clean) | 0% (small sample) | ~8% (LibriSpeech) | ~5% (U3-Pro) |
| Deployment | Self-hosted, 760MB | Cloud API | Cloud API |
| Latency (6.6s audio) | 650ms | ~200ms | ~500ms (async) |
| Cost | $0 (self-hosted) | $0.0043/min | $0.0062/min |
