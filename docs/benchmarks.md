# ox-whisper v0.3.0 Benchmarks

Server: ARM64 (Oracle Cloud A1.Flex), 4 vCPU, 24 GB RAM, CPU-only inference.
Docker image: 760 MB. Pool size: 2 recognizers per model.

## Latency (median of 3 runs)

| File | Lang | Audio | Latency | RTF | Notes |
|------|------|-------|---------|-----|-------|
| 0.wav | EN | 6.6s | 651ms | 0.10x | Moonshine v2, 16kHz |
| 1.wav | EN | 16.7s | 766ms | 0.05x | Empty result (model limit) |
| 8k.wav | EN | 4.8s | 315ms | 0.07x | 8kHz → resampled to 16kHz |
| 0.wav | RU | 9.2s | 699ms | 0.08x | Zipformer transducer |
| 1.wav | RU | 7.1s | 565ms | 0.08x | Zipformer transducer |
| example.wav | RU | 11.3s | 712ms | 0.06x | Zipformer transducer |

**Average RTF: 0.07x** (14x faster than real-time on CPU).

## WER (Word Error Rate)

| File | Reference | Hypothesis | WER |
|------|-----------|------------|-----|
| 0.wav (EN) | "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels" | "After early nightfall,, the yellow lamps would light up here and there the squalid quarter of the brothels." | **0%** |
| 8k.wav (EN) | "yet these thoughts affected hester prynne less with hope than apprehension" | "But, these thoughts are better just to train less,, but help than apprehension." | 72.7% |

- **EN 16kHz: WER 0%** on clean speech (Moonshine v2 base)
- **EN 8kHz: WER 73%** — expected degradation, model trained on 16kHz
- **EN 16.7s: empty** — Moonshine returns empty for some longer chunks (known limitation)
- **RU: no reference transcriptions** available for WER measurement

## Throughput

Sequential processing, single connection:
- **10 requests (6.6s EN audio each) in 7.1s**
- **1.4 req/s**
- **9.3x realtime throughput** (66s audio processed in 7.1s)

## Auto-Detection Overhead

| Mode | Latency |
|------|---------|
| Explicit `language=en` | ~820ms |
| Auto-detect | ~790ms |
| **Overhead** | **~0ms** (within noise) |

Language detection runs on first 3s of audio — negligible compared to full transcription.

## Known Limitations

1. **Moonshine v2 long audio**: Returns empty for some 15-17s single chunks. Workaround: VAD splits into shorter segments.
2. **8kHz audio**: Significant WER degradation. Input is resampled to 16kHz but information is lost.
3. **RU benchmarks**: Need CommonVoice/Golos test set for proper WER measurement.

## Comparison Context

| Metric | ox-whisper (CPU) | Deepgram Nova-3 (GPU) | AssemblyAI Best (GPU) |
|--------|-----------------|----------------------|----------------------|
| RTF | 0.07x | ~0.003x | ~0.01x |
| WER EN (clean) | 0% (small sample) | ~8% (LibriSpeech) | ~5% (U3-Pro) |
| Deployment | Self-hosted, 760MB | Cloud API | Cloud API |
| Latency (6.6s audio) | 650ms | ~200ms | ~500ms (async) |
| Cost | $0 (self-hosted) | $0.0043/min | $0.0062/min |
