# ox-whisper

Speech-to-text HTTP server. 8 languages, Silero VAD, word timestamps, punctuation. CPU-only, no GPU.

Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) v1.12.28 (Rust + axum).

## Benchmark

ARM64 (Oracle Cloud A1), 4 threads, CPU:

| Audio | Latency | RTF |
|-------|---------|-----|
| RU 7s | 200ms | 0.03 |
| RU 9s | 280ms | 0.03 |
| EN 9s | 230ms | 0.03 |
| RU 2min (VAD, 44s speech) | 2.5s | 0.06 |

Dockerfile applies ORT graph optimizations (`ORT_ENABLE_ALL`) — ~4x speedup for Moonshine, ~30% for Zipformer.

## Usage

```bash
# File path (server-side)
curl -X POST http://localhost:8092/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_path":"/data/recording.wav","language":"ru"}'

# File upload
curl -F file=@recording.mp3 -F language=en http://localhost:8092/transcribe/upload
```

Response:

```json
{
  "text": "transcribed text",
  "duration_ms": 280,
  "words": [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.95}],
  "confidence": 0.93
}
```

Optional fields: `vad` (bool), `punctuate` (bool), `max_chunk_len` (int).

SSE streaming: `POST /transcribe/stream` — same as upload, returns `{"type":"chunk",...}` events.

## Models

| Model | Languages | Size | Download |
|-------|-----------|------|----------|
| [Moonshine v2 Base](https://github.com/usefulsensors/moonshine) | AR, EN, ES, JA, UK, VI, ZH | 135 MB | [HuggingFace](https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-base-en-quantized-2026-02-27) |
| [Zipformer-RU INT8](https://github.com/k2-fsa/sherpa-onnx) | RU | 67 MB | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2) |
| [GigaAM v3 RNNT](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16) | RU | ~220 MB | HuggingFace |
| [Silero VAD](https://github.com/snakers4/silero-vad) | — | 0.6 MB | bundled |
| [Punctuation CNN-BiLSTM](https://github.com/k2-fsa/sherpa-onnx) | EN, RU | 7 MB | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2) |

RU model auto-detected at startup: GigaAM v3 (built-in punctuation) > GigaAM v2 CTC > Zipformer.

## Docker

```yaml
moonshine:
  build: /path/to/ox-whisper
  ports:
    - "8092:8092"
  volumes:
    - ./models/en:/models:ro
    - ./models/ru:/ru-models:ro
    - ./models/vad:/vad:ro
    - ./models/punct:/punct:ro
```

Image size: ~760 MB (includes sherpa-onnx + ONNX Runtime).

## Configuration

| Variable | Default | |
|----------|---------|---|
| `MOONSHINE_PORT` | `8092` | Server port |
| `MOONSHINE_MODELS_DIR` | `/models` | EN models directory |
| `ZIPFORMER_RU_DIR` | `/ru-models` | RU models directory |
| `MOONSHINE_THREADS` | `4` | Inference threads |
| `POOL_SIZE` | `2` | Recognizer instances per model |
| `MAX_AUDIO_DURATION_S` | `0` | Max duration, 0 = no limit |
| `VAD_MIN_DURATION_S` | `10` | Auto-enable VAD above this |

<details>
<summary>All env vars</summary>

| Variable | Default | |
|----------|---------|---|
| `SILERO_VAD_MODEL` | `/vad/silero_vad.onnx` | VAD model path |
| `PUNCT_MODEL` | `/punct/model.int8.onnx` | Punctuation model |
| `PUNCT_VOCAB` | `/punct/bpe.vocab` | Punctuation vocab |
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold |
| `VAD_MIN_SILENCE_S` | `0.5` | Min silence to split |
| `VAD_SPEECH_PAD_S` | `0.05` | Padding around speech |
| `VAD_MIN_SPEECH_S` | `0.25` | Min speech duration |
| `VAD_MAX_CHUNK_S` | `20` | Max VAD chunk seconds |
| `MAX_CHUNK_S` | `20` | Max non-VAD chunk seconds |
| `HALLUCINATION_THRESHOLD` | `2.4` | Compression ratio guard |
| `MAX_BODY_SIZE_MB` | `50` | Upload size limit |
| `ONNX_PROVIDER` | `cpu` | ONNX execution provider |

</details>
