# ox-whisper

[![Build](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml/badge.svg)](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024%20edition-orange?logo=rust)](Cargo.toml)
[![Platform](https://img.shields.io/badge/platform-linux%2Faarch64-blue)](#requirements)

Fast multilingual speech-to-text HTTP server. **8 languages**, Silero VAD, word timestamps, punctuation. CPU-only — no GPU required.

Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) v1.12.28 with Rust + axum. Drop-in **OpenAI-compatible** `/v1/audio/transcriptions` endpoint plus a real-time **WebSocket** streaming API.

> **Successor to [moonshine-whisper](https://github.com/anatolykoptev/moonshine-whisper)** (Go). ox-whisper is a Rust rewrite with multilingual support, VAD, punctuation, OpenAI compatibility and WebSocket streaming.

## Features

- **8 languages** out of the box — EN, RU, AR, ES, JA, UK, VI, ZH
- **OpenAI-compatible API** — point any OpenAI client at it (`/v1/audio/transcriptions`, `/v1/models`)
- **WebSocket streaming** — real-time transcription via `/v1/listen`
- **Silero VAD** — auto-segments long audio, skips silence
- **Word-level timestamps** + per-word confidence
- **Punctuation & truecasing** — CNN-BiLSTM model, automatic for EN
- **Hallucination guard** — compression-ratio filter on each chunk
- **Any input format** — ffmpeg auto-converts mp3 / ogg / flac / m4a / mp4 / wav…
- **Recognizer pool** — concurrent requests share warm model instances
- **All-in-memory** — models stay loaded, no per-request cold start

## Languages

| Code | Language    | Model                         |
|------|-------------|-------------------------------|
| `en` | English     | Moonshine v2 Base             |
| `ru` | Russian     | GigaAM v3 / Zipformer-RU INT8 |
| `ar` | Arabic      | Moonshine v2 Base             |
| `es` | Spanish     | Moonshine v2 Base             |
| `ja` | Japanese    | Moonshine v2 Base             |
| `uk` | Ukrainian   | Moonshine v2 Base             |
| `vi` | Vietnamese  | Moonshine v2 Base             |
| `zh` | Chinese     | Moonshine v2 Base             |

RU model auto-detected at startup: GigaAM v3 (built-in punctuation) → GigaAM v2 CTC → Zipformer.

## Benchmark

ARM64 (Oracle Cloud A1), 4 threads, CPU only. Best of 3 runs after warmup. Dockerfile applies ORT graph optimizations (`ORT_ENABLE_ALL`) — ~4× speedup for Moonshine, ~30% for Zipformer.

**Short audio (7–9 s):**

| Language | Model        | 9.2 s audio | 7.1 s audio |
|----------|--------------|-------------|-------------|
| EN       | Moonshine v2 | 215 ms      | 549 ms      |
| RU       | Zipformer    | 338 ms      | 461 ms      |
| AR       | Moonshine v2 | 195 ms      | 586 ms      |
| ES       | Moonshine v2 | 190 ms      | 636 ms      |
| JA       | Moonshine v2 | 207 ms      | 382 ms      |
| UK       | Moonshine v2 | 184 ms      | 450 ms      |
| VI       | Moonshine v2 | 221 ms      | 406 ms      |
| ZH       | Moonshine v2 | 164 ms      | 498 ms      |

**Long audio (123 s, VAD enabled, 44 s of speech):**

| Language | Latency | RTF  |
|----------|---------|------|
| EN       | 2.0 s   | 0.05 |
| RU       | 2.4 s   | 0.06 |

For comparison — `faster-whisper tiny int8` clocks RTF ≈ 0.075, `whisper.cpp tiny-q8_0` ≈ 0.13 on the same box.

## Quick start

One-line install (Linux aarch64, requires Docker — auto-installs if missing):

```bash
curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/install.sh | bash
```

This downloads `docker-compose.yml`, fetches ~463 MB of ASR models, pulls `ghcr.io/anatolykoptev/ox-whisper:latest`, and starts the container on port `8092` (HTTP/WS) + `9092` (metrics).

### Manual Docker

```bash
docker run -d \
  --name ox-whisper \
  --restart unless-stopped \
  -p 127.0.0.1:8092:8092 \
  -p 127.0.0.1:9092:9092 \
  -v $(pwd)/models/en:/models:ro \
  -v $(pwd)/models/ru:/ru-models:ro \
  -v $(pwd)/models/vad:/vad:ro \
  -v $(pwd)/models/punct-en:/punct:ro \
  ghcr.io/anatolykoptev/ox-whisper:latest
```

Models can be downloaded separately via `scripts/download-models.sh`.

## API

### Native — JSON / file path

```bash
curl -X POST http://localhost:8092/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_path":"/data/recording.wav","language":"ru"}'
```

### Native — multipart upload

```bash
curl -F file=@recording.mp3 -F language=en \
  http://localhost:8092/transcribe/upload
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

### Native — Server-Sent Events streaming

```bash
curl -N -F file=@long.mp3 -F language=en \
  http://localhost:8092/transcribe/stream
```

Returns `{"type":"chunk", ...}` events as VAD segments are decoded.

### OpenAI-compatible

Drop-in replacement for `openai.audio.transcriptions.create()`. Set `base_url=http://localhost:8092/v1` in your client.

```bash
curl -X POST http://localhost:8092/v1/audio/transcriptions \
  -F file=@recording.mp3 \
  -F model=whisper-1 \
  -F language=en \
  -F response_format=json
```

`GET /v1/models` lists available languages as model IDs.

### WebSocket — real-time streaming

Connect to `ws://localhost:8092/v1/listen?language=en`, send 16 kHz mono PCM frames, receive partial + final transcripts.

```python
import websockets, asyncio
async def stream():
    async with websockets.connect("ws://localhost:8092/v1/listen?language=en") as ws:
        # send raw 16k mono PCM, receive JSON events
        ...
```

## Metrics

Prometheus `/metrics` on port `9092` (override via `OXWHISPER_PROM_PORT`).

| Metric | Type | Labels |
|---|---|---|
| `oxwhisper_requests_total` | counter | `endpoint`, `status` |
| `oxwhisper_request_duration_seconds` | histogram | `endpoint` |
| `oxwhisper_transcribe_duration_seconds` | histogram | `lang` |
| `oxwhisper_audio_duration_seconds` | histogram | — |
| `oxwhisper_vad_speech_ratio` | gauge | `lang` |
| `oxwhisper_chunks_total` | counter | `lang` |
| `oxwhisper_hallucination_rejected_total` | counter | `lang` |
| `oxwhisper_recognizer_pool_size` | gauge | `lang` |
| `oxwhisper_recognizer_pool_busy` | gauge | `lang` |
| `oxwhisper_ws_active_connections` | gauge | — |

Scrape config:

```yaml
scrape_configs:
  - job_name: ox-whisper
    static_configs:
      - targets: ['ox-whisper:9092']
```

## Models

| Model | Languages | Size | Source |
|-------|-----------|------|--------|
| [Moonshine v2 Base](https://github.com/usefulsensors/moonshine) | AR, EN, ES, JA, UK, VI, ZH | 135 MB | [HuggingFace](https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-base-en-quantized-2026-02-27) |
| [Zipformer-RU INT8](https://github.com/k2-fsa/sherpa-onnx) | RU | 67 MB | [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2) |
| [GigaAM v3 RNNT](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16) | RU | ~220 MB | HuggingFace |
| [Silero VAD](https://github.com/snakers4/silero-vad) | — | 0.6 MB | bundled |
| [Punctuation CNN-BiLSTM](https://github.com/k2-fsa/sherpa-onnx) | EN, RU | 7 MB | [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2) |

## Configuration

| Variable | Default | |
|----------|---------|---|
| `MOONSHINE_PORT` | `8092` | Server port |
| `MOONSHINE_MODELS_DIR` | `/models` | EN/multilingual models directory |
| `ZIPFORMER_RU_DIR` | `/ru-models` | RU models directory |
| `MOONSHINE_THREADS` | `4` | Inference threads |
| `POOL_SIZE` | `2` | Recognizer instances per model |
| `MAX_AUDIO_DURATION_S` | `0` | Max input duration (`0` = unlimited) |
| `VAD_MIN_DURATION_S` | `10` | Auto-enable VAD above this length |

<details>
<summary>Full env reference</summary>

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

## Requirements

- **Linux / aarch64** (ARM64). The vendored `sherpa-onnx` shared libraries are pre-compiled for ARM64; x86_64 support requires recompiling `vendor/sherpa-rs`.
- `ffmpeg` (bundled in the Docker image) for input format conversion.

## Build from source

```bash
git clone https://github.com/anatolykoptev/ox-whisper
cd ox-whisper
cargo build --release        # native build, aarch64 host
docker build -t ox-whisper . # cross-build via QEMU
```

CI builds aarch64 release binaries on every tag — see [Actions](https://github.com/anatolykoptev/ox-whisper/actions).

## Project layout

| Path | Role |
|------|------|
| `src/main.rs` | axum server, route wiring |
| `src/handlers.rs` | native HTTP handlers |
| `src/handler_openai.rs` | OpenAI-compatible endpoint |
| `src/handler_stream.rs` | SSE streaming |
| `src/ws_handler.rs` | WebSocket `/v1/listen` |
| `src/transcribe.rs` | sherpa-onnx engine |
| `src/audio.rs` | ffmpeg audio conversion |
| `src/chunking.rs` | long-audio chunking + VAD |
| `src/vad.rs` | Silero VAD |
| `src/punctuate.rs` | punctuation restoration |
| `src/models.rs` | model loading + auto-detect |
| `vendor/sherpa-rs/` | vendored sherpa-onnx Rust bindings |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — inference engine
- [usefulsensors/moonshine](https://github.com/usefulsensors/moonshine) — multilingual ASR models
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) — voice activity detection
- [salesforce/CTranslate2](https://github.com/OpenNMT/CTranslate2) — RTF baseline reference
