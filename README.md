[![Build](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml/badge.svg)](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/Rust-2024%20edition-orange?logo=rust)](Cargo.toml)
[![Platform](https://img.shields.io/badge/platform-linux%2Faarch64-blue)](#limitations)
[![Docker](https://img.shields.io/badge/image-ghcr.io%2Fanatolykoptev%2Fox--whisper-blue?logo=docker)](https://ghcr.io/anatolykoptev/ox-whisper)

# ox-whisper

Fast multilingual speech-to-text HTTP server — 8 languages, Silero VAD, word timestamps, punctuation, CPU-only.

Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) v1.12.28 with Rust + axum. Drop-in **OpenAI-compatible** `/v1/audio/transcriptions` endpoint, **SSE streaming**, and a real-time **WebSocket** API.

> **Successor to [moonshine-whisper](https://github.com/anatolykoptev/moonshine-whisper)** (Go). ox-whisper is a Rust rewrite with multilingual support, VAD, punctuation, OpenAI compatibility, and WebSocket streaming.

---

## Why ox-whisper

| | ox-whisper | faster-whisper | whisper.cpp | OpenAI Whisper API |
|---|---|---|---|---|
| **Runtime** | Rust (axum) | Python | C++ | Cloud |
| **Protocol** | HTTP + WebSocket | library | library / server | HTTPS |
| **OpenAI compat** | yes | no (library) | partial | n/a |
| **WebSocket stream** | yes | no | no | no |
| **CPU RTF (aarch64)** | ~0.05 | ~0.075 tiny | ~0.13 tiny | n/a |
| **GPU required** | no | optional | optional | n/a |
| **Self-hosted** | yes | yes | yes | no |

Use ox-whisper when you need an always-on, low-latency HTTP/WebSocket endpoint on ARM64 hardware without a GPU — voice pipelines, real-time captioning, edge deployments.

---

## Features

- **8 languages** — EN, RU, AR, ES, JA, UK, VI, ZH
- **OpenAI-compatible API** — `/v1/audio/transcriptions` and `/v1/models` work with any OpenAI SDK
- **Real-time WebSocket** — send 16 kHz mono PCM, receive interim and final transcripts
- **SSE streaming** — VAD-segmented chunks delivered as Server-Sent Events
- **Silero VAD** — auto-segments long audio, skips silence
- **Word-level timestamps** and per-word confidence scores
- **Punctuation & truecasing** — CNN-BiLSTM model, automatic for EN
- **Hallucination guard** — compression-ratio filter rejects garbage chunks
- **Any input format** — ffmpeg converts mp3, ogg, flac, m4a, mp4, wav…
- **Recognizer pool** — warm model instances shared across concurrent requests
- **Prometheus metrics** — 10 metric families on a dedicated port (`:9092`)
- **Zero GPU** — pure CPU inference, runs on Oracle Cloud A1 free tier

---

## Quick start

One-line install (Linux aarch64 — Docker auto-installed if missing):

```bash
curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/install.sh | bash
```

Downloads `docker-compose.yml`, fetches ~463 MB of ASR models, pulls `ghcr.io/anatolykoptev/ox-whisper:latest`, and starts the container on port `8092` (HTTP/WS) + `9092` (metrics).

Verify:

```bash
curl http://localhost:8092/health
curl http://localhost:8092/v1/models
```

<details>
<summary>docker run (manual)</summary>

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

Download models separately:

```bash
curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/scripts/download-models.sh | bash
```

</details>

<details>
<summary>docker compose</summary>

```yaml
services:
  ox-whisper:
    image: ghcr.io/anatolykoptev/ox-whisper:latest
    restart: unless-stopped
    ports:
      - "127.0.0.1:8092:8092"
      - "127.0.0.1:9092:9092"
    environment:
      MOONSHINE_THREADS: "4"
      POOL_SIZE: "2"
    volumes:
      - ./models/en:/models:ro
      - ./models/ru:/ru-models:ro
      - ./models/vad:/vad:ro
      - ./models/punct-en:/punct:ro
```

</details>

---

## Supported languages

| Code | Language   | Model                         |
|------|------------|-------------------------------|
| `en` | English    | Moonshine v2 Base             |
| `ru` | Russian    | GigaAM v3 / Zipformer-RU INT8 |
| `ar` | Arabic     | Moonshine v2 Base             |
| `es` | Spanish    | Moonshine v2 Base             |
| `ja` | Japanese   | Moonshine v2 Base             |
| `uk` | Ukrainian  | Moonshine v2 Base             |
| `vi` | Vietnamese | Moonshine v2 Base             |
| `zh` | Chinese    | Moonshine v2 Base             |

RU model is auto-detected at startup: GigaAM v3 RNNT (built-in punctuation) → GigaAM v2 CTC → Zipformer-RU INT8.

---

## Benchmarks

Oracle Cloud A1, 4 threads, CPU-only. Best of 3 runs after warmup. The Dockerfile applies `ORT_ENABLE_ALL` graph optimizations — ~4× speedup for Moonshine, ~30% for Zipformer.

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

For comparison — `faster-whisper tiny int8` clocks RTF ≈ 0.075 and `whisper.cpp tiny-q8_0` ≈ 0.13 on the same box.

---

## API reference

### Health

```bash
GET /health
```

Returns status and per-language model readiness.

---

### Native — JSON (file path)

```bash
curl -X POST http://localhost:8092/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_path":"/data/recording.wav","language":"en"}'
```

Optional fields: `vad` (bool), `punctuate` (bool), `max_chunk_len` (int).

Response:

```json
{
  "text": "transcribed text",
  "duration_ms": 280,
  "words": [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.95}],
  "confidence": 0.93
}
```

---

### Native — multipart upload

```bash
curl -F file=@recording.mp3 -F language=en \
  http://localhost:8092/transcribe/upload
```

Same response shape as above.

---

### Native — SSE streaming

```bash
curl -N -F file=@long.mp3 -F language=en \
  http://localhost:8092/transcribe/stream
```

Emits `text/event-stream` chunks as VAD segments are decoded:

```
data: {"index":0,"text":"Hello world","type":"chunk"}

data: {"index":1,"text":"How are you","type":"chunk"}

data: {"type":"done"}
```

---

### OpenAI-compatible

Drop-in replacement for `openai.audio.transcriptions.create`. Set `base_url=http://localhost:8092/v1`.

```bash
curl -X POST http://localhost:8092/v1/audio/transcriptions \
  -F file=@recording.mp3 \
  -F model=whisper-1 \
  -F language=en \
  -F response_format=json
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8092/v1",
    api_key="unused",
)

with open("recording.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="en",
    )

print(result.text)
```

`GET /v1/models` lists available languages as model IDs.

---

### WebSocket — real-time streaming

Connect to `ws://localhost:8092/v1/listen?language=en`. Send raw 16 kHz mono PCM frames. Receive JSON events with interim and final transcripts.

```python
import asyncio
import websockets

async def stream_mic():
    uri = "ws://localhost:8092/v1/listen?language=en"
    async with websockets.connect(uri) as ws:
        # Send 16 kHz mono PCM chunks (e.g. 100 ms = 1600 samples * 2 bytes)
        chunk = b"\x00" * 3200  # replace with real audio
        await ws.send(chunk)

        async for message in ws:
            import json
            event = json.loads(message)
            # {"type": "partial", "text": "hello"}
            # {"type": "final",   "text": "hello world"}
            print(event)

asyncio.run(stream_mic())
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MOONSHINE_PORT` | `8092` | HTTP/WS server port |
| `MOONSHINE_MODELS_DIR` | `/models` | EN/multilingual models directory |
| `ZIPFORMER_RU_DIR` | `/ru-models` | RU models directory |
| `MOONSHINE_THREADS` | `4` | ONNX inference threads |
| `POOL_SIZE` | `2` | Recognizer instances per model |
| `MAX_AUDIO_DURATION_S` | `0` | Max input length in seconds (`0` = unlimited) |
| `VAD_MIN_DURATION_S` | `10` | Auto-enable VAD above this audio length |
| `OXWHISPER_PROM_PORT` | `9092` | Prometheus metrics port |

<details>
<summary>Full environment variable reference</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `SILERO_VAD_MODEL` | `/vad/silero_vad.onnx` | VAD model path |
| `PUNCT_MODEL` | `/punct/model.int8.onnx` | Punctuation model |
| `PUNCT_VOCAB` | `/punct/bpe.vocab` | Punctuation vocabulary |
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold |
| `VAD_MIN_SILENCE_S` | `0.5` | Min silence duration to split on |
| `VAD_SPEECH_PAD_S` | `0.05` | Padding around detected speech |
| `VAD_MIN_SPEECH_S` | `0.25` | Min speech duration to keep |
| `VAD_MAX_CHUNK_S` | `20` | Max VAD chunk length in seconds |
| `MAX_CHUNK_S` | `20` | Max non-VAD chunk length |
| `HALLUCINATION_THRESHOLD` | `2.4` | Compression-ratio guard cutoff |
| `MAX_BODY_SIZE_MB` | `50` | Upload size limit |
| `ONNX_PROVIDER` | `cpu` | ONNX execution provider |

</details>

---

## Models

| Model | Languages | Size | Source |
|-------|-----------|------|--------|
| [Moonshine v2 Base](https://github.com/usefulsensors/moonshine) | AR, EN, ES, JA, UK, VI, ZH | 135 MB | [HuggingFace](https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-base-en-quantized-2026-02-27) |
| [Zipformer-RU INT8](https://github.com/k2-fsa/sherpa-onnx) | RU | 67 MB | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2) |
| [GigaAM v3 RNNT](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16) | RU | ~220 MB | HuggingFace |
| [Silero VAD](https://github.com/snakers4/silero-vad) | — | 0.6 MB | bundled |
| [Punctuation CNN-BiLSTM](https://github.com/k2-fsa/sherpa-onnx) | EN, RU | 7 MB | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2) |

---

## Metrics

Prometheus exposition on port `9092` (override: `OXWHISPER_PROM_PORT`).

| Metric | Type | Labels |
|--------|------|--------|
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

Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: ox-whisper
    static_configs:
      - targets: ['ox-whisper:9092']
```

---

## Architecture

```
audio input (any format)
       │
       ▼
  ffmpeg decode ──────► 16 kHz mono PCM
       │
       ▼
  Silero VAD ──────────► chunk boundaries
       │
       ▼
  chunker (VAD / fixed)
       │
       ▼
  recognizer pool ◄──── per-language warm model instances
  (sherpa-onnx ONNX)
       │
       ├─► hallucination guard (compression ratio)
       │
       ▼
  punctuation (CNN-BiLSTM, EN/RU)
       │
       ├─► word timestamps + confidence
       │
       ├─► JSON response          (POST /transcribe, /transcribe/upload)
       ├─► SSE chunks             (POST /transcribe/stream)
       ├─► OpenAI response        (POST /v1/audio/transcriptions)
       └─► WebSocket events       (GET /v1/listen)
```

---

## Limitations

- **aarch64 only** — the vendored `sherpa-onnx` shared libraries (`.so`) are pre-compiled for ARM64. x86_64 requires recompiling `vendor/sherpa-rs` from source.
- **No real-time HTTP** — `/transcribe` and `/transcribe/upload` process the whole file before responding. Streaming is only via `/transcribe/stream` (SSE) or `/v1/listen` (WebSocket).
- **Diarization disabled by default** — diarization models are not bundled in the Docker image; the code path exists but is gated behind model availability.
- **No auth / multitenancy** — ox-whisper binds to `0.0.0.0` and has no API key or user isolation. Deploy behind a reverse proxy (nginx, Caddy) if exposing to the network.
- **English punctuation only** — the CNN-BiLSTM punctuation model supports EN. RU punctuation is handled by GigaAM v3 RNNT when that model is loaded.

---

## Building from source

```bash
git clone https://github.com/anatolykoptev/ox-whisper
cd ox-whisper
cargo build --release      # native aarch64 build
```

Docker (BuildKit required — uses cargo-chef layer caching):

```bash
docker build -t ox-whisper .
```

The Dockerfile patches `vendor/sherpa-rs-sys` to enable `ORT_ENABLE_ALL` graph optimizations at compile time. CI builds the aarch64 release binary on every push — see [Actions](https://github.com/anatolykoptev/ox-whisper/actions).

---

## Acknowledgements

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — ONNX inference engine
- [usefulsensors/moonshine](https://github.com/usefulsensors/moonshine) — multilingual ASR models
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) — voice activity detection
- [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) — RTF baseline reference

---

## License

MIT — see [LICENSE](LICENSE).
