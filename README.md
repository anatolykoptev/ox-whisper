# ox-whisper — Self-Hosted Whisper API Alternative

[![Build](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml/badge.svg)](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/Rust-2024-orange?logo=rust)](Cargo.toml)
[![Platform](https://img.shields.io/badge/platform-linux%2Faarch64-blue)](#limitations)
[![Docker](https://img.shields.io/badge/image-ghcr.io-blue?logo=docker)](https://ghcr.io/anatolykoptev/ox-whisper)

**Self-hosted, OpenAI-compatible speech-to-text (STT) HTTP server in Rust.** Drop-in replacement for the OpenAI Whisper API — runs on a single ARM64 CPU, no GPU. 8 languages. Real-time WebSocket streaming. Word-level timestamps. Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) + Moonshine v2.

Built for voice AI agents, live captioning, edge deployments, and privacy-sensitive transcription on Oracle Free Tier / Hetzner CAX / Raspberry Pi 5.

[Quick start](#quick-start) · [API](#api) · [Languages](#languages) · [Benchmarks](#benchmarks) · [Metrics](#metrics) · [Limitations](#limitations)

---

## Why ox-whisper

| | ox-whisper | faster-whisper | whisper.cpp | OpenAI Whisper API |
|---|---|---|---|---|
| **Runtime** | Rust + axum | Python | C++ | Cloud |
| **Protocol** | HTTP + WebSocket | library | library / examples | HTTPS |
| **OpenAI-compatible API** | yes | no | partial | n/a |
| **Real-time WebSocket** | yes | no | no | no |
| **CPU RTF (aarch64)** | ~0.05 | ~0.075 (tiny) | ~0.13 (tiny) | n/a |
| **GPU required** | no | optional | optional | n/a |

ox-whisper wins on a narrow but real wedge: **a polished HTTP/WebSocket server with OpenAI-compatible API, sub-100 ms RTF on a CPU, no GPU, ARM64-native.** Pair it with Pipecat / LiveKit / Vapi for self-hosted voice agents.

---

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/install.sh | bash
```

Linux aarch64. Docker auto-installed if missing. Pulls `ghcr.io/anatolykoptev/ox-whisper:latest`, fetches ~463 MB models, starts on `:8092` (HTTP/WS) + `:9092` (metrics).

```bash
curl http://localhost:8092/health
curl http://localhost:8092/v1/models
```

<details>
<summary>docker run · docker compose · download models manually</summary>

```bash
docker run -d --name ox-whisper --restart unless-stopped \
  -p 127.0.0.1:8092:8092 -p 127.0.0.1:9092:9092 \
  -v $(pwd)/models/en:/models:ro \
  -v $(pwd)/models/ru:/ru-models:ro \
  -v $(pwd)/models/vad:/vad:ro \
  -v $(pwd)/models/punct-en:/punct:ro \
  ghcr.io/anatolykoptev/ox-whisper:latest
```

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

```bash
curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/scripts/download-models.sh | bash
```

</details>

---

## API

### `POST /v1/audio/transcriptions` — OpenAI-compatible

Drop-in replacement for `openai.audio.transcriptions.create`. Point your OpenAI SDK at `http://localhost:8092/v1`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8092/v1", api_key="unused")
result = client.audio.transcriptions.create(
    model="whisper-1", file=open("recording.mp3", "rb"), language="en"
)
print(result.text)
```

### `GET /v1/listen` — WebSocket real-time streaming

```python
import asyncio, websockets, json
async def stream():
    async with websockets.connect("ws://localhost:8092/v1/listen?language=en") as ws:
        await ws.send(pcm_16khz_mono_chunk)  # repeat
        async for msg in ws:
            print(json.loads(msg))  # {"type":"partial"|"final","text":"..."}
asyncio.run(stream())
```

### `POST /transcribe` — JSON, file path

```bash
curl -X POST http://localhost:8092/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_path":"/data/recording.wav","language":"en"}'
```

Response: `{ "text", "duration_ms", "words":[{"word","start","end","confidence"}], "confidence" }`. Optional fields: `vad`, `punctuate`, `max_chunk_len`.

### `POST /transcribe/upload` — multipart

```bash
curl -F file=@recording.mp3 -F language=en http://localhost:8092/transcribe/upload
```

### `POST /transcribe/stream` — SSE chunks

```bash
curl -N -F file=@long.mp3 -F language=en http://localhost:8092/transcribe/stream
# data: {"index":0,"text":"...","type":"chunk"}
# data: {"type":"done"}
```

### `GET /health` · `GET /v1/models` · `GET /metrics` (port 9092)

---

## Languages

| Code | Language   | Model |
|------|------------|-------|
| `en` | English    | Moonshine v2 Base |
| `ru` | Russian    | GigaAM v3 / Zipformer-RU INT8 (auto-detected) |
| `ar` | Arabic     | Moonshine v2 Base |
| `es` | Spanish    | Moonshine v2 Base |
| `ja` | Japanese   | Moonshine v2 Base |
| `uk` | Ukrainian  | Moonshine v2 Base |
| `vi` | Vietnamese | Moonshine v2 Base |
| `zh` | Chinese    | Moonshine v2 Base |

**Need 99 languages?** Use `whisper-large-v3` instead — ox-whisper trades coverage for speed and CPU footprint.

---

## Benchmarks

Oracle Cloud A1 free tier, 4 ARM threads, CPU-only. Best of 3 runs after warmup.

| Audio length | Language | Latency | RTF |
|---|---|---|---|
| 9.2 s | EN (Moonshine v2) | 215 ms | 0.023 |
| 9.2 s | RU (Zipformer)    | 338 ms | 0.037 |
| 123 s | EN, VAD, 44 s speech | 2.0 s | 0.05 |
| 123 s | RU, VAD, 44 s speech | 2.4 s | 0.06 |

For comparison on the same box: `faster-whisper tiny int8` ~0.075 RTF, `whisper.cpp tiny-q8_0` ~0.13 RTF.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MOONSHINE_PORT` | `8092` | HTTP/WS port |
| `MOONSHINE_THREADS` | `4` | ONNX inference threads |
| `POOL_SIZE` | `2` | Recognizer instances per language |
| `MAX_AUDIO_DURATION_S` | `0` | Max input length, `0`=unlimited |
| `VAD_MIN_DURATION_S` | `10` | Auto-enable VAD above this length |
| `OXWHISPER_PROM_PORT` | `9092` | Prometheus metrics port |

<details>
<summary>Model paths and tuning knobs</summary>

| Variable | Default |
|----------|---------|
| `MOONSHINE_MODELS_DIR` | `/models` |
| `ZIPFORMER_RU_DIR` | `/ru-models` |
| `SILERO_VAD_MODEL` | `/vad/silero_vad.onnx` |
| `PUNCT_MODEL` | `/punct/model.int8.onnx` |
| `PUNCT_VOCAB` | `/punct/bpe.vocab` |
| `VAD_THRESHOLD` | `0.5` |
| `VAD_MIN_SILENCE_S` | `0.5` |
| `VAD_SPEECH_PAD_S` | `0.05` |
| `VAD_MIN_SPEECH_S` | `0.25` |
| `VAD_MAX_CHUNK_S` | `20` |
| `MAX_CHUNK_S` | `20` |
| `HALLUCINATION_THRESHOLD` | `2.4` |
| `MAX_BODY_SIZE_MB` | `50` |
| `ONNX_PROVIDER` | `cpu` |

</details>

---

## Models

| Model | Languages | Size | Source |
|-------|-----------|------|--------|
| Moonshine v2 Base | AR · EN · ES · JA · UK · VI · ZH | 135 MB | [HF](https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-base-en-quantized-2026-02-27) |
| Zipformer-RU INT8 | RU | 67 MB | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases) |
| GigaAM v3 RNNT | RU | ~220 MB | [HF](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16) |
| Silero VAD | — | 0.6 MB | bundled |
| Punctuation CNN-BiLSTM | EN, RU | 7 MB | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases) |

Downloaded by `scripts/download-models.sh` (run automatically by `install.sh`).

---

## Metrics

Prometheus exposition on port `9092`. Scrape:

```yaml
scrape_configs:
  - job_name: ox-whisper
    static_configs:
      - targets: ['ox-whisper:9092']
```

| Metric | Type | Labels |
|--------|------|--------|
| `oxwhisper_requests_total` | counter | `endpoint`, `status` |
| `oxwhisper_request_duration_seconds` | histogram | `endpoint` |
| `oxwhisper_transcribe_duration_seconds` | histogram | `lang` |
| `oxwhisper_audio_duration_seconds` | histogram | — |
| `oxwhisper_vad_speech_ratio` | gauge | `lang` |
| `oxwhisper_chunks_total` | counter | `lang` |
| `oxwhisper_hallucination_rejected_total` | counter | `lang` |
| `oxwhisper_recognizer_pool_size` · `_busy` | gauge | `lang` |
| `oxwhisper_ws_active_connections` | gauge | — |

---

## Limitations

- **aarch64 only.** Pre-built `.so` libs are ARM64; x86_64 needs source rebuild of `vendor/sherpa-rs-sys`.
- **No streaming for `/transcribe`.** Whole-file responses only. Use `/transcribe/stream` (SSE) or `/v1/listen` (WebSocket) for incremental output.
- **No auth.** Bind to `0.0.0.0`; put behind nginx / Caddy if exposed to the network.
- **8 languages.** For broader coverage, use `whisper-large-v3` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or [speaches](https://github.com/speaches-ai/speaches).
- **EN punctuation only** via the CNN-BiLSTM model. RU punctuation comes from GigaAM v3 when loaded.

---

## Build

```bash
git clone https://github.com/anatolykoptev/ox-whisper && cd ox-whisper
cargo build --release          # native aarch64
docker build -t ox-whisper .   # BuildKit + cargo-chef layer cache
```

CI publishes `ghcr.io/anatolykoptev/ox-whisper:vX.Y.Z` on every `v*` tag.

---

## License

MIT. Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [Moonshine](https://github.com/usefulsensors/moonshine), [Silero VAD](https://github.com/snakers4/silero-vad).
