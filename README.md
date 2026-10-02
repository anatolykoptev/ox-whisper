# ox-whisper — Self-Hosted Whisper API Alternative

[![Build](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml/badge.svg)](https://github.com/anatolykoptev/ox-whisper/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/Rust-2024-orange?logo=rust)](Cargo.toml)
[![Platform](https://img.shields.io/badge/platform-linux%2Faarch64-blue)](#limitations)
[![Docker](https://img.shields.io/badge/image-ghcr.io-blue?logo=docker)](https://ghcr.io/anatolykoptev/ox-whisper)

**Self-hosted, OpenAI-compatible speech-to-text (STT) HTTP server in Rust.** Drop-in replacement for the OpenAI Whisper API — runs on a single ARM64 CPU, no GPU. 25 European languages, Russian and English included, with case and punctuation, via NVIDIA Parakeet TDT 0.6B v3. Real-time WebSocket streaming. Word-level timestamps. Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

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
| **CPU RTF (aarch64, 4 threads)** | ~0.14–0.27 Parakeet · ~0.03 Moonshine | ~0.075 (tiny) | ~0.13 (tiny) | n/a |
| **GPU required** | no | optional | optional | n/a |

ox-whisper wins on a narrow but real wedge: **a polished HTTP/WebSocket server with an OpenAI-compatible API, faster than real time on a CPU, no GPU, ARM64-native.** Pair it with Pipecat / LiveKit / Vapi for self-hosted voice agents.

---

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/install.sh | bash
```

Linux aarch64. Docker auto-installed if missing. Pulls `ghcr.io/anatolykoptev/ox-whisper:latest`, fetches ~3 GB of models (Parakeet is 2.5 GB of it; `OX_WHISPER_PARAKEET=0` skips it, leaving ~463 MB), starts on `:8092` (HTTP/WS) + `:9092` (metrics). With Parakeet the container holds ~3.2 GB and peaks at ~4.3 GB on long audio; the bundled compose file allows 7 GB.

```bash
curl http://localhost:8092/health
curl http://localhost:8092/v1/models
```

<details>
<summary>docker run · docker compose · download models manually</summary>

```bash
docker run -d --name ox-whisper --restart unless-stopped \
  -p 127.0.0.1:8092:8092 -p 127.0.0.1:9092:9092 \
  --memory 7g -w /parakeet-models \
  -e PARAKEET_POOL_SIZE=1 -e PARAKEET_IDLE_EVICT_SECS=0 \
  -v $(pwd)/models/parakeet:/parakeet-models:ro \
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
    mem_limit: 7g
    working_dir: /parakeet-models   # the fp32 encoder loads encoder.weights from here
    ports:
      - "127.0.0.1:8092:8092"
      - "127.0.0.1:9092:9092"
    environment:
      MOONSHINE_THREADS: "4"
      POOL_SIZE: "2"
      PARAKEET_POOL_SIZE: "1"
      PARAKEET_IDLE_EVICT_SECS: "0"
    volumes:
      - ./models/parakeet:/parakeet-models:ro
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

| Codes | Model |
|-------|-------|
| `bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk` | Parakeet TDT 0.6B v3, when `PARAKEET_DIR` holds the model |
| `ru` without Parakeet (absent, or `ru` left out of `PARAKEET_LANGS`) | GigaAM / Zipformer-RU from `ZIPFORMER_RU_DIR` |
| every other code, and `en` without Parakeet | the one Moonshine model in `MOONSHINE_MODELS_DIR` |

`download-models.sh` fetches the **English** Moonshine model, so with the stock models only
English is transcribed correctly on that route: `ar ja vi zh` and other codes are decoded by the
English model and come back as English-sounding text, not an error
([#58](https://github.com/anatolykoptev/ox-whisper/issues/58)). Moonshine publishes separate
models per language; ox-whisper loads only one.

Parakeet is language-agnostic: a request sent with `language=ru` that carries English speech
is transcribed as English. It writes case and punctuation itself, so its output skips the
CNN-BiLSTM punctuation model. Set `PARAKEET_LANGS=off` to go back to the old models without a
rebuild (recreate the container).

**Parakeet memory and files.** The fp32 export is an `encoder.onnx` graph plus `encoder.weights`;
onnxruntime resolves the weights from the process working directory, so run the container with
`working_dir` set to the Parakeet directory (the loader refuses, and the languages fall back, if
it cannot). Size `mem_limit` for about 3.2 GB idle per `PARAKEET_POOL_SIZE` slot plus decode
buffers: one slot measured 3.5 GB peak on FLEURS clips and 4.3 GB on 5-minute files. Keep
`PARAKEET_IDLE_EVICT_SECS=0`: a reload after eviction more than doubled RSS (3.2 → 5.5 GB) and
added seconds to the next request. Use the fp32 export; the int8 one measured about 4 WER
points worse. With one slot, a second concurrent request waits (`POOL_ACQUIRE_TIMEOUT_S`)
instead of failing.

**Need 99 languages?** Use `whisper-large-v3` instead — ox-whisper trades coverage for speed and CPU footprint.

---

## Benchmarks

**Accuracy, 2026-10.** 100 Russian and 100 English FLEURS test utterances (public set, 4–23 s
each), sent through `/v1/audio/transcriptions` with VAD on as in production. The older rows
send each clip with its own `language`; the Parakeet row sends both sets with `language=ru`, as
a client that always says `ru` would. WER normalised (lowercase, punctuation stripped, `ё`→`е`); "case kept" keeps
capitalisation.

| | RU WER (case kept) | EN WER (case kept) | Empty results |
|---|---|---|---|
| Zipformer-RU + Moonshine v2 (before the VAD fixes) | 17.4% (25.4%) | 29.9% (32.8%) | 14 of 200 EN |
| Zipformer-RU + Moonshine v2, VAD fixed ([#54](https://github.com/anatolykoptev/ox-whisper/pull/54)) | 13.9% (22.3%) | 15.5% (19.7%) | 0 |
| **Parakeet TDT 0.6B v3 fp32** | **7.7% (9.5%)** | **9.7% (11.9%)** | **0** |

With VAD off, Parakeet scores 5.5% RU and 5.7% EN on the same clips; the VAD path costs the
rest ([#53](https://github.com/anatolykoptev/ox-whisper/issues/53)). On a 5-minute English file Moonshine returns almost nothing (95% WER); Parakeet
gets 38%. Details: [docs/benchmarks.md](docs/benchmarks.md).

**Speed.** Oracle Cloud A1 free tier, 4 ARM threads, CPU-only. Parakeet fp32 measured RTF
0.14–0.27 depending on load on this shared box (a 15 s note took 3.7 s end to end). Moonshine and Zipformer, best of 3 runs after warmup:

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
| `POOL_ACQUIRE_TIMEOUT_S` | `30` — how long a request waits for a busy recognizer |
| `PARAKEET_DIR` | `/parakeet-models` |
| `PARAKEET_LANGS` | all 25 Parakeet v3 languages; `off` disables Parakeet |
| `PARAKEET_POOL_SIZE` | `POOL_SIZE` |
| `PARAKEET_IDLE_EVICT_SECS` | `OX_WHISPER_IDLE_EVICT_SECS` — set `0` to keep Parakeet resident (recommended) |
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
| Moonshine v2 Base | EN (the stock download; other languages need their own model, [#58](https://github.com/anatolykoptev/ox-whisper/issues/58)) | 135 MB | [HF](https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-base-en-quantized-2026-02-27) |
| Zipformer-RU INT8 | RU | 67 MB | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases) |
| GigaAM v3 RNNT | RU | ~220 MB | [HF](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16) |
| Parakeet TDT 0.6B v3 (fp32) | 25 European languages | 2.5 GB | [HF](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3) — CC-BY-4.0 (NVIDIA) |
| Silero VAD | — | 0.6 MB | bundled |
| Punctuation CNN-BiLSTM | EN, RU | 7 MB | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases) |

Downloaded by `scripts/download-models.sh` (run automatically by `install.sh`); Parakeet's files are sha256-pinned. GigaAM is not downloaded: put it in `ZIPFORMER_RU_DIR` yourself if you want it for Russian without Parakeet.

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
| `oxwhisper_vad_no_speech_total` | counter | `caller` (`batch`, `sse`, `ws`; on `ws` it also counts checks before speech starts) |
| `oxwhisper_vad_mutex_poisoned_total` | counter | — |
| `oxwhisper_pool_acquire_wait_seconds` | histogram | — |
| `oxwhisper_pool_acquire_timeouts_total` | counter | — |
| `oxwhisper_route_fallback_total` | counter | `to` — Parakeet failed to reload after eviction |
| `oxwhisper_hallucination_rejected_total` | counter | `lang` |
| `oxwhisper_recognizer_pool_size` · `_busy` | gauge | `lang` |
| `oxwhisper_ws_active_connections` | gauge | — |

---

## Limitations

- **aarch64 only.** Pre-built `.so` libs are ARM64; x86_64 needs source rebuild of `vendor/sherpa-rs-sys`.
- **No streaming for `/transcribe`.** Whole-file responses only. Use `/transcribe/stream` (SSE) or `/v1/listen` (WebSocket) for incremental output.
- **No auth.** Bind to `0.0.0.0`; put behind nginx / Caddy if exposed to the network.
- **25 languages** (Parakeet), plus English on the Moonshine fallback. For broader coverage, use `whisper-large-v3` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or [speaches](https://github.com/speaches-ai/speaches).
- **Punctuation.** Parakeet writes case and punctuation for its 25 languages. Without Parakeet, the CNN-BiLSTM model punctuates Russian output (Zipformer, GigaAM CTC) and, on the SSE and WebSocket paths only, English; GigaAM v3 RNNT punctuates itself.
- **Memory.** Parakeet holds ~3.2 GB and peaks at ~4.3 GB on 5-minute audio. On a small box (Raspberry Pi 5, 4 GB) skip it with `OX_WHISPER_PARAKEET=0`.

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

MIT. Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0, NVIDIA), [Moonshine](https://github.com/usefulsensors/moonshine), [Silero VAD](https://github.com/snakers4/silero-vad).
