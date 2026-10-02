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
| **CPU RTF (aarch64, 4 threads)** | ~0.14–0.27 | ~0.075 (tiny) | ~0.13 (tiny) | n/a |
| **GPU required** | no | optional | optional | n/a |

ox-whisper wins on a narrow but real wedge: **a polished HTTP/WebSocket server with an OpenAI-compatible API, faster than real time on a CPU, no GPU, ARM64-native.** Pair it with Pipecat / LiveKit / Vapi for self-hosted voice agents.

---

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/install.sh | bash
```

Linux aarch64. Docker auto-installed if missing. Pulls `ghcr.io/anatolykoptev/ox-whisper:latest`, fetches ~2.5 GB of models (Parakeet and the VAD), starts on `:8092` (HTTP/WS) + `:9092` (metrics). The container holds ~3.2 GB and peaks at ~4.3 GB on long audio; the bundled compose file allows 7 GB.

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
  -v $(pwd)/models/vad:/vad:ro \
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
      PARAKEET_POOL_SIZE: "1"
      PARAKEET_IDLE_EVICT_SECS: "0"
    volumes:
      - ./models/parakeet:/parakeet-models:ro
      - ./models/vad:/vad:ro
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

### `GET /health` · `GET /v1/models` · `GET /metrics` (port 9092)

---

## Languages

One model, Parakeet TDT 0.6B v3, serves every request: `bg cs da de el en es et fi fr hr hu it
lt lv mt nl pl pt ro ru sk sl sv uk`. It writes case and punctuation itself.

The `language` field is a hint, never a route. Parakeet transcribes the language it hears, so a
request sent with `language=ru` that carries English speech is transcribed as English. What the
field does:

| `language` | Result |
|---|---|
| absent, empty, `auto` | transcribed; `verbose_json` carries no `language` |
| one of the 25 codes above, or a tag whose first part is one (`ru-RU`, `en_US`) | transcribed; `verbose_json` echoes the code |
| anything else (`zh`, `ja`, `ar`, `vi`, ...) | **400**, OpenAI-style error body (`code: language_not_supported`) |

A language the model does not cover is refused rather than decoded: the old English-only
fallback answered `zh`/`ja`/`ar`/`vi` with English-sounding text and HTTP 200
([#58](https://github.com/anatolykoptev/ox-whisper/issues/58)). The same check runs on the WebSocket
(refused before the upgrade).

**Memory and files.** The fp32 export is an `encoder.onnx` graph plus `encoder.weights`;
onnxruntime resolves the weights from the process working directory, so run the container with
`working_dir` set to the Parakeet directory (the loader refuses otherwise, and the service exits
at startup: with one model there is nothing to fall back to). Size `mem_limit` for about 3.2 GB
idle per `PARAKEET_POOL_SIZE` slot plus decode buffers: one slot measured 3.5 GB peak on FLEURS
clips and 4.3 GB on 5-minute files. Keep `PARAKEET_IDLE_EVICT_SECS=0` (the default): a reload
after eviction more than doubled RSS (3.2 → 5.5 GB) and added seconds to the next request. Use
the fp32 export; the int8 one measured about 4 WER points worse and is loaded only when no full
precision file is present. With one slot, a second concurrent request waits
(`POOL_ACQUIRE_TIMEOUT_S`) instead of failing.

`/health` answers 503 (`"status": "degraded"`) while Parakeet cannot serve, for example after a
failed reload.

**Need 99 languages?** Use `whisper-large-v3` instead — ox-whisper trades coverage for speed and CPU footprint.

---

## Upgrading to 0.9.0

0.9.0 serves everything with Parakeet and removes the rest. What an upgrader has to do:

1. **Re-run `scripts/download-models.sh`** (or `install.sh`). It now fetches only Parakeet and the VAD; the `en`, `ru` and `punct-en` model directories are no longer read and can be deleted after you have confirmed the new version. Parakeet fp32 needs about 3.2 GB of RAM per slot (`PARAKEET_POOL_SIZE`, default 1; was `POOL_SIZE`, default 2).
2. **The container refuses to start without the Parakeet model**, and with `working_dir` unset for an fp32 export. There is no fallback model, so check `docker logs` after the first start. Rollback is the previous image tag, not `PARAKEET_LANGS=off`.
3. **Remove the settings that no longer exist** (startup warns about each one still set): `PARAKEET_LANGS`, `ZIPFORMER_RU_DIR`, `MOONSHINE_MODELS_DIR`, `PUNCT_MODEL`, `PUNCT_VOCAB`, `DIARIZE_*`, `POOL_SIZE`, `OX_WHISPER_IDLE_EVICT_SECS`, `TTS_ENABLED`, and the matching volume mounts. `MOONSHINE_PORT` and `MOONSHINE_THREADS` keep their names.
4. **Routes and fields removed:** `/transcribe`, `/transcribe/upload` and `/transcribe/stream` (404); the `diarize` option (`diarize=true` is a 400); the `punctuate` option and the WebSocket `punctuate` / `smart_format` parameters; `verbose_json` fields `language_confidence` and `utterances`; the `chunks` field and `max_chunk_len`; `punctuation` and `tts` in `/health`.
5. **Behaviour changes:** a language Parakeet does not cover, an unknown `response_format`, and a WebSocket `sample_rate` other than 16000 are now HTTP 400. With no `language`, `verbose_json` has no `language` key and `smart_format` is skipped (its rules are per language). `/health` answers 503 while Parakeet cannot serve.
6. **Metrics:** removed series `oxwhisper_recognizer_pool_size{lang="ru"|"en"}` (the pool is `lang="parakeet"`), `oxwhisper_vad_no_speech_total{caller="sse"}`, `oxwhisper_route_fallback_total` and the `oxwhisper_tts_*` family. New: `oxwhisper_ws_buffer_limit_total`, `caller="ws_poll"`, and `lang="auto"` on `oxwhisper_transcribe_duration_seconds` (and the VAD/chunk series) for requests with no language. Update alert rules and dashboards that select the old labels.

---

## Benchmarks

**Accuracy, 2026-10.** 100 Russian and 100 English FLEURS test utterances (public set, 4–23 s
each), sent through `/v1/audio/transcriptions` with VAD on as in production. The older rows
send each clip with its own `language`; the Parakeet row sends both sets with `language=ru`, as
a client that always says `ru` would. WER normalised (lowercase, punctuation stripped, `ё`→`е`); "case kept" keeps
capitalisation.

| | RU WER (case kept) | EN WER (case kept) | Empty results |
|---|---|---|---|
| *Removed:* Zipformer-RU + Moonshine v2 (before the VAD fixes) | 17.4% (25.4%) | 29.9% (32.8%) | 14 of 200 EN |
| *Removed:* Zipformer-RU + Moonshine v2, VAD fixed ([#54](https://github.com/anatolykoptev/ox-whisper/pull/54)) | 13.9% (22.3%) | 15.5% (19.7%) | 0 |
| **Parakeet TDT 0.6B v3 fp32** | **7.7% (9.5%)** | **9.7% (11.9%)** | **0** |

With VAD off, Parakeet scores 5.5% RU and 5.7% EN on the same clips; the VAD path costs the
rest ([#53](https://github.com/anatolykoptev/ox-whisper/issues/53)). On a 5-minute English file the old Moonshine model returned almost nothing (95% WER); Parakeet
gets 38%. Details: [docs/benchmarks.md](docs/benchmarks.md).

**Speed.** Oracle Cloud A1 free tier, 4 ARM threads, CPU-only. Parakeet fp32 measured RTF
0.14–0.27 depending on load on this shared box (a 15 s note took 3.7 s end to end).
Moonshine and Zipformer, which earlier releases also shipped, ran at RTF 0.02–0.06 but were
removed (see [docs/benchmarks.md](docs/benchmarks.md) for their numbers).

For comparison on the same box: `faster-whisper tiny int8` ~0.075 RTF, `whisper.cpp tiny-q8_0` ~0.13 RTF.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MOONSHINE_PORT` | `8092` | HTTP/WS port (the name predates Parakeet) |
| `MOONSHINE_THREADS` | `4` | ONNX inference threads (ditto) |
| `MAX_AUDIO_DURATION_S` | `0` | Max input length, `0`=unlimited |
| `WS_MAX_BUFFER_S` | `120` | Longest audio a WebSocket session may buffer; past it the client gets an `Error` frame and the connection closes (also capped by `MAX_AUDIO_DURATION_S` when set) |
| `VAD_MIN_DURATION_S` | `10` | Auto-enable VAD above this length |
| `OXWHISPER_PROM_PORT` | `9092` | Prometheus metrics port |

<details>
<summary>Model paths and tuning knobs</summary>

| Variable | Default |
|----------|---------|
| `POOL_ACQUIRE_TIMEOUT_S` | `30` — how long a request waits for a busy recognizer |
| `PARAKEET_DIR` | `/parakeet-models` |
| `PARAKEET_POOL_SIZE` | `1` — each slot holds ~3.2 GB |
| `PARAKEET_IDLE_EVICT_SECS` | `0` — keep Parakeet resident (recommended); it does not inherit `OX_WHISPER_IDLE_EVICT_SECS` |
| `SILERO_VAD_MODEL` | `/vad/silero_vad.onnx` |
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
| Parakeet TDT 0.6B v3 (fp32) | 25 European languages | 2.5 GB | [HF](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3) — CC-BY-4.0 (NVIDIA) |
| Silero VAD | — | 0.6 MB | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |

Downloaded by `scripts/download-models.sh` (run automatically by `install.sh`); Parakeet's files are sha256-pinned.

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
| `oxwhisper_vad_no_speech_total` | counter | `caller` (`batch`; `ws_poll` is the per-frame WebSocket check, where silence is normal, so do not alert on it) |
| `oxwhisper_vad_mutex_poisoned_total` | counter | — |
| `oxwhisper_pool_acquire_wait_seconds` | histogram | — |
| `oxwhisper_pool_acquire_timeouts_total` | counter | — |
| `oxwhisper_hallucination_rejected_total` | counter | `lang` |
| `oxwhisper_recognizer_pool_size` · `_busy` | gauge | `lang` |
| `oxwhisper_ws_active_connections` | gauge | — |
| `oxwhisper_ws_buffer_limit_total` | counter | — (sessions closed at `WS_MAX_BUFFER_S`) |

---

## Limitations

- **aarch64 only.** Pre-built `.so` libs are ARM64; x86_64 needs source rebuild of `vendor/sherpa-rs-sys`.
- **Whole-file responses.** `/v1/audio/transcriptions` answers once the file is decoded. Use `/v1/listen` (WebSocket, 16 kHz mono PCM only) for incremental output.
- **No auth.** Bind to `0.0.0.0`; put behind nginx / Caddy if exposed to the network.
- **25 languages** (Parakeet); anything else is a 400. For broader coverage, use `whisper-large-v3` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or [speaches](https://github.com/speaches-ai/speaches).
- **Memory.** Parakeet holds ~3.2 GB and peaks at ~4.3 GB on 5-minute audio; it needs a box with room for that.
- **Removed in this release.** Moonshine, Zipformer/GigaAM and the English punctuation model (Parakeet replaces all three), language auto-detection (`language_confidence` is gone from `verbose_json`), speaker diarization (`diarize=true` is a 400) and the `punctuate` option (Parakeet always punctuates), the native endpoints (`/transcribe`, `/transcribe/upload`, `/transcribe/stream`), and the optional text-to-speech child. An unknown `response_format` is now a 400 (it used to fall back to `json`), and `/v1/listen` refuses a `sample_rate` other than 16000 with a 400. `smart_format`, `redact`, `paragraphs`, `custom_spelling` and `keywords` are plain text transforms and stay.

---

## Build

```bash
git clone https://github.com/anatolykoptev/ox-whisper && cd ox-whisper
cargo build --release          # native aarch64
docker build -t ox-whisper .   # BuildKit + cargo-chef layer cache
```

Each release publishes `ghcr.io/anatolykoptev/ox-whisper:X.Y.Z`, `:X.Y` and `:latest`.

---

## License

MIT. Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0, NVIDIA), [Silero VAD](https://github.com/snakers4/silero-vad).
