# ox-whisper — Speech-to-Text Server

**Rust** 1.93 edition 2024 | Docker container on port 8092

## Structure

| File | Role |
|------|------|
| `src/main.rs` | Axum server, route wiring |
| `src/config.rs` | Environment config |
| `src/handlers.rs` · `src/handler_openai.rs` · `src/handler_stream.rs` | HTTP handlers (native, OpenAI-compatible, SSE) |
| `src/ws_handler.rs` · `src/ws_session.rs` | WebSocket streaming |
| `src/routing.rs` | Which engine serves a language (Parakeet / RU model / Moonshine) |
| `src/transcribe.rs` · `src/streaming.rs` | Transcription engine: VAD or chunk split, bounded batch decode, punctuation |
| `src/models.rs` · `src/recognizer.rs` | Model loading, fallbacks after a failed reload |
| `src/pool.rs` | Recognizer pools: bounded-wait `acquire`, idle eviction |
| `src/vad.rs` | Shared Silero VAD (reset per request, poison recovery) |
| `src/detect.rs` | Language auto-detection |
| `src/chunking.rs` | Fixed-length split of long audio when VAD is off |
| `src/audio.rs` | Audio format conversion (ffmpeg) |
| `src/punctuate.rs` | Punctuation restoration |
| `src/smart_format/` · `pii.rs` · `paragraphs.rs` · `spelling.rs` · `diarize.rs` | Post-processing options |
| `src/tts/` | Optional TTS child process |
| `vendor/sherpa-rs/` | Vendored sherpa-onnx Rust bindings |

## API

- `POST /v1/audio/transcriptions` — OpenAI-compatible (multipart: file, model, language, response_format)
- `GET /v1/models` — list available models
- `WS /v1/listen` — real-time streaming transcription

## Models

Parakeet TDT 0.6B v3 (fp32) serves 25 European languages, `ru` and `en` included, with its
own case and punctuation; Moonshine v2 serves `ar ja vi zh` (and `en es uk` without Parakeet);
Zipformer/GigaAM serve `ru` only when Parakeet does not. Models load at startup from mounted
volumes. Benchmarks: README and `docs/benchmarks.md`.

## Deploy

Build the image and recreate the container, e.g.:

```bash
docker build -t ox-whisper . && docker compose up -d --no-deps ox-whisper
```

With Parakeet the container needs `working_dir` = the Parakeet directory, about 6 GB of
`mem_limit`, `PARAKEET_POOL_SIZE=1` and `PARAKEET_IDLE_EVICT_SECS=0` (see README, Languages).
`PARAKEET_LANGS=off` rolls back to the old models without a rebuild.

## Gotchas

- **aarch64 only** — sherpa-onnx `.so` libs are pre-compiled for ARM64
- CI runs natively on `ubuntu-24.04-arm`: `cargo nextest` plus a full `docker buildx` image build
- ffmpeg required in container for audio format conversion
- The Silero VAD is one process-wide instance: anything that feeds it must go through
  `lock_vad()` + `apply_vad()`, which resets it first — state carried between requests once
  produced empty transcripts
- Waiting `pool.acquire()` blocks for up to `POOL_ACQUIRE_TIMEOUT_S`: call it only from
  `spawn_blocking`, never on an async task (use `try_acquire()` there)
