# ox-whisper — Speech-to-Text Server

**Rust** 1.93 edition 2024 | Docker container on port 8092

## Structure

| File | Role |
|------|------|
| `src/main.rs` | Axum server, route wiring |
| `src/config.rs` | Environment config |
| `src/handlers.rs` · `src/handler_openai.rs` | `/health`; the OpenAI-compatible transcription endpoint |
| `src/ws_handler.rs` · `src/ws_session.rs` | WebSocket streaming |
| `src/language.rs` | The language hint: validated against Parakeet's 25 languages, else 400 |
| `src/transcribe.rs` | Transcription engine: VAD or chunk split, bounded batch decode |
| `src/models.rs` | Loads Parakeet + VAD; refuses to start without Parakeet |
| `src/pool.rs` | Recognizer pool: bounded-wait `acquire`, opt-in idle eviction, `is_healthy` |
| `src/vad.rs` | Shared Silero VAD (reset per request, poison recovery) |
| `src/tmpfile.rs` | `TempFile`: owns an upload / ffmpeg temp file, removes it on drop |
| `src/chunking.rs` | Text helpers (`sanitize_utf8`); the audio split `split_audio_chunks` is in `transcribe.rs` |
| `src/audio.rs` | Audio format conversion (ffmpeg) |
| `src/smart_format/` · `pii.rs` · `paragraphs.rs` · `spelling.rs` | Post-processing options (pure text transforms) |
| `vendor/sherpa-rs/` | Vendored sherpa-onnx Rust bindings |

## API

- `POST /v1/audio/transcriptions` — OpenAI-compatible (multipart: file, model, language, response_format)
- `GET /v1/models` — list available models
- `WS /v1/listen` — real-time streaming transcription

## Models

One model: Parakeet TDT 0.6B v3 (fp32) serves 25 European languages with its own case and
punctuation, and transcribes whatever language it hears. `language` is a hint that is checked
(unsupported -> 400, never decoded) and echoed, not a route. The service exits at startup if
Parakeet does not load. Silero VAD is the only other model. Benchmarks: README and
`docs/benchmarks.md`.

## Deploy

Releases publish `ghcr.io/anatolykoptev/ox-whisper:X.Y.Z`, `:X.Y` and `:latest`; the bundled
compose runs `:latest` unless `OX_WHISPER_VERSION` pins one. To run a local build instead, tag it and point the compose
`image:` at the tag (or use a compose with a `build:` section), then
`docker compose up -d --no-deps ox-whisper`.

The container needs `working_dir` = the Parakeet directory (the fp32 encoder resolves
`encoder.weights` from the working directory), about 7 GB of `mem_limit`,
`PARAKEET_POOL_SIZE=1` and `PARAKEET_IDLE_EVICT_SECS=0` (see README, Languages). Rollback is the
previous image tag.

## Gotchas

- **aarch64 only** — sherpa-onnx `.so` libs are pre-compiled for ARM64
- CI runs natively on `ubuntu-24.04-arm`: `cargo nextest` plus a full `docker buildx` image build
- ffmpeg required in container for audio format conversion
- The Silero VAD is one process-wide instance: anything that feeds it must go through
  `lock_vad()` + `apply_vad()`, which resets it first — state carried between requests once
  produced empty transcripts
- Anything that writes an upload or a conversion to disk owns it through `TempFile`; never
  `std::fs::write` a path and remove it by hand (a cancelled handler skips the removal)
- Waiting `pool.acquire()` blocks for up to `POOL_ACQUIRE_TIMEOUT_S`: call it only from
  `spawn_blocking`, never on an async task (use `try_acquire()` there)
