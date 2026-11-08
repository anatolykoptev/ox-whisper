# ox-whisper — Speech-to-Text Server

**Rust** 1.88 edition 2024 (the version CI and the Dockerfile builder use) | Docker container on port 8092

Every document is indexed in [docs/INDEX.md](docs/INDEX.md).

## Structure

| File | Role |
|------|------|
| `src/main.rs` | Axum server, route wiring |
| `src/config.rs` | Environment config |
| `src/server.rs` | Serve loop: graceful shutdown with a bounded drain; `ShutdownSignal` tells WebSocket sessions to close (1001) |
| `src/handlers.rs` · `src/handler_openai.rs` | `/health`; the OpenAI-compatible transcription endpoint |
| `src/ws_handler.rs` · `src/ws_session.rs` | WebSocket streaming |
| `src/language.rs` | The language hint: validated against Parakeet's 25 languages, else 400 |
| `src/transcribe.rs` | Transcription engine: `plan_chunks` (contiguous audio, no VAD), bounded batch decode |
| `src/models.rs` | Loads Parakeet + VAD; refuses to start without Parakeet |
| `src/pool.rs` | Recognizer pool: bounded-wait `acquire`, opt-in idle eviction, `is_healthy` |
| `src/vad.rs` | Shared Silero VAD, used only by WebSocket speech detection (reset per request, poison recovery) |
| `src/tmpfile.rs` | `TempFile`: owns an upload / ffmpeg temp file, removes it on drop |
| `src/chunking.rs` | `chunk_bounds` / `split_at_quiet` (cut at quiet points, never drop or pad); text helper `sanitize_utf8` |
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

## Gate

CI (`.github/workflows/ci.yml`) runs three jobs, all of which must pass: `fmt` (`cargo fmt --check`),
`build` (`docker buildx build --platform linux/arm64 .`, the full image on a native arm64 runner),
and `test` (`cargo nextest run --workspace --locked` with `SHERPA_LIB_PATH` and `LD_LIBRARY_PATH`
pointing at `vendor/sherpa-onnx`). Run the same locally with `cargo fmt --check`, `docker build .` and
`cargo nextest run --workspace --locked`. Clippy is not in CI; `make lint` is optional but keep new
code warning-free. A pull request that changes only documents (Markdown, `docs/`, licence, issue and
PR templates) skips the jobs through the `docs-scope` block in `ci.yml`; that block is canonical
across the fleet and compared byte for byte, so do not edit it here.

## Invariants that are easy to break

- **Versions come from release-please** (Conventional Commits). Never edit `Cargo.toml`'s version,
  `CHANGELOG.md` or `.release-please-manifest.json` by hand, and never tag manually.
- **`vendor/` is vendored code.** `vendor/sherpa-rs*` and the prebuilt `.so` files under
  `vendor/sherpa-onnx/` are built for aarch64; do not hand-edit them or add an x86_64 build here.
- **The service has no authentication.** It is meant to sit behind a reverse proxy; do not add a
  feature that assumes callers are trusted with anything more than transcription.
- **The `language` field is a checked hint, not a route.** An unsupported value is a 400 before any
  decode; do not reintroduce a silent fallback model.
- **Post-processing options are pure text transforms** (`smart_format`, `redact`, `paragraphs`,
  `custom_spelling`, `keywords`); keep them free of I/O and model calls.
- **This repository is public.** No hostnames, operator paths, account names, tokens or personal
  data in any tracked file, fixtures and docs included.

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
