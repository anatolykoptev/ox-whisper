# ox-whisper — Speech-to-Text Server

**Rust** 1.93 edition 2024 | Docker container on port 8092

## Structure

| File | Role |
|------|------|
| `src/main.rs` | Axum server, route wiring |
| `src/config.rs` | Environment config |
| `src/handlers.rs` | HTTP + WebSocket handlers |
| `src/transcribe.rs` | sherpa-onnx transcription engine |
| `src/audio.rs` | Audio format conversion (ffmpeg) |
| `src/chunking.rs` | Long audio chunking with VAD |
| `src/vad.rs` | Voice activity detection |
| `src/punctuate.rs` | Punctuation restoration |
| `src/models.rs` | Model management |
| `vendor/sherpa-rs/` | Vendored sherpa-onnx Rust bindings |

## API

- `POST /v1/audio/transcriptions` — OpenAI-compatible (multipart: file, model, language, response_format)
- `GET /v1/models` — list available models
- `WS /v1/listen` — real-time streaming transcription

## Models

7 languages (Moonshine v2) + Russian. Models loaded at startup from mounted volume.

## Deploy

```bash
cd ~/deploy/krolik-server
docker compose build --no-cache ox-whisper && docker compose up -d --no-deps --force-recreate ox-whisper
```

## Gotchas

- **aarch64 only** — sherpa-onnx `.so` libs are pre-compiled for ARM64
- CI uses Docker buildx with QEMU (no native clippy — native deps)
- ffmpeg required in container for audio format conversion
