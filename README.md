# ox-whisper

Fast multilingual speech-to-text server built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) v1.12.28 with Rust and axum. Runs entirely on CPU — no GPU required.

## Supported Models

| Language | Model | Backend | Notes |
|----------|-------|---------|-------|
| EN, AR, ES, JA, UK, VI, ZH | [Moonshine v2 Base](https://github.com/usefulsensors/moonshine) | Moonshine | 68MB, multilingual |
| RU | [GigaAM v2 CTC](https://github.com/salute-developers/GigaAM) | NeMo CTC | 226MB int8, batch decode |
| RU | [GigaAM v3 RNNT](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16) | NeMo Transducer | Built-in punctuation |
| RU | Zipformer Transducer | Transducer | Fallback |

Model type is auto-detected at startup based on files present in the model directories.

## Features

- **VAD**: Silero VAD splits long audio into speech segments, auto-enabled for audio >= 10s
- **Punctuation**: External punctuation model (CNN-BiLSTM) for EN and RU
- **Word timestamps**: Per-word start/end times with confidence scores (`exp(avg_log_prob)`)
- **Streaming**: SSE endpoint for real-time chunk-by-chunk transcription progress
- **Batch decode**: VAD chunks decoded in a single batch call for RU models
- **Pool**: Configurable recognizer pool for concurrent requests (`POOL_SIZE`)
- **No audio limit**: Handles arbitrarily long files (YouTube, podcasts, etc.)
- **Format conversion**: Accepts any ffmpeg-supported format, auto-converts to WAV

## API

### `GET /health`

Returns server status, loaded models, and available languages.

```json
{
  "status": "ok",
  "engine": "sherpa-onnx",
  "version": "0.1.0",
  "vad": true,
  "punctuation": true,
  "languages": {
    "en": {"model": "moonshine-v2-base", "ready": true},
    "ru": {"model": "zipformer-transducer", "ready": true}
  }
}
```

### `POST /transcribe`

JSON body with path to audio file on the server filesystem.

```json
{
  "audio_path": "/data/recording.wav",
  "language": "en",
  "vad": true,
  "punctuate": true,
  "max_chunk_len": 500
}
```

### `POST /transcribe/upload`

Multipart upload. Fields: `file` (audio), `language`, `vad`, `punctuate`, `max_chunk_len`.

```bash
curl -F file=@recording.mp3 -F language=ru http://localhost:8092/transcribe/upload
```

### `POST /transcribe/stream`

Multipart upload with SSE response. Same fields as `/transcribe/upload`. Returns chunk-by-chunk progress:

```
data: {"type":"chunk","index":0,"total":3,"text":"first segment"}
data: {"type":"chunk","index":1,"total":3,"text":"second segment"}
data: {"type":"done","text":"full transcription","duration_ms":1234.5,"speech_ms":890.0}
```

### Response

```json
{
  "text": "transcribed text with punctuation",
  "chunks": ["chunk 1", "chunk 2"],
  "duration_ms": 1234.5,
  "speech_ms": 890.0,
  "words": [
    {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.95}
  ],
  "confidence": 0.92
}
```

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MOONSHINE_PORT` | `8092` | Server port |
| `MOONSHINE_MODELS_DIR` | `/models` | EN models directory |
| `ZIPFORMER_RU_DIR` | `/ru-models` | RU models directory |
| `SILERO_VAD_MODEL` | `/vad/silero_vad.onnx` | Silero VAD model path |
| `PUNCT_MODEL` | `/punct/model.int8.onnx` | Punctuation model path |
| `PUNCT_VOCAB` | `/punct/bpe.vocab` | Punctuation BPE vocab path |
| `MOONSHINE_THREADS` | `4` | Inference threads per recognizer |
| `POOL_SIZE` | `2` | Recognizer instances per model |
| `VAD_MIN_DURATION_S` | `10.0` | Auto-enable VAD above this duration |
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold |
| `VAD_MIN_SILENCE_S` | `0.5` | Min silence to split segments |
| `VAD_SPEECH_PAD_S` | `0.05` | Padding around speech segments |
| `MAX_AUDIO_DURATION_S` | `0` | Max audio duration (0 = no limit) |
| `ONNX_PROVIDER` | `cpu` | ONNX execution provider |
| `RUST_LOG` | `info` | Log level |

## Docker

```yaml
moonshine:
  build: /path/to/ox-whisper
  ports:
    - "8092:8092"
  volumes:
    - /path/to/en-models:/models:ro
    - /path/to/ru-models:/ru-models:ro
    - /path/to/vad:/vad:ro
    - /path/to/punct:/punct:ro
  environment:
    - MOONSHINE_THREADS=4
    - POOL_SIZE=2
```

Build applies ORT optimizations automatically via Dockerfile sed patches:
- `ORT_ENABLE_ALL` graph optimization (~4x speedup for Moonshine)
- `inter_op_threads=1` (reduces context-switching overhead)

## Performance

Benchmarked on ARM64 (Oracle Cloud A1), 4 threads, CPU only:

| Model | 5s audio | 60s audio | 300s audio |
|-------|----------|-----------|------------|
| EN Moonshine v2 | ~0.7s | ~6s | ~30s |
| RU Zipformer | ~3s | ~65s | ~5min |

## Project Structure

```
src/
  main.rs          — axum server setup
  config.rs        — env var parsing
  models.rs        — model loading, auto-detection, warmup
  recognizer.rs    — RuRecognizer enum (NeMo/Zipformer/CTC)
  transcribe.rs    — core transcription pipeline
  handlers.rs      — HTTP handlers (JSON, multipart)
  handler_stream.rs — SSE streaming handler
  streaming.rs     — streaming transcription logic
  audio.rs         — WAV loading, ffmpeg conversion
  vad.rs           — Silero VAD integration
  punctuate.rs     — external punctuation model
  words.rs         — word timestamps + confidence
  chunking.rs      — text splitting, UTF-8 sanitization
  pool.rs          — generic recognizer pool
vendor/
  sherpa-rs/       — Rust bindings for sherpa-onnx
  sherpa-rs-sys/   — FFI sys crate with sherpa-onnx C++ source
  sherpa-onnx/     — pre-built shared libraries for Docker runtime
```

## License

Private repository.
