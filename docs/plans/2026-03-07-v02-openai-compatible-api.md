# v0.2.0 OpenAI-Compatible API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenAI Whisper-compatible `/v1/audio/transcriptions` endpoint so any OpenAI SDK client works as drop-in replacement, plus SRT/VTT subtitle output and language auto-detection.

**Architecture:** New `src/openai.rs` handler module implements the OpenAI multipart API contract. New `src/formats.rs` generates SRT/VTT/verbose_json from existing `WordTimestamp` data. Language detection uses audio sampling + multi-model scoring (no new dependencies). Existing `/transcribe` endpoints unchanged — v1 routes are additive.

**Tech Stack:** Rust, axum 0.8 (multipart), serde, existing sherpa-onnx pipeline.

---

### Task 1: SRT/VTT Subtitle Generator

Create `src/formats.rs` — converts word timestamps into SRT and VTT subtitle formats.

**Files:**
- Create: `src/formats.rs`
- Modify: `src/main.rs:8` (add `mod formats;`)

**Step 1: Write tests for SRT generation**

In `src/formats.rs`:

```rust
use crate::words::WordTimestamp;

/// Format seconds as SRT timestamp: HH:MM:SS,mmm
fn format_srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

/// Format seconds as VTT timestamp: HH:MM:SS.mmm
fn format_vtt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

/// Group words into subtitle segments of ~max_words each.
/// Returns (start_time, end_time, text) tuples.
fn group_words(words: &[WordTimestamp], max_words: usize) -> Vec<(f32, f32, String)> {
    if words.is_empty() {
        return Vec::new();
    }
    let max_words = if max_words == 0 { 8 } else { max_words };
    let mut segments = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + max_words).min(words.len());
        let text: String = words[i..end].iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
        segments.push((words[i].start, words[end - 1].end, text));
        i = end;
    }
    segments
}

/// Generate SRT subtitle string from word timestamps.
pub fn to_srt(words: &[WordTimestamp]) -> String {
    let segments = group_words(words, 8);
    let mut out = String::new();
    for (i, (start, end, text)) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{} --> {}\n", format_srt_time(*start), format_srt_time(*end)));
        out.push_str(text);
        out.push_str("\n\n");
    }
    out
}

/// Generate WebVTT subtitle string from word timestamps.
pub fn to_vtt(words: &[WordTimestamp]) -> String {
    let segments = group_words(words, 8);
    let mut out = String::from("WEBVTT\n\n");
    for (start, end, text) in &segments {
        out.push_str(&format!("{} --> {}\n", format_vtt_time(*start), format_vtt_time(*end)));
        out.push_str(text);
        out.push_str("\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_words() -> Vec<WordTimestamp> {
        vec![
            WordTimestamp { word: "Hello".into(), start: 0.0, end: 0.3, confidence: None },
            WordTimestamp { word: "world".into(), start: 0.4, end: 0.8, confidence: None },
            WordTimestamp { word: "this".into(), start: 1.0, end: 1.2, confidence: None },
            WordTimestamp { word: "is".into(), start: 1.3, end: 1.4, confidence: None },
            WordTimestamp { word: "a".into(), start: 1.5, end: 1.55, confidence: None },
            WordTimestamp { word: "test".into(), start: 1.6, end: 2.0, confidence: None },
        ]
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_format_vtt_time() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
        assert_eq!(format_vtt_time(1.5), "00:00:01.500");
    }

    #[test]
    fn test_to_srt() {
        let words = sample_words();
        let srt = to_srt(&words);
        assert!(srt.starts_with("1\n00:00:00,000 --> "));
        assert!(srt.contains("Hello world this is a test"));
    }

    #[test]
    fn test_to_vtt() {
        let words = sample_words();
        let vtt = to_vtt(&words);
        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("00:00:00.000 --> "));
    }

    #[test]
    fn test_to_srt_empty() {
        assert_eq!(to_srt(&[]), "");
    }

    #[test]
    fn test_to_vtt_empty() {
        assert_eq!(to_vtt(&[]), "WEBVTT\n\n");
    }

    #[test]
    fn test_group_words_splits() {
        // 10 words with max_words=4 => 3 segments (4+4+2)
        let words: Vec<WordTimestamp> = (0..10).map(|i| WordTimestamp {
            word: format!("w{}", i), start: i as f32 * 0.5, end: i as f32 * 0.5 + 0.4, confidence: None,
        }).collect();
        let segments = group_words(&words, 4);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].2, "w0 w1 w2 w3");
        assert_eq!(segments[2].2, "w8 w9");
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib formats`
Expected: All 7 tests pass.

**Step 3: Add `mod formats;` to main.rs**

Add `mod formats;` after `mod config;` in `src/main.rs:10`.

**Step 4: Commit**

```bash
git add src/formats.rs src/main.rs
git commit -m "feat: add SRT/VTT subtitle generation from word timestamps"
```

---

### Task 2: OpenAI-Compatible Response Types

Create `src/openai.rs` with all OpenAI API response structs and the `response_format` enum.

**Files:**
- Create: `src/openai.rs`
- Modify: `src/main.rs` (add `mod openai;`)

**Step 1: Define response types**

In `src/openai.rs`:

```rust
use serde::{Deserialize, Serialize};

/// OpenAI-compatible response_format values.
/// We support the whisper-1 set: json, verbose_json, text, srt, vtt.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Json,
    VerboseJson,
    Text,
    Srt,
    Vtt,
}

impl Default for ResponseFormat {
    fn default() -> Self { Self::Json }
}

/// OpenAI /v1/audio/transcriptions "json" response.
#[derive(Serialize)]
pub struct JsonResponse {
    pub text: String,
}

/// A single segment in verbose_json output.
#[derive(Serialize, Clone)]
pub struct Segment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// A single word in verbose_json output.
#[derive(Serialize, Clone)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// OpenAI /v1/audio/transcriptions "verbose_json" response.
#[derive(Serialize)]
pub struct VerboseJsonResponse {
    pub text: String,
    pub language: String,
    pub duration: f64,
    pub segments: Vec<Segment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
}

/// Build segments from word timestamps by grouping into ~8-word chunks.
pub fn words_to_segments(words: &[crate::words::WordTimestamp]) -> Vec<Segment> {
    if words.is_empty() {
        return Vec::new();
    }
    let max_words = 8;
    let mut segments = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + max_words).min(words.len());
        let text: String = words[i..end].iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
        segments.push(Segment {
            id: segments.len(),
            start: words[i].start as f64,
            end: words[end - 1].end as f64,
            text,
        });
        i = end;
    }
    segments
}

/// Convert internal word timestamps to OpenAI Word format.
pub fn words_to_openai(words: &[crate::words::WordTimestamp]) -> Vec<Word> {
    words.iter().map(|w| Word {
        word: w.word.clone(),
        start: w.start as f64,
        end: w.end as f64,
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::WordTimestamp;

    #[test]
    fn test_response_format_default() {
        assert_eq!(ResponseFormat::default(), ResponseFormat::Json);
    }

    #[test]
    fn test_response_format_deserialize() {
        let f: ResponseFormat = serde_json::from_str("\"verbose_json\"").unwrap();
        assert_eq!(f, ResponseFormat::VerboseJson);
    }

    #[test]
    fn test_words_to_segments() {
        let words = vec![
            WordTimestamp { word: "hello".into(), start: 0.0, end: 0.5, confidence: None },
            WordTimestamp { word: "world".into(), start: 0.6, end: 1.0, confidence: None },
        ];
        let segments = words_to_segments(&words);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello world");
        assert_eq!(segments[0].id, 0);
    }

    #[test]
    fn test_json_response_serialize() {
        let r = JsonResponse { text: "hello".into() };
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(json, r#"{"text":"hello"}"#);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib openai`
Expected: All 4 tests pass.

**Step 3: Add `mod openai;` to main.rs**

**Step 4: Commit**

```bash
git add src/openai.rs src/main.rs
git commit -m "feat: add OpenAI-compatible response types and format enum"
```

---

### Task 3: Language Detection

Add language detection by running a short audio sample through each available model and comparing confidence.

**Files:**
- Create: `src/detect.rs`
- Modify: `src/main.rs` (add `mod detect;`)

**Step 1: Implement detection**

In `src/detect.rs`:

```rust
use crate::models::Models;

/// Supported language codes for auto-detection.
const DETECT_LANGUAGES: &[&str] = &["en", "ru"];

/// Detect language by transcribing a short sample with each model
/// and picking the one with higher confidence / longer output.
///
/// Only tests EN vs RU (the two models we load). For other languages
/// routed through the EN (Moonshine) model, we can't distinguish —
/// returns "en" as default.
pub fn detect_language(models: &Models, samples: &[f32]) -> String {
    // Take first ~3 seconds for detection (48000 samples at 16kHz)
    let clip = if samples.len() > 48000 { &samples[..48000] } else { samples };

    let mut best_lang = "en".to_string();
    let mut best_score: f64 = 0.0;

    // Try EN model
    if let Some(ref pool) = models.en {
        if let Some(mut rec) = pool.acquire() {
            let result = rec.transcribe(16000, clip);
            let text = result.text.trim();
            if !text.is_empty() {
                let score = text.len() as f64;
                if score > best_score {
                    best_score = score;
                    best_lang = "en".to_string();
                }
            }
        }
    }

    // Try RU model
    if let Some(ref pool) = models.ru {
        if let Some(mut rec) = pool.acquire() {
            let result = rec.transcribe(16000, clip);
            let text = result.text.trim();
            if !text.is_empty() {
                // RU text in Cyrillic is typically longer in bytes per word
                // Use char count for fairer comparison
                let score = text.chars().count() as f64;
                if score > best_score {
                    best_lang = "ru".to_string();
                }
            }
        }
    }

    best_lang
}
```

**Step 2: Add `mod detect;` to main.rs**

**Step 3: Commit**

```bash
git add src/detect.rs src/main.rs
git commit -m "feat: add basic language detection (EN vs RU)"
```

---

### Task 4: OpenAI Handler — `/v1/audio/transcriptions`

The main endpoint. Parses OpenAI multipart format, runs transcription, returns in requested format.

**Files:**
- Create: `src/handler_openai.rs`
- Modify: `src/main.rs` (add `mod handler_openai;`, add route)

**Step 1: Implement handler**

In `src/handler_openai.rs`:

```rust
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::audio::load_wav;
use crate::detect::detect_language;
use crate::formats;
use crate::handlers::AppState;
use crate::openai::*;
use crate::transcribe;

/// POST /v1/audio/transcriptions
///
/// OpenAI-compatible multipart endpoint.
/// Fields: file (required), model, language, response_format, temperature,
///         timestamp_granularities[]
pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let parsed = match parse_openai_multipart(&mut multipart).await {
        Ok(p) => p,
        Err(msg) => {
            return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({
                "error": { "message": msg, "type": "invalid_request_error" }
            }))).into_response();
        }
    };

    let file_path = parsed.file_path;
    let language = parsed.language.clone();
    let format = parsed.response_format.clone();
    let want_words = parsed.timestamp_granularities.contains(&"word".to_string());
    let p = file_path.clone();

    let result = tokio::task::spawn_blocking(move || {
        // Auto-detect language if not specified
        let lang = if language == "auto" || language.is_empty() {
            if let Ok((samples, _)) = load_wav(&p) {
                detect_language(&state.models, &samples)
            } else {
                "en".to_string()
            }
        } else {
            language
        };

        let r = transcribe::transcribe(
            &state.models, &state.config, &p, &lang,
            None, None, 0,
        );
        let _ = std::fs::remove_file(&file_path);
        (r, lang)
    }).await;

    let (result, detected_lang) = match result {
        Ok((r, lang)) => (r, lang),
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(serde_json::json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            }))).into_response();
        }
    };

    match result {
        Ok(r) => format_response(r, &detected_lang, format, want_words),
        Err(e) => {
            let status = match &e {
                transcribe::TranscribeError::Audio(_) => StatusCode::BAD_REQUEST,
                transcribe::TranscribeError::TooLong(_, _) => StatusCode::BAD_REQUEST,
                transcribe::TranscribeError::LanguageNotAvailable(_) => StatusCode::BAD_REQUEST,
                transcribe::TranscribeError::NoRecognizer => StatusCode::SERVICE_UNAVAILABLE,
            };
            (status, axum::Json(serde_json::json!({
                "error": { "message": e.to_string(), "type": "invalid_request_error" }
            }))).into_response()
        }
    }
}

fn format_response(
    r: transcribe::TranscribeResult,
    language: &str,
    format: ResponseFormat,
    want_words: bool,
) -> Response {
    let duration = r.speech_ms / 1000.0;

    match format {
        ResponseFormat::Text => {
            (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/plain")], r.text).into_response()
        }
        ResponseFormat::Srt => {
            let srt = formats::to_srt(&r.words);
            (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/plain")], srt).into_response()
        }
        ResponseFormat::Vtt => {
            let vtt = formats::to_vtt(&r.words);
            (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/vtt")], vtt).into_response()
        }
        ResponseFormat::Json => {
            axum::Json(JsonResponse { text: r.text }).into_response()
        }
        ResponseFormat::VerboseJson => {
            let segments = words_to_segments(&r.words);
            let words = if want_words { words_to_openai(&r.words) } else { Vec::new() };
            axum::Json(VerboseJsonResponse {
                text: r.text,
                language: language.to_string(),
                duration,
                segments,
                words,
            }).into_response()
        }
    }
}

struct OpenAiUpload {
    file_path: std::path::PathBuf,
    language: String,
    response_format: ResponseFormat,
    timestamp_granularities: Vec<String>,
}

async fn parse_openai_multipart(multipart: &mut Multipart) -> Result<OpenAiUpload, String> {
    let mut file_path: Option<std::path::PathBuf> = None;
    let mut language = String::new();
    let mut response_format = ResponseFormat::Json;
    let mut timestamp_granularities = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let ext = field.file_name()
                    .and_then(|n| std::path::Path::new(n).extension().map(|e| e.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "wav".to_string());
                let tmp = format!("/tmp/{}.{}", uuid::Uuid::new_v4(), ext);
                let data = field.bytes().await.map_err(|e| e.to_string())?;
                std::fs::write(&tmp, &data).map_err(|e: std::io::Error| e.to_string())?;
                file_path = Some(std::path::PathBuf::from(tmp));
            }
            "language" => language = field.text().await.unwrap_or_default(),
            "response_format" => {
                let val = field.text().await.unwrap_or_default();
                response_format = match val.as_str() {
                    "text" => ResponseFormat::Text,
                    "srt" => ResponseFormat::Srt,
                    "vtt" => ResponseFormat::Vtt,
                    "verbose_json" => ResponseFormat::VerboseJson,
                    _ => ResponseFormat::Json,
                };
            }
            "timestamp_granularities[]" => {
                let val = field.text().await.unwrap_or_default();
                timestamp_granularities.push(val);
            }
            "model" | "temperature" | "prompt" => {
                // Accepted but ignored — we use our own models
                let _ = field.text().await;
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    Ok(OpenAiUpload {
        file_path: file_path.ok_or("missing 'file' field")?,
        language,
        response_format,
        timestamp_granularities,
    })
}
```

**Step 2: Register route in main.rs**

Add to `src/main.rs` after line 48 (the stream route):

```rust
.route("/v1/audio/transcriptions", post(handler_openai::transcriptions))
```

And add `mod handler_openai;` with the other module declarations.

**Step 3: Build and verify**

Run: `cargo build`
Expected: Compiles without errors.

**Step 4: Commit**

```bash
git add src/handler_openai.rs src/main.rs
git commit -m "feat: add OpenAI-compatible /v1/audio/transcriptions endpoint"
```

---

### Task 5: `/v1/models` Endpoint

Lists available models — required for OpenAI SDK compatibility checks.

**Files:**
- Modify: `src/handler_openai.rs` (add handler)
- Modify: `src/main.rs` (add route)

**Step 1: Add models list handler**

Append to `src/handler_openai.rs`:

```rust
/// GET /v1/models
///
/// Returns list of available models in OpenAI format.
pub async fn list_models(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let mut models = Vec::new();
    if state.models.en.is_some() {
        models.push(serde_json::json!({
            "id": "moonshine-v2-base",
            "object": "model",
            "owned_by": "ox-whisper",
        }));
    }
    if state.models.ru.is_some() {
        let name = state.models.ru.as_ref()
            .and_then(|p| p.acquire())
            .map(|r| r.model_name().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        models.push(serde_json::json!({
            "id": name,
            "object": "model",
            "owned_by": "ox-whisper",
        }));
    }
    axum::Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}
```

**Step 2: Add route in main.rs**

```rust
.route("/v1/models", get(handler_openai::list_models))
```

**Step 3: Build**

Run: `cargo build`

**Step 4: Commit**

```bash
git add src/handler_openai.rs src/main.rs
git commit -m "feat: add /v1/models endpoint for OpenAI SDK compatibility"
```

---

### Task 6: Deploy, Test, Verify

Build Docker image, deploy, test all formats with curl.

**Files:**
- No code changes — testing only.

**Step 1: Build and deploy**

```bash
cd ~/deploy/krolik-server
docker compose build --no-cache moonshine
docker compose up -d --no-deps --force-recreate moonshine
```

**Step 2: Test OpenAI JSON format (default)**

```bash
curl -s -F file=@/path/to/test.wav http://localhost:8092/v1/audio/transcriptions | jq .
```

Expected: `{"text": "..."}`

**Step 3: Test verbose_json with word timestamps**

```bash
curl -s -F file=@/path/to/test.wav \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word" \
  http://localhost:8092/v1/audio/transcriptions | jq .
```

Expected: `{"text": "...", "language": "en", "duration": ..., "segments": [...], "words": [...]}`

**Step 4: Test SRT output**

```bash
curl -s -F file=@/path/to/test.wav -F response_format=srt http://localhost:8092/v1/audio/transcriptions
```

Expected:
```
1
00:00:00,000 --> 00:00:01,200
Hello world...
```

**Step 5: Test VTT output**

```bash
curl -s -F file=@/path/to/test.wav -F response_format=vtt http://localhost:8092/v1/audio/transcriptions
```

Expected:
```
WEBVTT

00:00:00.000 --> 00:00:01.200
Hello world...
```

**Step 6: Test text output**

```bash
curl -s -F file=@/path/to/test.wav -F response_format=text http://localhost:8092/v1/audio/transcriptions
```

Expected: plain text, no JSON wrapper.

**Step 7: Test auto language detection (no language field)**

```bash
curl -s -F file=@/path/to/russian_audio.wav \
  -F response_format=verbose_json \
  http://localhost:8092/v1/audio/transcriptions | jq .language
```

Expected: `"ru"`

**Step 8: Test /v1/models**

```bash
curl -s http://localhost:8092/v1/models | jq .
```

Expected: list of loaded models.

**Step 9: Test with Python OpenAI SDK**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8092/v1", api_key="unused")
with open("test.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-1", file=f)
    print(result.text)
```

Expected: transcription text returned without errors.

**Step 10: Commit and tag**

```bash
# Update version in Cargo.toml to 0.2.0
git add -A
git commit -m "release: v0.2.0 — OpenAI-compatible API, SRT/VTT, language detection"
git tag v0.2.0
git push origin master --tags
```

---

### Summary

| Task | What | Files |
|------|------|-------|
| 1 | SRT/VTT subtitle generator | `src/formats.rs` |
| 2 | OpenAI response types | `src/openai.rs` |
| 3 | Language detection | `src/detect.rs` |
| 4 | OpenAI handler + route | `src/handler_openai.rs`, `src/main.rs` |
| 5 | `/v1/models` endpoint | `src/handler_openai.rs`, `src/main.rs` |
| 6 | Deploy + test all formats | Testing only |

Total: 4 new files, 1 modified file. ~400 lines of new code.
