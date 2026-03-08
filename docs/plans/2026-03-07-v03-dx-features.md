# v0.3.0 DX Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `extra` passthrough, `custom_spelling`, and `language_confidence` to both native and OpenAI-compatible APIs.

**Architecture:** Three independent features touching shared handler/response types. Each feature is a pure post-processing addition — no changes to core transcription logic.

**Tech Stack:** Rust, axum, serde_json

---

### Task 1: `language_confidence` in detection and verbose_json

Return detection confidence score alongside detected language. Deepgram returns `language_confidence` as a float — we do the same.

**Files:**
- Modify: `src/detect.rs`
- Modify: `src/openai.rs`
- Modify: `src/handler_openai.rs`

**Step 1: Update detect.rs to return confidence**

Change `detect_language` to return `(String, f64)` — language code and confidence.
Confidence = `winner_chars / (winner_chars + loser_chars)`. If both are 0, return `("en", 0.0)`.

```rust
// detect.rs
pub struct DetectResult {
    pub language: String,
    pub confidence: f64,
}

pub(crate) fn detect_language(models: &Models, samples: &[f32]) -> DetectResult {
    let clip = if samples.len() > DETECT_SAMPLES {
        &samples[..DETECT_SAMPLES]
    } else {
        samples
    };

    let en_len = try_transcribe_en(models, clip);
    let ru_len = try_transcribe_ru(models, clip);

    tracing::debug!("language detection: en={} chars, ru={} chars", en_len, ru_len);

    let total = (en_len + ru_len) as f64;
    if ru_len > en_len {
        DetectResult {
            language: "ru".to_string(),
            confidence: if total > 0.0 { ru_len as f64 / total } else { 0.0 },
        }
    } else {
        DetectResult {
            language: "en".to_string(),
            confidence: if total > 0.0 { en_len as f64 / total } else { 0.0 },
        }
    }
}
```

**Step 2: Add `language_confidence` to VerboseJsonResponse**

```rust
// openai.rs — add field to VerboseJsonResponse
pub struct VerboseJsonResponse {
    pub text: String,
    pub language: String,
    pub duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_confidence: Option<f64>,
    pub segments: Vec<Segment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
}
```

**Step 3: Update handler_openai.rs**

In `detect_language_from_file` — return `DetectResult` instead of `String`.
In `transcriptions` — pass confidence to `format_response`.
In `format_response` — set `language_confidence` in verbose_json.

**Step 4: Add tests**

```rust
// detect.rs tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_calculation() {
        // When both are 0, confidence should be 0.0
        let total = 0.0_f64;
        assert_eq!(if total > 0.0 { 10.0 / total } else { 0.0 }, 0.0);

        // 80 en, 20 ru → en with 0.8 confidence
        let en = 80_usize;
        let ru = 20_usize;
        let total = (en + ru) as f64;
        let conf = en as f64 / total;
        assert!((conf - 0.8).abs() < f64::EPSILON);
    }
}
```

**Step 5: Run tests**

Run: `cargo test`
Expected: all tests pass

**Step 6: Commit**

```bash
git add src/detect.rs src/openai.rs src/handler_openai.rs
git commit -m "feat: add language_confidence to detection and verbose_json"
```

---

### Task 2: `custom_spelling` post-processing

Replace words/phrases in transcription output. Useful for domain-specific terminology: `{"from": ["докер", "докера"], "to": "Docker"}`.

**Files:**
- Create: `src/spelling.rs` (< 80 lines)
- Modify: `src/handler_openai.rs`
- Modify: `src/main.rs` (add `mod spelling`)

**Step 1: Write tests for spelling module**

```rust
// spelling.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_no_rules() {
        assert_eq!(apply_spelling("hello world", &[]), "hello world");
    }

    #[test]
    fn test_apply_single_replacement() {
        let rules = vec![SpellingRule {
            from: vec!["докер".to_string()],
            to: "Docker".to_string(),
        }];
        assert_eq!(apply_spelling("запустим докер", &rules), "запустим Docker");
    }

    #[test]
    fn test_apply_multiple_from_variants() {
        let rules = vec![SpellingRule {
            from: vec!["докер".to_string(), "докера".to_string()],
            to: "Docker".to_string(),
        }];
        assert_eq!(apply_spelling("версия докера", &rules), "версия Docker");
    }

    #[test]
    fn test_apply_case_insensitive() {
        let rules = vec![SpellingRule {
            from: vec!["kubernetes".to_string()],
            to: "Kubernetes".to_string(),
        }];
        assert_eq!(apply_spelling("deploy to KUBERNETES", &rules), "deploy to Kubernetes");
    }

    #[test]
    fn test_apply_multiple_rules() {
        let rules = vec![
            SpellingRule { from: vec!["гит".to_string()], to: "Git".to_string() },
            SpellingRule { from: vec!["хаб".to_string()], to: "Hub".to_string() },
        ];
        assert_eq!(apply_spelling("гит хаб", &rules), "Git Hub");
    }

    #[test]
    fn test_apply_word_boundary() {
        let rules = vec![SpellingRule {
            from: vec!["к".to_string()],
            to: "K".to_string(),
        }];
        // Should NOT replace "к" inside words like "как"
        assert_eq!(apply_spelling("к нам", &rules), "K нам");
        // "как" should stay unchanged
        assert_eq!(apply_spelling("как дела", &rules), "как дела");
    }

    #[test]
    fn test_apply_to_words() {
        let rules = vec![SpellingRule {
            from: vec!["докер".to_string()],
            to: "Docker".to_string(),
        }];
        let mut words = vec![
            WordTimestamp { word: "запустим".to_string(), start: 0.0, end: 0.5, confidence: None },
            WordTimestamp { word: "докер".to_string(), start: 0.6, end: 1.0, confidence: None },
        ];
        apply_spelling_to_words(&mut words, &rules);
        assert_eq!(words[0].word, "запустим");
        assert_eq!(words[1].word, "Docker");
    }
}
```

**Step 2: Implement spelling module**

```rust
// spelling.rs
use serde::Deserialize;
use crate::words::WordTimestamp;

#[derive(Debug, Clone, Deserialize)]
pub struct SpellingRule {
    pub from: Vec<String>,
    pub to: String,
}

/// Apply spelling rules to full text (case-insensitive, word-boundary-aware).
pub fn apply_spelling(text: &str, rules: &[SpellingRule]) -> String {
    if rules.is_empty() {
        return text.to_string();
    }
    let mut result = text.to_string();
    for rule in rules {
        for from in &rule.from {
            // Word-boundary replacement using regex-free approach
            result = replace_word(&result, from, &rule.to);
        }
    }
    result
}

/// Apply spelling rules to individual word timestamps.
pub fn apply_spelling_to_words(words: &mut [WordTimestamp], rules: &[SpellingRule]) {
    for word in words.iter_mut() {
        for rule in rules {
            for from in &rule.from {
                if word.word.eq_ignore_ascii_case(from) ||
                   word.word.to_lowercase() == from.to_lowercase() {
                    word.word = rule.to.clone();
                    break;
                }
            }
        }
    }
}

fn replace_word(text: &str, from: &str, to: &str) -> String {
    // Split by whitespace, replace matching words, rejoin
    text.split_whitespace()
        .map(|w| {
            if w.to_lowercase() == from.to_lowercase() {
                to.to_string()
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
```

**Step 3: Parse `custom_spelling` in handler_openai.rs**

Add `custom_spelling` field to `OpenAIUpload`. Parse from multipart as JSON string:
`custom_spelling=[{"from":["докер"],"to":"Docker"}]`

Apply after transcription in `transcriptions()`:
```rust
if !upload.custom_spelling.is_empty() {
    result.text = spelling::apply_spelling(&result.text, &upload.custom_spelling);
    spelling::apply_spelling_to_words(&mut result.words, &upload.custom_spelling);
}
```

**Step 4: Add `mod spelling` to main.rs**

**Step 5: Run tests**

Run: `cargo test`
Expected: all tests pass including 7 new spelling tests

**Step 6: Commit**

```bash
git add src/spelling.rs src/handler_openai.rs src/main.rs
git commit -m "feat: add custom_spelling post-processing for domain terminology"
```

---

### Task 3: `extra` passthrough

Accept arbitrary KV pairs in request, return them unchanged in response. Like Deepgram `extra=job_id:123`.

**Files:**
- Modify: `src/handler_openai.rs`
- Modify: `src/openai.rs`

**Step 1: Write tests**

```rust
// openai.rs — add test
#[test]
fn verbose_response_includes_extra() {
    let resp = VerboseJsonResponse {
        text: "hi".to_string(),
        language: "en".to_string(),
        duration: 1.0,
        language_confidence: None,
        segments: vec![],
        words: vec![],
        extra: Some(serde_json::json!({"job_id": "123", "user": "test"})),
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["extra"]["job_id"], "123");
}

#[test]
fn verbose_response_omits_null_extra() {
    let resp = VerboseJsonResponse {
        text: "hi".to_string(),
        language: "en".to_string(),
        duration: 1.0,
        language_confidence: None,
        segments: vec![],
        words: vec![],
        extra: None,
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert!(json.get("extra").is_none());
}
```

**Step 2: Add `extra` to response types**

```rust
// openai.rs
pub struct VerboseJsonResponse {
    // ... existing fields ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

pub struct JsonResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}
```

**Step 3: Parse `extra` in handler_openai.rs**

Accept `extra` multipart field as JSON string. Parse into `serde_json::Value`.
Pass through to `format_response`.

**Step 4: Run tests**

Run: `cargo test`
Expected: all tests pass

**Step 5: Commit**

```bash
git add src/openai.rs src/handler_openai.rs
git commit -m "feat: add extra passthrough for request metadata"
```

---

### Task 4: Version bump + deploy + integration test

**Step 1: Update Cargo.toml version to 0.3.0**

**Step 2: Run all tests**

Run: `cargo test`
Expected: all tests pass (32 existing + new)

**Step 3: Build and deploy Docker**

```bash
cd ~/deploy/krolik-server
docker compose build --no-cache moonshine
docker compose up -d --no-deps --force-recreate moonshine
```

**Step 4: Integration tests**

Test `language_confidence`:
```bash
curl -s -X POST http://127.0.0.1:8092/v1/audio/transcriptions \
  -F "file=@test.wav" -F "response_format=verbose_json" | jq .language_confidence
```

Test `custom_spelling`:
```bash
curl -s -X POST http://127.0.0.1:8092/v1/audio/transcriptions \
  -F "file=@test.wav" -F 'custom_spelling=[{"from":["test"],"to":"TEST"}]' | jq .text
```

Test `extra`:
```bash
curl -s -X POST http://127.0.0.1:8092/v1/audio/transcriptions \
  -F "file=@test.wav" -F 'extra={"job_id":"abc"}' -F "response_format=verbose_json" | jq .extra
```

**Step 5: Commit version bump**

```bash
git add Cargo.toml
git commit -m "chore: bump version to 0.3.0"
```
