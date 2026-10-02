/// OpenAI-compatible API response types for /v1/audio/transcriptions.
use crate::words::WordTimestamp;

const WORDS_PER_SEGMENT: usize = 8;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum ResponseFormat {
    #[default]
    Json,
    VerboseJson,
    Text,
    Srt,
    Vtt,
}

/// A `response_format` the server does not produce. Falling back to `json`
/// would hand a client that asked for `srt` a JSON body with a 200.
#[derive(Debug, PartialEq, Eq)]
pub struct InvalidResponseFormat(pub String);

impl std::fmt::Display for InvalidResponseFormat {
    // Wording avoids "unsupported", "corrupted" and "invalid file": OpenAI-SDK
    // callers retry a 400 containing them as a bad-container error.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "response_format '{}' is not valid; use one of: json, verbose_json, text, srt, vtt",
            self.0
        )
    }
}

impl axum::response::IntoResponse for InvalidResponseFormat {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": "invalid_request_error",
                "param": "response_format",
                "code": "invalid_response_format",
            }
        });
        (axum::http::StatusCode::BAD_REQUEST, axum::Json(body)).into_response()
    }
}

/// Parses the `response_format` field. Empty means the default, `json`;
/// anything that is not one of the five formats is an error.
pub fn parse_response_format(raw: &str) -> Result<ResponseFormat, InvalidResponseFormat> {
    match raw.trim() {
        "" | "json" => Ok(ResponseFormat::Json),
        "verbose_json" => Ok(ResponseFormat::VerboseJson),
        "text" => Ok(ResponseFormat::Text),
        "srt" => Ok(ResponseFormat::Srt),
        "vtt" => Ok(ResponseFormat::Vtt),
        other => Err(InvalidResponseFormat(other.to_string())),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct JsonResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Segment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VerboseJsonResponse {
    pub text: String,
    /// The language hint the request carried; absent when it carried none —
    /// the model does not report what it heard, and none is guessed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub duration: f64,
    pub segments: Vec<Segment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

/// Group word timestamps into segments of up to `WORDS_PER_SEGMENT` words.
pub fn words_to_segments(words: &[WordTimestamp]) -> Vec<Segment> {
    words
        .chunks(WORDS_PER_SEGMENT)
        .enumerate()
        .map(|(id, chunk)| {
            let text = chunk
                .iter()
                .map(|w| w.word.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            Segment {
                id,
                start: chunk.first().map_or(0.0, |w| w.start as f64),
                end: chunk.last().map_or(0.0, |w| w.end as f64),
                text,
            }
        })
        .collect()
}

/// Convert internal word timestamps to OpenAI Word format.
pub fn words_to_openai(words: &[WordTimestamp]) -> Vec<Word> {
    words
        .iter()
        .map(|w| Word {
            word: w.word.clone(),
            start: w.start as f64,
            end: w.end as f64,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_format_is_json() {
        assert_eq!(ResponseFormat::default(), ResponseFormat::Json);
    }

    #[test]
    fn response_formats_parse_and_unknown_ones_are_errors() {
        for (raw, want) in [
            ("", ResponseFormat::Json),
            ("json", ResponseFormat::Json),
            ("verbose_json", ResponseFormat::VerboseJson),
            ("text", ResponseFormat::Text),
            ("srt", ResponseFormat::Srt),
            ("vtt", ResponseFormat::Vtt),
        ] {
            assert_eq!(parse_response_format(raw), Ok(want), "{raw:?}");
        }
        for bad in ["verbose-json", "JSON", "xml", "diarized_json"] {
            assert_eq!(
                parse_response_format(bad),
                Err(InvalidResponseFormat(bad.to_string())),
                "{bad}"
            );
        }
        let msg = InvalidResponseFormat("xml".into())
            .to_string()
            .to_lowercase();
        for word in ["unsupported", "corrupted", "invalid file"] {
            assert!(!msg.contains(word), "{msg}");
        }
    }

    #[test]
    fn words_to_segments_groups_by_eight() {
        let words: Vec<WordTimestamp> = (0..20)
            .map(|i| WordTimestamp {
                word: format!("w{i}"),
                start: i as f32,
                end: i as f32 + 0.5,
                confidence: None,
            })
            .collect();

        let segments = words_to_segments(&words);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].id, 0);
        assert_eq!(segments[0].text, "w0 w1 w2 w3 w4 w5 w6 w7");
        assert!((segments[0].start - 0.0).abs() < f64::EPSILON);
        assert!((segments[0].end - 7.5).abs() < f64::EPSILON);
        assert_eq!(segments[2].id, 2);
        assert_eq!(segments[2].text, "w16 w17 w18 w19");
    }

    #[test]
    fn json_response_serialize() {
        let resp = JsonResponse {
            text: "hello world".to_string(),
            extra: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["text"], "hello world");
    }

    fn verbose(language: Option<&str>) -> VerboseJsonResponse {
        VerboseJsonResponse {
            text: "hi".to_string(),
            language: language.map(str::to_string),
            duration: 1.5,
            segments: vec![],
            words: vec![],
            extra: None,
        }
    }

    #[test]
    fn verbose_response_omits_empty_words() {
        let json = serde_json::to_string(&verbose(Some("en"))).unwrap();
        assert!(!json.contains("words"));
    }

    /// The fields clients read from `verbose_json`: text,
    /// language (echoed hint), duration.
    #[test]
    fn verbose_response_keeps_the_fields_clients_read() {
        let json = serde_json::to_value(verbose(Some("ru"))).unwrap();
        assert_eq!(json["text"], "hi");
        assert_eq!(json["language"], "ru");
        assert_eq!(json["duration"], 1.5);
        assert!(json["segments"].is_array());
    }

    /// No hint, no language: the model does not report one and none is made up.
    #[test]
    fn verbose_response_has_no_language_when_none_was_given() {
        let json = serde_json::to_value(verbose(None)).unwrap();
        assert!(json.get("language").is_none(), "{json}");
    }

    #[test]
    fn json_response_includes_extra() {
        let resp = JsonResponse {
            text: "hi".to_string(),
            extra: Some(serde_json::json!({"job_id": "123"})),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["extra"]["job_id"], "123");
    }

    #[test]
    fn verbose_response_omits_null_extra() {
        let json = serde_json::to_value(verbose(Some("en"))).unwrap();
        assert!(json.get("extra").is_none());
    }
}
