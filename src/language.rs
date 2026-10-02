//! Which request languages this server answers.
//!
//! Every request is decoded by one model, Parakeet TDT 0.6B v3. It transcribes
//! the language it hears, so a language hint never selects a model; it is
//! checked against what the model was trained on and echoed back. A language
//! the model does not cover is refused with 400: decoding it anyway returns
//! plausible-looking text in the wrong language with HTTP 200, which a client
//! cannot tell from a correct answer.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Languages Parakeet TDT 0.6B v3 transcribes (its model card's 25 European
/// languages), as ISO 639-1 codes.
pub const PARAKEET_V3_LANGS: &[&str] = &[
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];

/// A requested language the model does not cover.
#[derive(Debug, PartialEq, Eq)]
pub struct UnsupportedLanguage(pub String);

impl std::fmt::Display for UnsupportedLanguage {
    // "not supported", never the word "unsupported": the OpenAI Python SDK's
    // callers commonly retry a 400 whose text contains it as a bad-container
    // error, transcoding the audio for a request that can never succeed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "language '{}' is not supported; supported: {}",
            self.0,
            PARAKEET_V3_LANGS.join(", ")
        )
    }
}

/// Validates a request's language hint.
///
/// `None` means "no hint": an empty value or `auto`. Otherwise the primary
/// subtag of the tag, lowercased (`ru-RU` is `ru`), must be one of
/// [`PARAKEET_V3_LANGS`]; the returned code is that subtag.
pub fn resolve(raw: &str) -> Result<Option<&'static str>, UnsupportedLanguage> {
    let tag = raw.trim().to_lowercase();
    if tag.is_empty() || tag == "auto" {
        return Ok(None);
    }
    let primary = tag.split(['-', '_']).next().unwrap_or_default();
    PARAKEET_V3_LANGS
        .iter()
        .find(|l| **l == primary)
        .map(|l| Some(*l))
        .ok_or_else(|| UnsupportedLanguage(raw.trim().to_string()))
}

impl IntoResponse for UnsupportedLanguage {
    /// 400 with an OpenAI-style error body.
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": "invalid_request_error",
                "param": "language",
                "code": "language_not_supported",
            }
        });
        (StatusCode::BAD_REQUEST, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_parakeet_language_is_accepted_as_itself() {
        for lang in PARAKEET_V3_LANGS {
            assert_eq!(resolve(lang), Ok(Some(*lang)), "{lang}");
        }
        assert_eq!(PARAKEET_V3_LANGS.len(), 25);
    }

    /// The routing table: what the consumers send, what clients commonly send,
    /// and what the model cannot do.
    #[test]
    fn routing_table_per_language() {
        // No hint: Parakeet transcribes what it hears.
        for none in ["", "  ", "auto", "AUTO"] {
            assert_eq!(resolve(none), Ok(None), "{none:?}");
        }
        // Spelling the same language differently.
        for (raw, want) in [
            ("ru", "ru"),
            (" RU ", "ru"),
            ("ru-RU", "ru"),
            ("en_US", "en"),
            ("EN-gb", "en"),
        ] {
            assert_eq!(resolve(raw), Ok(Some(want)), "{raw:?}");
        }
        // Not covered by Parakeet: refused, never decoded.
        for raw in ["zh", "ja", "ar", "vi", "ko", "hi", "xx", "russian", "zh-CN"] {
            assert_eq!(
                resolve(raw),
                Err(UnsupportedLanguage(raw.to_string())),
                "{raw:?}"
            );
        }
    }

    #[test]
    fn the_error_text_avoids_the_word_clients_retry_on() {
        let msg = UnsupportedLanguage("zh".into()).to_string();
        assert!(msg.contains("not supported"));
        assert!(!msg.to_lowercase().contains("unsupported"), "{msg}");
    }

    #[tokio::test]
    async fn the_response_is_a_400_with_an_openai_style_body() {
        let res = UnsupportedLanguage("zh".into()).into_response();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "language");
        assert_eq!(body["error"]["code"], "language_not_supported");
        assert!(!body.to_string().to_lowercase().contains("unsupported"));
    }
}
