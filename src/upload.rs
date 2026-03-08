/// Multipart upload parsing for the OpenAI-compatible API.

use std::path::Path;

use axum::extract::Multipart;

use crate::openai::ResponseFormat;

pub struct OpenAIUpload {
    pub file_path: std::path::PathBuf,
    pub language: String,
    pub response_format: ResponseFormat,
    pub want_words: bool,
    pub custom_spelling: Vec<crate::spelling::SpellingRule>,
    pub smart_format: bool,
    pub paragraphs: bool,
    pub pii_types: Vec<crate::pii::PiiEntityType>,
    pub pii_format: crate::pii::RedactFormat,
    pub keywords: Vec<String>,
    pub keywords_boost: f64,
    pub diarize: bool,
    pub diarize_speakers: Option<i32>,
    pub extra: Option<serde_json::Value>,
}

pub async fn parse_openai_upload(multipart: &mut Multipart) -> Result<OpenAIUpload, String> {
    let mut file_path: Option<std::path::PathBuf> = None;
    let mut language = String::new();
    let mut response_format = ResponseFormat::default();
    let mut want_words = false;
    let mut custom_spelling = Vec::new();
    let mut smart_format_flag = false;
    let mut paragraphs_flag = false;
    let mut pii_types = Vec::new();
    let mut pii_format = crate::pii::RedactFormat::default();
    let mut keywords: Vec<String> = Vec::new();
    let mut keywords_boost: f64 = 0.8;
    let mut diarize_flag = false;
    let mut diarize_speakers: Option<i32> = None;
    let mut extra: Option<serde_json::Value> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let ext = field
                    .file_name()
                    .and_then(|n| {
                        Path::new(n).extension().map(|e| e.to_string_lossy().to_string())
                    })
                    .unwrap_or_else(|| "wav".to_string());
                let tmp = format!("/tmp/{}.{}", uuid::Uuid::new_v4(), ext);
                let data = field.bytes().await.map_err(|e| e.to_string())?;
                std::fs::write(&tmp, &data).map_err(|e: std::io::Error| e.to_string())?;
                file_path = Some(std::path::PathBuf::from(tmp));
            }
            "language" => language = field.text().await.unwrap_or_default(),
            "response_format" => {
                let val = field.text().await.unwrap_or_default();
                let quoted = format!("\"{}\"", val);
                response_format = serde_json::from_str(&quoted).unwrap_or_default();
            }
            "timestamp_granularities[]" => {
                let val = field.text().await.unwrap_or_default();
                if val == "word" { want_words = true; }
            }
            "custom_spelling" => {
                let val = field.text().await.unwrap_or_default();
                if let Ok(rules) = serde_json::from_str::<Vec<crate::spelling::SpellingRule>>(&val) {
                    custom_spelling = rules;
                }
            }
            "smart_format" => {
                let val = field.text().await.unwrap_or_default();
                smart_format_flag = val == "true" || val == "1";
            }
            "paragraphs" => {
                let val = field.text().await.unwrap_or_default();
                paragraphs_flag = val == "true" || val == "1";
            }
            "redact" => {
                let val = field.text().await.unwrap_or_default();
                pii_types = crate::pii::parse_pii_types(&val);
            }
            "redact_format" => {
                let val = field.text().await.unwrap_or_default();
                pii_format = match val.as_str() {
                    "mask" => crate::pii::RedactFormat::Mask,
                    _ => crate::pii::RedactFormat::Marker,
                };
            }
            "keywords" => {
                let val = field.text().await.unwrap_or_default();
                if let Ok(kw) = serde_json::from_str::<Vec<String>>(&val) {
                    keywords = kw;
                }
            }
            "keywords_boost" => {
                let val = field.text().await.unwrap_or_default();
                keywords_boost = val.parse().unwrap_or(0.8);
            }
            "diarize" => {
                let val = field.text().await.unwrap_or_default();
                diarize_flag = val == "true" || val == "1";
            }
            "diarize_speakers" => {
                let val = field.text().await.unwrap_or_default();
                diarize_speakers = val.parse().ok();
            }
            "extra" => {
                let val = field.text().await.unwrap_or_default();
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&val) {
                    extra = Some(parsed);
                }
            }
            // model, temperature, prompt — accepted but ignored
            _ => { let _ = field.bytes().await; }
        }
    }

    Ok(OpenAIUpload {
        file_path: file_path.ok_or("missing 'file' field")?,
        language: normalize_language(&language),
        response_format,
        want_words,
        custom_spelling,
        smart_format: smart_format_flag,
        paragraphs: paragraphs_flag,
        pii_types,
        pii_format,
        keywords,
        keywords_boost,
        diarize: diarize_flag,
        diarize_speakers,
        extra,
    })
}

fn normalize_language(lang: &str) -> String {
    lang.trim().to_lowercase()
}
