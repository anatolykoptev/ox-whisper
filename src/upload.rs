/// Multipart upload parsing for the OpenAI-compatible API.
use std::path::Path;

use axum::extract::Multipart;
use axum::extract::multipart::Field;

use crate::tmpfile::TempFile;

pub struct OpenAIUpload {
    /// The uploaded audio, removed when the last owner drops it. Shared so a
    /// blocking decode job can outlive a cancelled handler without losing the
    /// file under it.
    pub file: std::sync::Arc<TempFile>,
    /// The language hint as sent; validated with [`crate::language::resolve`].
    pub language: String,
    /// As sent; validated with [`crate::openai::parse_response_format`].
    pub response_format: String,
    pub want_words: bool,
    pub custom_spelling: Vec<crate::spelling::SpellingRule>,
    pub smart_format: bool,
    pub paragraphs: bool,
    pub pii_types: Vec<crate::pii::PiiEntityType>,
    pub pii_format: crate::pii::RedactFormat,
    pub keywords: Vec<String>,
    pub keywords_boost: f64,
    pub extra: Option<serde_json::Value>,
}

/// Stores a multipart audio part in `dir` and puts it in `slot`.
///
/// A request carries one audio part. A second one is refused before its body
/// is read, and the first stays in `slot`, so whoever drops the slot removes
/// it: the error path cannot strand a file.
pub(crate) async fn store_audio_part(
    field: Field<'_>,
    dir: &Path,
    slot: &mut Option<TempFile>,
) -> Result<(), String> {
    if slot.is_some() {
        return Err("only one audio file per request".to_string());
    }
    let ext = field
        .file_name()
        .and_then(|n| {
            Path::new(n)
                .extension()
                .map(|e| e.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| "wav".to_string());
    let data = field.bytes().await.map_err(|e| e.to_string())?;
    *slot = Some(TempFile::create(dir, &ext, &data).map_err(|e| e.to_string())?);
    Ok(())
}

/// A text part's value. A part that cannot be read (truncated body, invalid
/// UTF-8) is an error, not an empty string: an empty `language` would silently
/// change which model answers.
pub(crate) async fn text_part(field: Field<'_>) -> Result<String, String> {
    let name = field.name().unwrap_or("").to_string();
    field
        .text()
        .await
        .map_err(|e| format!("field '{name}': {e}"))
}

/// The next multipart part. A body that ends abruptly or is malformed is an
/// error: treating it as "no more fields" would drop later fields silently.
pub(crate) async fn next_part<'a>(
    multipart: &'a mut Multipart,
) -> Result<Option<Field<'a>>, String> {
    multipart
        .next_field()
        .await
        .map_err(|e| format!("invalid multipart body: {e}"))
}

pub async fn parse_openai_upload(
    multipart: &mut Multipart,
    dir: &Path,
) -> Result<OpenAIUpload, String> {
    let mut file: Option<TempFile> = None;
    let mut language = String::new();
    let mut response_format = String::new();
    let mut want_words = false;
    let mut custom_spelling = Vec::new();
    let mut smart_format_flag = false;
    let mut paragraphs_flag = false;
    let mut pii_types = Vec::new();
    let mut pii_format = crate::pii::RedactFormat::default();
    let mut keywords: Vec<String> = Vec::new();
    let mut keywords_boost: f64 = 0.8;
    let mut extra: Option<serde_json::Value> = None;

    while let Some(field) = next_part(multipart).await? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => store_audio_part(field, dir, &mut file).await?,
            "language" => language = text_part(field).await?,
            "response_format" => response_format = text_part(field).await?,
            "timestamp_granularities[]" => {
                let val = text_part(field).await?;
                if val == "word" {
                    want_words = true;
                }
            }
            "custom_spelling" => {
                let val = text_part(field).await?;
                if let Ok(rules) = serde_json::from_str::<Vec<crate::spelling::SpellingRule>>(&val)
                {
                    custom_spelling = rules;
                }
            }
            "smart_format" => {
                let val = text_part(field).await?;
                smart_format_flag = val == "true" || val == "1";
            }
            "paragraphs" => {
                let val = text_part(field).await?;
                paragraphs_flag = val == "true" || val == "1";
            }
            "redact" => {
                let val = text_part(field).await?;
                pii_types = crate::pii::parse_pii_types(&val);
            }
            "redact_format" => {
                let val = text_part(field).await?;
                pii_format = match val.as_str() {
                    "mask" => crate::pii::RedactFormat::Mask,
                    _ => crate::pii::RedactFormat::Marker,
                };
            }
            "keywords" => {
                let val = text_part(field).await?;
                if let Ok(kw) = serde_json::from_str::<Vec<String>>(&val) {
                    keywords = kw;
                }
            }
            "keywords_boost" => {
                let val = text_part(field).await?;
                keywords_boost = val.parse().unwrap_or(0.8);
            }
            // Speaker diarization was removed: answering without speakers would
            // look like a single-speaker result, so it is refused instead.
            "diarize" => {
                let val = text_part(field).await?;
                if val == "true" || val == "1" {
                    return Err("diarization is not supported".to_string());
                }
            }
            "extra" => {
                let val = text_part(field).await?;
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&val) {
                    extra = Some(parsed);
                }
            }
            // model, temperature, prompt — accepted but ignored
            _ => {
                field.bytes().await.map_err(|e| e.to_string())?;
            }
        }
    }

    Ok(OpenAIUpload {
        file: std::sync::Arc::new(file.ok_or("missing 'file' field")?),
        language,
        response_format,
        want_words,
        custom_spelling,
        smart_format: smart_format_flag,
        paragraphs: paragraphs_flag,
        pii_types,
        pii_format,
        keywords,
        keywords_boost,
        extra,
    })
}

#[cfg(test)]
#[path = "upload_tests.rs"]
mod tests;
