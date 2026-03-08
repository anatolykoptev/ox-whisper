/// OpenAI-compatible API response types for /v1/audio/transcriptions.

use crate::words::WordTimestamp;

const WORDS_PER_SEGMENT: usize = 8;

#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Json,
    VerboseJson,
    Text,
    Srt,
    Vtt,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct JsonResponse {
    pub text: String,
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
    pub language: String,
    pub duration: f64,
    pub segments: Vec<Segment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
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
    fn deserialize_verbose_json() {
        let fmt: ResponseFormat =
            serde_json::from_str(r#""verbose_json""#).unwrap();
        assert_eq!(fmt, ResponseFormat::VerboseJson);
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
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["text"], "hello world");
    }

    #[test]
    fn verbose_response_omits_empty_words() {
        let resp = VerboseJsonResponse {
            text: "hi".to_string(),
            language: "en".to_string(),
            duration: 1.0,
            segments: vec![],
            words: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("words"));
    }
}
