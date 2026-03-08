/// Speaker diarization wrapper around sherpa-onnx vendor code.

use std::path::Path;
use std::sync::Mutex;

use sherpa_rs::diarize::{Diarize, DiarizeConfig, Segment};

use crate::words::WordTimestamp;

pub struct DiarizeEngine {
    inner: Mutex<Diarize>,
}

impl DiarizeEngine {
    /// Load diarization models. Returns None if model files don't exist.
    pub fn load(segmentation_model: &str, embedding_model: &str) -> Option<Self> {
        if !Path::new(segmentation_model).exists()
            || !Path::new(embedding_model).exists()
        {
            tracing::warn!(
                "Diarization models not found (seg={}, emb={}), skipping",
                segmentation_model,
                embedding_model
            );
            return None;
        }
        let config = DiarizeConfig::default();
        match Diarize::new(segmentation_model, embedding_model, config) {
            Ok(d) => {
                tracing::info!(
                    "Diarization loaded (seg={}, emb={})",
                    segmentation_model,
                    embedding_model
                );
                Some(Self {
                    inner: Mutex::new(d),
                })
            }
            Err(e) => {
                tracing::error!("Diarization load failed: {}", e);
                None
            }
        }
    }

    /// Run diarization on audio samples, assign speaker IDs to words.
    pub fn assign_speakers(
        &self,
        samples: &[f32],
        words: &mut [WordTimestamp],
        _num_speakers: Option<i32>,
    ) -> bool {
        let mut diarize = match self.inner.lock() {
            Ok(d) => d,
            Err(_) => return false,
        };

        let segments = match diarize.compute(samples.to_vec(), None) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Diarization failed: {}", e);
                return false;
            }
        };

        assign_speakers_to_words(words, &segments);
        true
    }
}

/// Map diarization segments to words by time overlap.
/// For each word, find the segment where word.start falls in [seg.start, seg.end).
pub fn assign_speakers_to_words(words: &mut [WordTimestamp], segments: &[Segment]) {
    for word in words.iter_mut() {
        for seg in segments {
            if word.start >= seg.start && word.start < seg.end {
                word.speaker = Some(seg.speaker);
                break;
            }
        }
    }
}

/// Group consecutive words with the same speaker into utterances.
pub fn words_to_utterances(words: &[WordTimestamp]) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    let mut current_speaker: Option<i32> = None;
    let mut current_words: Vec<&WordTimestamp> = Vec::new();

    for word in words {
        let speaker = word.speaker.unwrap_or(-1);
        if current_speaker == Some(speaker) {
            current_words.push(word);
        } else {
            if !current_words.is_empty() {
                utterances.push(make_utterance(
                    &current_words,
                    current_speaker.unwrap_or(-1),
                ));
            }
            current_speaker = Some(speaker);
            current_words = vec![word];
        }
    }
    if !current_words.is_empty() {
        utterances.push(make_utterance(
            &current_words,
            current_speaker.unwrap_or(-1),
        ));
    }

    utterances
}

fn make_utterance(words: &[&WordTimestamp], speaker: i32) -> Utterance {
    Utterance {
        speaker,
        start: words.first().map_or(0.0, |w| w.start as f64),
        end: words.last().map_or(0.0, |w| w.end as f64),
        text: words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Utterance {
    pub speaker: i32,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[cfg(test)]
#[path = "diarize_tests.rs"]
mod tests;
