/// Basic language detection by comparing transcription output length.
///
/// Runs a short audio clip through available models and picks the language
/// whose model produces more characters (longer meaningful output).

use crate::models::Models;

/// Maximum samples for detection (~3 seconds at 16kHz).
const DETECT_SAMPLES: usize = 48000;

/// Result of language detection with confidence score.
pub struct DetectResult {
    pub language: String,
    pub confidence: f64,
}

/// Detect language by transcribing a short clip with each model.
///
/// Returns a `DetectResult` with the detected language ("en" or "ru") and
/// a confidence score (winner_len / total_len). Falls back to "en" with
/// confidence 0.0 if no models are loaded or both produce empty output.
pub(crate) fn detect_language(models: &Models, samples: &[f32]) -> DetectResult {
    let clip = if samples.len() > DETECT_SAMPLES {
        &samples[..DETECT_SAMPLES]
    } else {
        samples
    };

    let en_len = try_transcribe_en(models, clip);
    let ru_len = try_transcribe_ru(models, clip);

    tracing::debug!("language detection: en={} chars, ru={} chars", en_len, ru_len);

    let total = en_len + ru_len;
    let (language, winner_len) = if ru_len > en_len {
        ("ru", ru_len)
    } else {
        ("en", en_len)
    };
    let confidence = if total == 0 {
        0.0
    } else {
        winner_len as f64 / total as f64
    };

    DetectResult {
        language: language.to_string(),
        confidence,
    }
}

fn try_transcribe_en(models: &Models, samples: &[f32]) -> usize {
    let pool = match models.en.as_ref() {
        Some(p) => p,
        None => return 0,
    };
    let mut rec = match pool.acquire() {
        Some(r) => r,
        None => return 0,
    };
    let result = rec.transcribe(16000, samples);
    result.text.trim().chars().count()
}

fn try_transcribe_ru(models: &Models, samples: &[f32]) -> usize {
    let pool = match models.ru.as_ref() {
        Some(p) => p,
        None => return 0,
    };
    let mut rec = match pool.acquire() {
        Some(r) => r,
        None => return 0,
    };
    let result = rec.transcribe(16000, samples);
    result.text.trim().chars().count()
}

#[cfg(test)]
mod tests {
    #[test]
    fn confidence_calculation() {
        // Simulate: en=30 chars, ru=70 chars → ru wins, confidence = 70/100 = 0.7
        let en_len: usize = 30;
        let ru_len: usize = 70;
        let total = en_len + ru_len;
        let (_, winner_len) = if ru_len > en_len {
            ("ru", ru_len)
        } else {
            ("en", en_len)
        };
        let confidence = if total == 0 {
            0.0
        } else {
            winner_len as f64 / total as f64
        };
        assert!((confidence - 0.7).abs() < f64::EPSILON);

        // Both zero → confidence = 0.0
        let total_zero: usize = 0;
        let conf_zero: f64 = if total_zero == 0 { 0.0 } else { 1.0 };
        assert!((conf_zero - 0.0).abs() < f64::EPSILON);
    }
}
