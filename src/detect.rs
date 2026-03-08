/// Basic language detection by comparing transcription output length.
///
/// Runs a short audio clip through available models and picks the language
/// whose model produces more characters (longer meaningful output).

use crate::models::Models;

/// Maximum samples for detection (~3 seconds at 16kHz).
const DETECT_SAMPLES: usize = 48000;

/// Detect language by transcribing a short clip with each model.
///
/// Returns "en" or "ru" based on which model produces more characters.
/// Falls back to "en" if no models are loaded or both produce empty output.
pub(crate) fn detect_language(models: &Models, samples: &[f32]) -> String {
    let clip = if samples.len() > DETECT_SAMPLES {
        &samples[..DETECT_SAMPLES]
    } else {
        samples
    };

    let en_len = try_transcribe_en(models, clip);
    let ru_len = try_transcribe_ru(models, clip);

    tracing::debug!("language detection: en={} chars, ru={} chars", en_len, ru_len);

    if ru_len > en_len { "ru".to_string() } else { "en".to_string() }
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
