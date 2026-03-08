/// Unified recognizer enum wrapping different sherpa-onnx backends for RU.

use sherpa_rs::OfflineRecognizerResult;
use sherpa_rs::nemo_ctc::NemoCtcRecognizer;
use sherpa_rs::transducer::TransducerRecognizer;

pub enum RuRecognizer {
    Transducer(TransducerRecognizer),
    NemoCtc(NemoCtcRecognizer),
}

impl RuRecognizer {
    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> OfflineRecognizerResult {
        match self {
            Self::Transducer(r) => r.transcribe(sample_rate, samples),
            Self::NemoCtc(r) => r.transcribe(sample_rate, samples),
        }
    }

    pub fn transcribe_batch(&mut self, sample_rate: u32, chunks: &[&[f32]]) -> Vec<OfflineRecognizerResult> {
        match self {
            Self::NemoCtc(r) => r.transcribe_batch(sample_rate, chunks),
            Self::Transducer(r) => r.transcribe_batch(sample_rate, chunks),
        }
    }

    /// Returns true if the model produces punctuated text natively (GigaAM v3 transducer).
    pub fn has_builtin_punct(&self) -> bool {
        matches!(self, Self::Transducer(_))
    }
}
