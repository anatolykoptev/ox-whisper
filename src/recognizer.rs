/// Unified recognizer enum wrapping different sherpa-onnx backends for RU.

use sherpa_rs::OfflineRecognizerResult;
use sherpa_rs::nemo_ctc::NemoCtcRecognizer;
use sherpa_rs::transducer::TransducerRecognizer;

pub enum RuRecognizer {
    /// NeMo transducer (GigaAM v3) — has built-in punctuation
    NemoTransducer(TransducerRecognizer),
    /// Standard Zipformer transducer — no built-in punctuation
    Transducer(TransducerRecognizer),
    NemoCtc(NemoCtcRecognizer),
}

impl RuRecognizer {
    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> OfflineRecognizerResult {
        match self {
            Self::NemoTransducer(r) | Self::Transducer(r) => r.transcribe(sample_rate, samples),
            Self::NemoCtc(r) => r.transcribe(sample_rate, samples),
        }
    }

    pub fn transcribe_batch(&mut self, sample_rate: u32, chunks: &[&[f32]]) -> Vec<OfflineRecognizerResult> {
        match self {
            Self::NemoCtc(r) => r.transcribe_batch(sample_rate, chunks),
            Self::NemoTransducer(r) | Self::Transducer(r) => r.transcribe_batch(sample_rate, chunks),
        }
    }

    /// Returns true if the model produces punctuated text natively (GigaAM v3 NeMo transducer).
    pub fn has_builtin_punct(&self) -> bool {
        matches!(self, Self::NemoTransducer(_))
    }

    pub fn model_name(&self) -> &'static str {
        match self {
            Self::NemoTransducer(_) => "gigaam-v3-rnnt-punct",
            Self::Transducer(_) => "zipformer-transducer",
            Self::NemoCtc(_) => "gigaam-v2-ctc",
        }
    }
}
