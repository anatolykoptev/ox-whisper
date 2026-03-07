/// Unified recognizer enum wrapping different sherpa-onnx backends for RU.

use sherpa_rs::nemo_ctc::NemoCtcRecognizer;
use sherpa_rs::transducer::TransducerRecognizer;

pub enum RuRecognizer {
    Transducer(TransducerRecognizer),
    NemoCtc(NemoCtcRecognizer),
}

impl RuRecognizer {
    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> String {
        match self {
            Self::Transducer(r) => r.transcribe(sample_rate, samples),
            Self::NemoCtc(r) => r.transcribe(sample_rate, samples).text,
        }
    }
}
