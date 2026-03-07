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

    /// Batch-decode multiple chunks. Falls back to sequential for Transducer.
    pub fn transcribe_batch(&mut self, sample_rate: u32, chunks: &[&[f32]]) -> Vec<String> {
        match self {
            Self::NemoCtc(r) => {
                r.transcribe_batch(sample_rate, chunks)
                    .into_iter()
                    .map(|r| r.text)
                    .collect()
            }
            Self::Transducer(r) => {
                chunks.iter().map(|c| r.transcribe(sample_rate, c)).collect()
            }
        }
    }
}
