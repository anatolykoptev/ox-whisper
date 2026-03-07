use sherpa_rs::online_punctuate::OnlinePunctuation;

/// Adds punctuation to text using the CNN-BiLSTM online punctuation model.
/// Returns empty string if input is empty.
pub fn add_punctuation(punct: &OnlinePunctuation, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    punct.add_punctuation(text)
}
