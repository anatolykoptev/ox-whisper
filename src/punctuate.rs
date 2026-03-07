use sherpa_rs::punctuate::Punctuation;

/// Adds punctuation to text using the sherpa-rs punctuation model.
/// Returns empty string if input is empty.
pub fn add_punctuation(punct: &mut Punctuation, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    punct.add_punctuation(text)
}
