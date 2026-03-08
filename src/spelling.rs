/// Custom spelling replacement — apply domain-specific word corrections.

use serde::Deserialize;
use crate::words::WordTimestamp;

#[derive(Debug, Clone, Deserialize)]
pub struct SpellingRule {
    pub from: Vec<String>,
    pub to: String,
}

/// Apply spelling rules to full text. Case-insensitive word replacement.
pub fn apply_spelling(text: &str, rules: &[SpellingRule]) -> String {
    if rules.is_empty() {
        return text.to_string();
    }
    let mut result = text.to_string();
    for rule in rules {
        for from in &rule.from {
            result = replace_word_ci(&result, from, &rule.to);
        }
    }
    result
}

/// Apply spelling rules to individual word timestamps.
pub fn apply_spelling_to_words(words: &mut [WordTimestamp], rules: &[SpellingRule]) {
    for word in words.iter_mut() {
        for rule in rules {
            if rule.from.iter().any(|f| words_match(&word.word, f)) {
                word.word = rule.to.clone();
                break;
            }
        }
    }
}

/// Case-insensitive word match (handles Unicode).
fn words_match(a: &str, b: &str) -> bool {
    a.to_lowercase() == b.to_lowercase()
}

/// Replace whole words case-insensitively by splitting on whitespace.
fn replace_word_ci(text: &str, from: &str, to: &str) -> String {
    text.split_whitespace()
        .map(|w| if words_match(w, from) { to.to_string() } else { w.to_string() })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(w: &str) -> WordTimestamp {
        WordTimestamp { word: w.to_string(), start: 0.0, end: 0.0, confidence: None }
    }

    #[test]
    fn no_rules_returns_unchanged() {
        assert_eq!(apply_spelling("hello world", &[]), "hello world");
    }

    #[test]
    fn single_replacement() {
        let rules = vec![SpellingRule { from: vec!["докер".into()], to: "Docker".into() }];
        assert_eq!(apply_spelling("запустим докер", &rules), "запустим Docker");
    }

    #[test]
    fn multiple_from_variants() {
        let rules = vec![SpellingRule { from: vec!["докер".into(), "докера".into()], to: "Docker".into() }];
        assert_eq!(apply_spelling("версия докера", &rules), "версия Docker");
    }

    #[test]
    fn case_insensitive() {
        let rules = vec![SpellingRule { from: vec!["kubernetes".into()], to: "Kubernetes".into() }];
        assert_eq!(apply_spelling("deploy to kubernetes", &rules), "deploy to Kubernetes");
    }

    #[test]
    fn word_boundary_respected() {
        let rules = vec![SpellingRule { from: vec!["к".into()], to: "K".into() }];
        assert_eq!(apply_spelling("к нам", &rules), "K нам");
        assert_eq!(apply_spelling("как дела", &rules), "как дела");
    }

    #[test]
    fn apply_to_word_timestamps() {
        let rules = vec![SpellingRule { from: vec!["докер".into()], to: "Docker".into() }];
        let mut words = vec![word("запустим"), word("докер")];
        apply_spelling_to_words(&mut words, &rules);
        assert_eq!(words[0].word, "запустим");
        assert_eq!(words[1].word, "Docker");
    }
}
