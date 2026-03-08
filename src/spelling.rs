/// Custom spelling replacement — apply domain-specific word corrections.

use serde::Deserialize;
use strsim;
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

/// Normalized Levenshtein similarity (0.0-1.0).
fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    if a == b { return 1.0; }
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 { return 1.0; }
    1.0 - (strsim::levenshtein(&a, &b) as f64 / max_len as f64)
}

/// Boost keywords by fuzzy-matching words in text.
/// Words with similarity >= threshold to any keyword get replaced.
/// Short words (<4 chars) require exact match to avoid false positives.
pub fn apply_keyword_boost(text: &str, keywords: &[String], threshold: f64) -> String {
    text.split_whitespace()
        .map(|word| {
            let (clean, punct) = split_trailing_punct(word);
            if clean.chars().count() < 4 {
                for kw in keywords {
                    if clean.to_lowercase() == kw.to_lowercase() {
                        return format!("{kw}{punct}");
                    }
                }
            } else {
                for kw in keywords {
                    if levenshtein_similarity(clean, kw) >= threshold {
                        return format!("{kw}{punct}");
                    }
                }
            }
            word.to_string()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Apply keyword boost to word timestamps.
pub fn apply_keyword_boost_to_words(
    words: &mut [WordTimestamp],
    keywords: &[String],
    threshold: f64,
) {
    for word in words.iter_mut() {
        let clean = word.word.trim_end_matches(|c: char| c.is_ascii_punctuation());
        if clean.chars().count() < 4 {
            for kw in keywords {
                if clean.to_lowercase() == kw.to_lowercase() {
                    word.word = kw.clone();
                    break;
                }
            }
        } else {
            for kw in keywords {
                if levenshtein_similarity(clean, kw) >= threshold {
                    word.word = kw.clone();
                    break;
                }
            }
        }
    }
}

fn split_trailing_punct(word: &str) -> (&str, &str) {
    let end = word.trim_end_matches(|c: char| c.is_ascii_punctuation());
    let punct = &word[end.len()..];
    (end, punct)
}

#[cfg(test)]
#[path = "spelling_tests.rs"]
mod tests;
