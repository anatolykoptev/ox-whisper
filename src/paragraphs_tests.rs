use super::*;

fn w(word: &str, start: f32, end: f32) -> WordTimestamp {
    WordTimestamp {
        word: word.to_string(),
        start,
        end,
        confidence: None,
        speaker: None,
    }
}

#[test]
fn empty_text() {
    let result = split_paragraphs("", &[], 1.5);
    assert_eq!(result, "");
}

#[test]
fn single_word() {
    let words = vec![w("hello", 0.0, 0.5)];
    let result = split_paragraphs("hello", &words, 1.5);
    assert_eq!(result, "hello");
}

#[test]
fn no_long_pauses() {
    let words = vec![
        w("hello", 0.0, 0.3),
        w("world", 0.4, 0.7),
        w("today", 0.8, 1.1),
    ];
    let result = split_paragraphs("hello world today", &words, 1.5);
    assert_eq!(result, "hello world today");
}

#[test]
fn one_paragraph_break() {
    let words = vec![
        w("hello", 0.0, 0.3),
        w("world", 0.4, 0.7),
        w("new", 2.5, 2.8),
        w("paragraph", 2.9, 3.3),
    ];
    let result = split_paragraphs("hello world new paragraph", &words, 1.5);
    assert_eq!(result, "hello world\n\nnew paragraph");
}

#[test]
fn multiple_paragraph_breaks() {
    let words = vec![
        w("one", 0.0, 0.3),
        w("two", 2.0, 2.3),
        w("three", 4.0, 4.3),
    ];
    let result = split_paragraphs("one two three", &words, 1.5);
    assert_eq!(result, "one\n\ntwo\n\nthree");
}

#[test]
fn threshold_exact_boundary() {
    let words = vec![
        w("hello", 0.0, 0.5),
        w("world", 2.0, 2.5),
    ];
    // gap = 2.0 - 0.5 = 1.5, exactly at threshold — should trigger break
    let result = split_paragraphs("hello world", &words, 1.5);
    assert_eq!(result, "hello\n\nworld");
}

#[test]
fn threshold_just_below() {
    let words = vec![
        w("hello", 0.0, 0.5),
        w("world", 1.99, 2.5),
    ];
    // gap = 1.49, just below threshold — no break
    let result = split_paragraphs("hello world", &words, 1.5);
    assert_eq!(result, "hello world");
}

#[test]
fn default_threshold_value() {
    assert!((default_threshold() - 1.5).abs() < f32::EPSILON);
}
