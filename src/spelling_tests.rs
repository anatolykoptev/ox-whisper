use super::*;

fn word(w: &str) -> WordTimestamp {
    WordTimestamp { word: w.to_string(), start: 0.0, end: 0.0, confidence: None, speaker: None }
}

// --- spelling rules tests ---

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
    let rules = vec![SpellingRule {
        from: vec!["докер".into(), "докера".into()],
        to: "Docker".into(),
    }];
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

// --- keyword boost tests ---

#[test]
fn keyword_exact_match() {
    assert_eq!(
        apply_keyword_boost("kubernetes is great", &["Kubernetes".into()], 0.8),
        "Kubernetes is great"
    );
}

#[test]
fn keyword_fuzzy_match() {
    // "kubernetis" is 1 char different from "Kubernetes" (10 chars) = 0.9 similarity
    assert_eq!(
        apply_keyword_boost("kubernetis is great", &["Kubernetes".into()], 0.8),
        "Kubernetes is great"
    );
}

#[test]
fn keyword_below_threshold() {
    assert_eq!(
        apply_keyword_boost("hello world", &["Kubernetes".into()], 0.8),
        "hello world"
    );
}

#[test]
fn keyword_multiple() {
    assert_eq!(
        apply_keyword_boost(
            "deploying doker on kubernetis",
            &["Docker".into(), "Kubernetes".into()],
            0.7,
        ),
        "deploying Docker on Kubernetes"
    );
}

#[test]
fn keyword_short_word_exact_only() {
    // Short words (<4 chars) should not fuzzy match
    assert_eq!(
        apply_keyword_boost("the cat sat", &["car".into()], 0.5),
        "the cat sat" // "cat" should NOT match "car" even though similar
    );
}

#[test]
fn keyword_preserves_punctuation() {
    assert_eq!(
        apply_keyword_boost("using kubernetis.", &["Kubernetes".into()], 0.8),
        "using Kubernetes."
    );
}

#[test]
fn keyword_no_keywords() {
    assert_eq!(
        apply_keyword_boost("hello world", &[], 0.8),
        "hello world"
    );
}

#[test]
fn keyword_word_timestamps() {
    let mut words = vec![word("kubernetis"), word("deploy")];
    apply_keyword_boost_to_words(&mut words, &["Kubernetes".into()], 0.8);
    assert_eq!(words[0].word, "Kubernetes");
    assert_eq!(words[1].word, "deploy");
}
