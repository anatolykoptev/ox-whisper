use super::*;
use sherpa_rs::diarize::Segment;

fn w(word: &str, start: f32, end: f32) -> WordTimestamp {
    WordTimestamp {
        word: word.to_string(),
        start,
        end,
        confidence: None,
        speaker: None,
    }
}

fn seg(start: f32, end: f32, speaker: i32) -> Segment {
    Segment {
        start,
        end,
        speaker,
    }
}

#[test]
fn assign_speakers_single_segment() {
    let mut words = vec![w("hello", 0.1, 0.3), w("world", 0.5, 0.8)];
    let segments = vec![seg(0.0, 1.0, 0)];
    assign_speakers_to_words(&mut words, &segments);
    assert_eq!(words[0].speaker, Some(0));
    assert_eq!(words[1].speaker, Some(0));
}

#[test]
fn assign_speakers_two_segments() {
    let mut words = vec![
        w("hello", 0.1, 0.3),
        w("world", 0.5, 0.8),
        w("goodbye", 1.5, 1.8),
    ];
    let segments = vec![seg(0.0, 1.0, 0), seg(1.0, 2.0, 1)];
    assign_speakers_to_words(&mut words, &segments);
    assert_eq!(words[0].speaker, Some(0));
    assert_eq!(words[1].speaker, Some(0));
    assert_eq!(words[2].speaker, Some(1));
}

#[test]
fn assign_speakers_no_matching_segment() {
    let mut words = vec![w("orphan", 5.0, 5.5)];
    let segments = vec![seg(0.0, 1.0, 0)];
    assign_speakers_to_words(&mut words, &segments);
    assert_eq!(words[0].speaker, None);
}

#[test]
fn assign_speakers_boundary() {
    let mut words = vec![w("edge", 1.0, 1.3)];
    let segments = vec![seg(0.0, 1.0, 0), seg(1.0, 2.0, 1)];
    assign_speakers_to_words(&mut words, &segments);
    // word.start == seg.start of second segment should match second
    assert_eq!(words[0].speaker, Some(1));
}

#[test]
fn words_to_utterances_single_speaker() {
    let words = vec![
        WordTimestamp { speaker: Some(0), ..w("hello", 0.0, 0.3) },
        WordTimestamp { speaker: Some(0), ..w("world", 0.4, 0.7) },
    ];
    let utts = words_to_utterances(&words);
    assert_eq!(utts.len(), 1);
    assert_eq!(utts[0].speaker, 0);
    assert_eq!(utts[0].text, "hello world");
}

#[test]
fn words_to_utterances_alternating() {
    let words = vec![
        WordTimestamp { speaker: Some(0), ..w("hi", 0.0, 0.2) },
        WordTimestamp { speaker: Some(1), ..w("hey", 0.5, 0.8) },
        WordTimestamp { speaker: Some(0), ..w("bye", 1.0, 1.3) },
    ];
    let utts = words_to_utterances(&words);
    assert_eq!(utts.len(), 3);
    assert_eq!(utts[0].speaker, 0);
    assert_eq!(utts[1].speaker, 1);
    assert_eq!(utts[2].speaker, 0);
}

#[test]
fn words_to_utterances_empty() {
    let utts = words_to_utterances(&[]);
    assert!(utts.is_empty());
}

#[test]
fn utterance_serialization() {
    let u = Utterance {
        speaker: 1,
        start: 0.5,
        end: 1.5,
        text: "hello world".to_string(),
    };
    let json: serde_json::Value = serde_json::to_value(&u).unwrap();
    assert_eq!(json["speaker"], 1);
    assert_eq!(json["text"], "hello world");
    assert_eq!(json["start"], 0.5);
    assert_eq!(json["end"], 1.5);
}
