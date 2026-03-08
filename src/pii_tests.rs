use super::*;
use crate::words::WordTimestamp;

fn word(w: &str) -> WordTimestamp {
    WordTimestamp { word: w.to_string(), start: 0.0, end: 0.0, confidence: None, speaker: None }
}

#[test]
fn phone_us_marker() {
    let r = PiiRedactor::new();
    let out = r.redact_text("call +1 (555) 123-4567 now", &[PiiEntityType::Phone], RedactFormat::Marker);
    assert_eq!(out, "call [PHONE_1] now");
}

#[test]
fn phone_ru_detected() {
    let r = PiiRedactor::new();
    let out = r.redact_text("звоните +7 999 123-45-67", &[PiiEntityType::Phone], RedactFormat::Marker);
    assert!(out.contains("[PHONE_1]"), "got: {out}");
}

#[test]
fn email_detected() {
    let r = PiiRedactor::new();
    let out = r.redact_text("write to john@example.com", &[PiiEntityType::Email], RedactFormat::Marker);
    assert_eq!(out, "write to [EMAIL_1]");
}

#[test]
fn ssn_detected() {
    let r = PiiRedactor::new();
    let out = r.redact_text("SSN 123-45-6789", &[PiiEntityType::Ssn], RedactFormat::Marker);
    assert_eq!(out, "SSN [SSN_1]");
}

#[test]
fn credit_card_valid_luhn() {
    let r = PiiRedactor::new();
    let out = r.redact_text("card 4111111111111111", &[PiiEntityType::CreditCard], RedactFormat::Marker);
    assert_eq!(out, "card [CREDIT_CARD_1]");
}

#[test]
fn credit_card_invalid_luhn_ignored() {
    let r = PiiRedactor::new();
    let out = r.redact_text("card 1234567890123456", &[PiiEntityType::CreditCard], RedactFormat::Marker);
    assert_eq!(out, "card 1234567890123456");
}

#[test]
fn ip_address_detected() {
    let r = PiiRedactor::new();
    let out = r.redact_text("server at 192.168.1.100", &[PiiEntityType::IpAddress], RedactFormat::Marker);
    assert_eq!(out, "server at [IP_ADDRESS_1]");
}

#[test]
fn mask_format() {
    let r = PiiRedactor::new();
    let out = r.redact_text("write to john@example.com", &[PiiEntityType::Email], RedactFormat::Mask);
    assert_eq!(out, "write to ####");
}

#[test]
fn multiple_same_type_incrementing() {
    let r = PiiRedactor::new();
    let out = r.redact_text(
        "emails: a@b.com and c@d.com", &[PiiEntityType::Email], RedactFormat::Marker,
    );
    assert_eq!(out, "emails: [EMAIL_1] and [EMAIL_2]");
}

#[test]
fn no_match_unchanged() {
    let r = PiiRedactor::new();
    let out = r.redact_text("nothing here", &[PiiEntityType::Phone], RedactFormat::Marker);
    assert_eq!(out, "nothing here");
}

#[test]
fn parse_pii_all() {
    let types = parse_pii_types("pii");
    assert_eq!(types.len(), 5);
}

#[test]
fn parse_pii_selective() {
    let types = parse_pii_types("phone,email");
    assert_eq!(types, vec![PiiEntityType::Phone, PiiEntityType::Email]);
}

#[test]
fn word_level_redaction() {
    let r = PiiRedactor::new();
    let mut words = vec![word("call"), word("john@example.com")];
    r.redact_words(&mut words, &[PiiEntityType::Email], RedactFormat::Marker);
    assert_eq!(words[0].word, "call");
    assert_eq!(words[1].word, "[EMAIL_1]");
}

#[test]
fn mixed_entity_types() {
    let r = PiiRedactor::new();
    let out = r.redact_text(
        "contact john@example.com or call (555) 123-4567",
        &[PiiEntityType::Email, PiiEntityType::Phone],
        RedactFormat::Marker,
    );
    assert!(out.contains("[EMAIL_1]"), "got: {out}");
    assert!(out.contains("[PHONE_1]"), "got: {out}");
}

#[test]
fn luhn_valid() {
    assert!(luhn_check("4111111111111111"));
}

#[test]
fn luhn_invalid() {
    assert!(!luhn_check("1234567890123456"));
}
