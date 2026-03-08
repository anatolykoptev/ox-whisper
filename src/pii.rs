/// PII redaction — detect and replace personally identifiable information.

use regex::Regex;

use crate::words::WordTimestamp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiEntityType {
    Phone,
    Email,
    Ssn,
    CreditCard,
    IpAddress,
}

const ALL_TYPES: [PiiEntityType; 5] = [
    PiiEntityType::Phone,
    PiiEntityType::Email,
    PiiEntityType::Ssn,
    PiiEntityType::CreditCard,
    PiiEntityType::IpAddress,
];

#[derive(Debug, Clone, Copy, Default)]
pub enum RedactFormat {
    #[default]
    Marker,
    Mask,
}

pub struct PiiRedactor {
    phone_us: Regex,
    phone_ru: Regex,
    email: Regex,
    ssn: Regex,
    credit_card: Regex,
    ipv4: Regex,
}

impl PiiRedactor {
    pub fn new() -> Self {
        Self {
            phone_us: Regex::new(r"(?:\+1[\s.\-]?|\b)\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b").unwrap(),
            phone_ru: Regex::new(r"(?:\+7|8)[\s.\-]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?\d{2}").unwrap(),
            email: Regex::new(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}").unwrap(),
            ssn: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            credit_card: Regex::new(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b").unwrap(),
            ipv4: Regex::new(
                r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
            )
            .unwrap(),
        }
    }

    pub fn redact_text(
        &self,
        text: &str,
        types: &[PiiEntityType],
        format: RedactFormat,
    ) -> String {
        let mut result = text.to_string();
        for &ty in types {
            let mut counter = 0u32;
            let label = entity_label(ty);
            for re in self.patterns_for(ty) {
                let r = result.clone();
                let mut out = String::new();
                let mut last = 0;
                for m in re.find_iter(&r) {
                    if ty == PiiEntityType::CreditCard && !luhn_check(m.as_str()) {
                        continue;
                    }
                    counter += 1;
                    out.push_str(&r[last..m.start()]);
                    out.push_str(&replacement(label, counter, format));
                    last = m.end();
                }
                out.push_str(&r[last..]);
                result = out;
            }
        }
        result
    }

    pub fn redact_words(
        &self,
        words: &mut [WordTimestamp],
        types: &[PiiEntityType],
        format: RedactFormat,
    ) {
        use std::collections::HashMap;
        let mut counters: HashMap<PiiEntityType, u32> = HashMap::new();
        for word in words.iter_mut() {
            'outer: for &ty in types {
                for re in self.patterns_for(ty) {
                    if re.is_match(&word.word) {
                        if ty == PiiEntityType::CreditCard && !luhn_check(&word.word) {
                            continue;
                        }
                        let c = counters.entry(ty).or_insert(0);
                        *c += 1;
                        word.word = replacement(entity_label(ty), *c, format);
                        break 'outer;
                    }
                }
            }
        }
    }

    fn patterns_for(&self, ty: PiiEntityType) -> Vec<&Regex> {
        match ty {
            PiiEntityType::Phone => vec![&self.phone_ru, &self.phone_us],
            PiiEntityType::Email => vec![&self.email],
            PiiEntityType::Ssn => vec![&self.ssn],
            PiiEntityType::CreditCard => vec![&self.credit_card],
            PiiEntityType::IpAddress => vec![&self.ipv4],
        }
    }
}

fn entity_label(ty: PiiEntityType) -> &'static str {
    match ty {
        PiiEntityType::Phone => "PHONE",
        PiiEntityType::Email => "EMAIL",
        PiiEntityType::Ssn => "SSN",
        PiiEntityType::CreditCard => "CREDIT_CARD",
        PiiEntityType::IpAddress => "IP_ADDRESS",
    }
}

fn replacement(label: &str, counter: u32, format: RedactFormat) -> String {
    match format {
        RedactFormat::Marker if counter > 0 => format!("[{label}_{counter}]"),
        RedactFormat::Marker => format!("[{label}]"),
        RedactFormat::Mask => "####".to_string(),
    }
}

pub fn luhn_check(raw: &str) -> bool {
    let digits: Vec<u32> = raw.chars().filter(|c| c.is_ascii_digit()).filter_map(|c| c.to_digit(10)).collect();
    if digits.len() < 13 {
        return false;
    }
    let mut sum = 0u32;
    for (i, &d) in digits.iter().rev().enumerate() {
        let v = if i % 2 == 1 { d * 2 } else { d };
        sum += if v > 9 { v - 9 } else { v };
    }
    sum % 10 == 0
}

pub fn parse_pii_types(input: &str) -> Vec<PiiEntityType> {
    let input = input.trim().to_lowercase();
    if input == "pii" || input == "all" {
        return ALL_TYPES.to_vec();
    }
    input
        .split(',')
        .filter_map(|s| match s.trim() {
            "phone" => Some(PiiEntityType::Phone),
            "email" => Some(PiiEntityType::Email),
            "ssn" => Some(PiiEntityType::Ssn),
            "credit_card" | "creditcard" | "cc" => Some(PiiEntityType::CreditCard),
            "ip" | "ip_address" => Some(PiiEntityType::IpAddress),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
#[path = "pii_tests.rs"]
mod tests;
