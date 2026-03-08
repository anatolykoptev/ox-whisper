/// Smart formatting: convert spoken numbers, currency, percentages to written form.

mod ru;

pub fn smart_format(text: &str, language: &str) -> String {
    match language {
        "ru" => ru::format_ru(text),
        _ => format_en(text),
    }
}

// --- English ---

fn number_value_en(word: &str) -> Option<i64> {
    match word {
        "zero" => Some(0), "one" => Some(1), "two" => Some(2), "three" => Some(3),
        "four" => Some(4), "five" => Some(5), "six" => Some(6), "seven" => Some(7),
        "eight" => Some(8), "nine" => Some(9), "ten" => Some(10), "eleven" => Some(11),
        "twelve" => Some(12), "thirteen" => Some(13), "fourteen" => Some(14),
        "fifteen" => Some(15), "sixteen" => Some(16), "seventeen" => Some(17),
        "eighteen" => Some(18), "nineteen" => Some(19),
        "twenty" => Some(20), "thirty" => Some(30), "forty" => Some(40),
        "fifty" => Some(50), "sixty" => Some(60), "seventy" => Some(70),
        "eighty" => Some(80), "ninety" => Some(90),
        _ => None,
    }
}

fn scale_value_en(word: &str) -> Option<i64> {
    match word {
        "hundred" => Some(100),
        "thousand" => Some(1_000),
        "million" => Some(1_000_000),
        "billion" => Some(1_000_000_000),
        _ => None,
    }
}

fn is_skip_en(word: &str) -> bool {
    matches!(word, "and" | "a")
}

/// Parse consecutive number words into a number. Returns (value, words_consumed).
fn words_to_number_en(words: &[&str], start: usize) -> Option<(i64, usize)> {
    let mut result: i64 = 0;
    let mut current: i64 = 0;
    let mut consumed = 0;
    let mut has_number = false;
    let mut i = start;

    while i < words.len() {
        let w = words[i].to_lowercase();
        if w == "a" && !has_number && i + 1 < words.len() {
            if scale_value_en(&words[i + 1].to_lowercase()).is_some() {
                i += 1;
                consumed += 1;
                continue;
            }
            break;
        }
        if is_skip_en(&w) && has_number && i + 1 < words.len() {
            let next = words[i + 1].to_lowercase();
            if number_value_en(&next).is_some() || scale_value_en(&next).is_some() {
                i += 1;
                consumed += 1;
                continue;
            }
            break;
        }
        if let Some(v) = number_value_en(&w) {
            current += v;
            has_number = true;
            consumed += 1;
            i += 1;
        } else if let Some(s) = scale_value_en(&w) {
            if !has_number && s == 100 {
                current = 100;
            } else if s == 100 {
                current *= 100;
            } else {
                if current == 0 { current = 1; }
                result += current * s;
                current = 0;
            }
            has_number = true;
            consumed += 1;
            i += 1;
        } else {
            break;
        }
    }
    if has_number { Some((result + current, consumed)) } else { None }
}

fn ordinal_info(word: &str) -> Option<(i64, &'static str)> {
    match word {
        "first" => Some((1, "st")),
        "second" => Some((2, "nd")),
        "third" => Some((3, "rd")),
        _ => None,
    }
}

fn format_number_suffix(n: i64, next: Option<&str>) -> (String, bool) {
    let word = next.map(|w| w.to_lowercase());
    match word.as_deref() {
        Some("dollar" | "dollars") => (format!("${n}"), true),
        Some("euro" | "euros") => (format!("\u{20ac}{n}"), true),
        Some("percent") => (format!("{n}%"), true),
        _ => (n.to_string(), false),
    }
}

fn format_en(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;

    while i < words.len() {
        let lower = words[i].to_lowercase();
        if let Some((val, suffix)) = ordinal_info(&lower) {
            let prev_is_num = i > 0 && number_value_en(&words[i - 1].to_lowercase()).is_some();
            if !prev_is_num {
                out.push(format!("{val}{suffix}"));
                i += 1;
                continue;
            }
        }
        if number_value_en(&lower).is_some() || scale_value_en(&lower).is_some()
            || (lower == "a" && i + 1 < words.len()
                && scale_value_en(&words[i + 1].to_lowercase()).is_some())
        {
            if let Some((num, consumed)) = words_to_number_en(&words, i) {
                let next_idx = i + consumed;
                let next_word = words.get(next_idx).copied();
                if let Some(nw) = next_word {
                    if let Some((val, suffix)) = ordinal_info(&nw.to_lowercase()) {
                        out.push(format!("{}{suffix}", num + val));
                        i = next_idx + 1;
                        continue;
                    }
                }
                let (formatted, consumed_next) = format_number_suffix(num, next_word);
                out.push(formatted);
                i = next_idx + if consumed_next { 1 } else { 0 };
                continue;
            }
        }
        out.push(words[i].to_string());
        i += 1;
    }
    out.join(" ")
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
