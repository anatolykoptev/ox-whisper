/// Russian number formatting: spoken numbers → digits with currency/percent.

fn number_value_ru(word: &str) -> Option<i64> {
    match word {
        "ноль" => Some(0), "один" | "одна" | "одно" => Some(1),
        "два" | "две" => Some(2), "три" => Some(3), "четыре" => Some(4),
        "пять" => Some(5), "шесть" => Some(6), "семь" => Some(7),
        "восемь" => Some(8), "девять" => Some(9), "десять" => Some(10),
        "одиннадцать" => Some(11), "двенадцать" => Some(12),
        "тринадцать" => Some(13), "четырнадцать" => Some(14),
        "пятнадцать" => Some(15), "шестнадцать" => Some(16),
        "семнадцать" => Some(17), "восемнадцать" => Some(18),
        "девятнадцать" => Some(19),
        "двадцать" => Some(20), "тридцать" => Some(30), "сорок" => Some(40),
        "пятьдесят" => Some(50), "шестьдесят" => Some(60),
        "семьдесят" => Some(70), "восемьдесят" => Some(80), "девяносто" => Some(90),
        "сто" => Some(100), "двести" => Some(200), "триста" => Some(300),
        "четыреста" => Some(400), "пятьсот" => Some(500), "шестьсот" => Some(600),
        "семьсот" => Some(700), "восемьсот" => Some(800), "девятьсот" => Some(900),
        _ => None,
    }
}

fn scale_value_ru(word: &str) -> Option<i64> {
    match word {
        "тысяча" | "тысячи" | "тысяч" => Some(1_000),
        "миллион" | "миллиона" | "миллионов" => Some(1_000_000),
        _ => None,
    }
}

fn words_to_number_ru(words: &[&str], start: usize) -> Option<(i64, usize)> {
    let mut result: i64 = 0;
    let mut current: i64 = 0;
    let mut consumed = 0;
    let mut has_number = false;
    let mut i = start;

    while i < words.len() {
        let w = words[i].to_lowercase();
        if let Some(v) = number_value_ru(&w) {
            current += v;
            has_number = true;
            consumed += 1;
            i += 1;
        } else if let Some(s) = scale_value_ru(&w) {
            if current == 0 { current = 1; }
            result += current * s;
            current = 0;
            has_number = true;
            consumed += 1;
            i += 1;
        } else {
            break;
        }
    }
    if has_number { Some((result + current, consumed)) } else { None }
}

fn format_number_suffix_ru(n: i64, next: Option<&str>) -> (String, bool) {
    let word = next.map(|w| w.to_lowercase());
    match word.as_deref() {
        Some("рубль" | "рубля" | "рублей") => (format!("{n} руб."), true),
        Some("доллар" | "доллара" | "долларов") => (format!("${n}"), true),
        Some("евро") => (format!("\u{20ac}{n}"), true),
        Some("процент" | "процента" | "процентов") => (format!("{n}%"), true),
        _ => (n.to_string(), false),
    }
}

pub fn format_ru(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;

    while i < words.len() {
        let lower = words[i].to_lowercase();
        if number_value_ru(&lower).is_some() || scale_value_ru(&lower).is_some() {
            if let Some((num, consumed)) = words_to_number_ru(&words, i) {
                let next_idx = i + consumed;
                let next_word = words.get(next_idx).copied();
                let (formatted, consumed_next) = format_number_suffix_ru(num, next_word);
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
