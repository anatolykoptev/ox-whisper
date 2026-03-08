use super::*;

// EN cardinals
#[test]
fn en_simple_number() {
    assert_eq!(smart_format("twenty three dogs", "en"), "23 dogs");
}

#[test]
fn en_hundred() {
    assert_eq!(smart_format("one hundred", "en"), "100");
}

#[test]
fn en_thousand() {
    assert_eq!(smart_format("three thousand five hundred", "en"), "3500");
}

#[test]
fn en_a_hundred() {
    assert_eq!(smart_format("a hundred people", "en"), "100 people");
}

#[test]
fn en_hundred_and() {
    assert_eq!(smart_format("one hundred and twenty three", "en"), "123");
}

// EN currency
#[test]
fn en_dollars() {
    assert_eq!(smart_format("five dollars", "en"), "$5");
}

#[test]
fn en_euros() {
    assert_eq!(smart_format("ten euros", "en"), "\u{20ac}10");
}

// EN percent
#[test]
fn en_percent() {
    assert_eq!(smart_format("fifty percent", "en"), "50%");
}

// EN ordinals
#[test]
fn en_first() {
    assert_eq!(smart_format("the first time", "en"), "the 1st time");
}

#[test]
fn en_second() {
    assert_eq!(smart_format("twenty second floor", "en"), "22nd floor");
}

#[test]
fn en_third() {
    assert_eq!(smart_format("the third place", "en"), "the 3rd place");
}

// RU cardinals
#[test]
fn ru_simple() {
    assert_eq!(smart_format("двадцать три собаки", "ru"), "23 собаки");
}

#[test]
fn ru_hundred() {
    assert_eq!(smart_format("двести", "ru"), "200");
}

#[test]
fn ru_thousand() {
    assert_eq!(smart_format("две тысячи пятьсот", "ru"), "2500");
}

// RU currency
#[test]
fn ru_rubles() {
    assert_eq!(smart_format("сто рублей", "ru"), "100 руб.");
}

#[test]
fn ru_dollars() {
    assert_eq!(smart_format("пять долларов", "ru"), "$5");
}

// RU percent
#[test]
fn ru_percent() {
    assert_eq!(smart_format("пятьдесят процентов", "ru"), "50%");
}

// No change
#[test]
fn no_change() {
    assert_eq!(smart_format("hello world", "en"), "hello world");
}

// Mixed
#[test]
fn mixed() {
    assert_eq!(
        smart_format("I have twenty three cats and five dogs", "en"),
        "I have 23 cats and 5 dogs",
    );
}

#[test]
fn en_million() {
    assert_eq!(smart_format("two million", "en"), "2000000");
}

#[test]
fn ru_million() {
    assert_eq!(smart_format("три миллиона рублей", "ru"), "3000000 руб.");
}
