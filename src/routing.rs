//! Which recognizer serves a request language.
//!
//! Every transcription path — the batch HTTP endpoints, SSE streaming and the
//! WebSocket session — asks [`Engine::route`], so the language → model decision
//! and the punctuation decision live in one place.

/// The recognizer a request is sent to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Parakeet TDT v3: multilingual, writes case and punctuation itself.
    Parakeet,
    /// The `ru` pool: Zipformer transducer or GigaAM.
    Ru,
    /// Moonshine, the route for every other language.
    Moonshine,
}

impl Engine {
    /// Routes `language` to Parakeet when it is loaded and the language is in
    /// `PARAKEET_LANGS`; otherwise `ru` goes to the RU pool and everything else
    /// to Moonshine. A missing Parakeet model therefore falls back to the old
    /// routes instead of failing the request.
    pub fn route(language: &str, parakeet_loaded: bool, parakeet_langs: &[String]) -> Self {
        if parakeet_loaded && parakeet_langs.iter().any(|l| l == language) {
            Self::Parakeet
        } else if language == "ru" {
            Self::Ru
        } else {
            Self::Moonshine
        }
    }

    /// Metric label for the pool behind this engine.
    pub fn label(self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet",
            Self::Ru => "ru",
            Self::Moonshine => "en",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_parakeet_langs;

    fn default_langs() -> Vec<String> {
        parse_parakeet_langs(None)
    }

    #[test]
    fn parakeet_serves_ru_en_and_european_languages() {
        let langs = default_langs();
        for lang in ["ru", "en", "uk", "es", "de", "fr", "pl"] {
            assert_eq!(
                Engine::route(lang, true, &langs),
                Engine::Parakeet,
                "{lang}"
            );
        }
    }

    #[test]
    fn languages_parakeet_does_not_cover_stay_on_moonshine() {
        let langs = default_langs();
        for lang in ["zh", "ar", "vi", "ja"] {
            assert_eq!(
                Engine::route(lang, true, &langs),
                Engine::Moonshine,
                "{lang}"
            );
        }
    }

    #[test]
    fn missing_parakeet_model_falls_back_to_old_routes() {
        let langs = default_langs();
        assert_eq!(Engine::route("ru", false, &langs), Engine::Ru);
        assert_eq!(Engine::route("en", false, &langs), Engine::Moonshine);
        assert_eq!(Engine::route("uk", false, &langs), Engine::Moonshine);
    }

    #[test]
    fn rollback_switch_disables_parakeet_for_every_language() {
        for raw in ["", "off", " OFF ", "none"] {
            let langs = parse_parakeet_langs(Some(raw));
            assert!(langs.is_empty(), "{raw:?} must disable Parakeet");
            assert_eq!(Engine::route("ru", true, &langs), Engine::Ru, "{raw:?}");
            assert_eq!(
                Engine::route("en", true, &langs),
                Engine::Moonshine,
                "{raw:?}"
            );
        }
    }

    #[test]
    fn parakeet_langs_can_be_narrowed() {
        let langs = parse_parakeet_langs(Some("ru, EN,xx,ru"));
        assert_eq!(langs, vec!["ru".to_string(), "en".to_string()]);
        assert_eq!(Engine::route("ru", true, &langs), Engine::Parakeet);
        assert_eq!(Engine::route("uk", true, &langs), Engine::Moonshine);
    }
}
