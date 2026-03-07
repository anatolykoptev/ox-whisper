use eyre::{bail, Result};

use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
};

#[derive(Debug, Default, Clone)]
pub struct OnlinePunctuationConfig {
    pub cnn_bilstm: String,
    pub bpe_vocab: String,
    pub debug: bool,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
}

pub struct OnlinePunctuation {
    punct: *const sherpa_rs_sys::SherpaOnnxOnlinePunctuation,
}

impl OnlinePunctuation {
    pub fn new(config: OnlinePunctuationConfig) -> Result<Self> {
        let cnn_bilstm = cstring_from_str(&config.cnn_bilstm);
        let bpe_vocab = cstring_from_str(&config.bpe_vocab);
        let provider = cstring_from_str(&config.provider.unwrap_or(get_default_provider()));

        let sherpa_config = sherpa_rs_sys::SherpaOnnxOnlinePunctuationConfig {
            model: sherpa_rs_sys::SherpaOnnxOnlinePunctuationModelConfig {
                cnn_bilstm: cnn_bilstm.as_ptr(),
                bpe_vocab: bpe_vocab.as_ptr(),
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.into(),
                provider: provider.as_ptr(),
            },
        };

        let punct =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlinePunctuation(&sherpa_config) };

        if punct.is_null() {
            bail!("Failed to create online punctuation");
        }
        Ok(Self { punct })
    }

    pub fn add_punctuation(&self, text: &str) -> String {
        let text = cstring_from_str(text);
        unsafe {
            let result = sherpa_rs_sys::SherpaOnnxOnlinePunctuationAddPunct(
                self.punct,
                text.as_ptr(),
            );
            let punctuated = cstr_to_string(result as _);
            sherpa_rs_sys::SherpaOnnxOnlinePunctuationFreeText(result);
            punctuated
        }
    }
}

unsafe impl Send for OnlinePunctuation {}
unsafe impl Sync for OnlinePunctuation {}

impl Drop for OnlinePunctuation {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlinePunctuation(self.punct);
        }
    }
}
