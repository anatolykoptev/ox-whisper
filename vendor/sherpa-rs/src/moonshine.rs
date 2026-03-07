use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::ffi::CString;
use std::mem;

#[derive(Debug)]
pub struct MoonshineRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type MoonshineRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct MoonshineConfig {
    pub preprocessor: String,

    pub encoder: String,
    pub uncached_decoder: String,
    pub cached_decoder: String,
    pub merged_decoder: String,

    pub tokens: String,

    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for MoonshineConfig {
    fn default() -> Self {
        Self {
            preprocessor: String::new(),
            encoder: String::new(),
            cached_decoder: String::new(),
            uncached_decoder: String::new(),
            merged_decoder: String::new(),
            tokens: String::new(),

            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

fn optional_cstr(s: &str) -> Option<CString> {
    if s.is_empty() {
        None
    } else {
        Some(cstring_from_str(s))
    }
}

fn cstr_ptr(opt: &Option<CString>) -> *const std::os::raw::c_char {
    match opt {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

impl MoonshineRecognizer {
    pub fn new(config: MoonshineConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(2);

        // Only create CStrings for non-empty fields; empty = NULL
        let preprocessor_ptr = optional_cstr(&config.preprocessor);
        let encoder_ptr = optional_cstr(&config.encoder);
        let cached_decoder_ptr = optional_cstr(&config.cached_decoder);
        let uncached_decoder_ptr = optional_cstr(&config.uncached_decoder);
        let merged_decoder_ptr = optional_cstr(&config.merged_decoder);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let model_config = unsafe {
            let mut c: sherpa_rs_sys::SherpaOnnxOfflineModelConfig = mem::zeroed();
            c.debug = debug;
            c.num_threads = num_threads;
            c.moonshine.preprocessor = cstr_ptr(&preprocessor_ptr);
            c.moonshine.encoder = cstr_ptr(&encoder_ptr);
            c.moonshine.uncached_decoder = cstr_ptr(&uncached_decoder_ptr);
            c.moonshine.cached_decoder = cstr_ptr(&cached_decoder_ptr);
            c.moonshine.merged_decoder = cstr_ptr(&merged_decoder_ptr);
            c.tokens = tokens_ptr.as_ptr();
            c.provider = provider_ptr.as_ptr();
            c
        };

        let config = unsafe {
            let mut c: sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig = mem::zeroed();
            c.decoding_method = decoding_method_ptr.as_ptr();
            c.feat_config.sample_rate = 16000;
            c.feat_config.feature_dim = 80;
            c.model_config = model_config;
            c
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> MoonshineRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let result = MoonshineRecognizerResult::new(&raw_result);
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for MoonshineRecognizer {}
unsafe impl Sync for MoonshineRecognizer {}

impl Drop for MoonshineRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
