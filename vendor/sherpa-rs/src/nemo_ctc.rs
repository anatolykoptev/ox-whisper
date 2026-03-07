use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

#[derive(Debug)]
pub struct NemoCtcRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type NemoCtcRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct NemoCtcConfig {
    pub model: String,
    pub tokens: String,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for NemoCtcConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            tokens: String::new(),
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl NemoCtcRecognizer {
    pub fn new(config: NemoCtcConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(2);
        let model_ptr = cstring_from_str(&config.model);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let model_config = unsafe {
            let mut c: sherpa_rs_sys::SherpaOnnxOfflineModelConfig = mem::zeroed();
            c.nemo_ctc.model = model_ptr.as_ptr();
            c.tokens = tokens_ptr.as_ptr();
            c.num_threads = num_threads;
            c.debug = debug;
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
            bail!("Failed to create NeMo CTC recognizer");
        }
        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> NemoCtcRecognizerResult {
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
            let result = NemoCtcRecognizerResult::new(&raw_result);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }

    /// Batch-decode multiple audio chunks in parallel via SherpaOnnxDecodeMultipleOfflineStreams.
    pub fn transcribe_batch(
        &mut self, sample_rate: u32, chunks: &[&[f32]],
    ) -> Vec<NemoCtcRecognizerResult> {
        if chunks.is_empty() {
            return Vec::new();
        }
        unsafe {
            let mut streams = Vec::with_capacity(chunks.len());
            for samples in chunks {
                let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
                sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                    stream, sample_rate as i32, samples.as_ptr(),
                    samples.len().try_into().unwrap(),
                );
                streams.push(stream);
            }
            let mut stream_ptrs: Vec<_> = streams.iter().map(|s| *s as *const _).collect();
            sherpa_rs_sys::SherpaOnnxDecodeMultipleOfflineStreams(
                self.recognizer, stream_ptrs.as_mut_ptr(), streams.len() as i32,
            );
            let mut results = Vec::with_capacity(streams.len());
            for stream in &streams {
                let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(*stream);
                let raw = result_ptr.read();
                results.push(NemoCtcRecognizerResult::new(&raw));
                sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            }
            for stream in streams {
                sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            }
            results
        }
    }
}

unsafe impl Send for NemoCtcRecognizer {}
unsafe impl Sync for NemoCtcRecognizer {}

impl Drop for NemoCtcRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
