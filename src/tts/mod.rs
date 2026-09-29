//! Text-to-speech support: supervision of the external `tts-server` child.

// Dead in the bin target until the /v1/audio/speech proxy lands —
// ensure_ready/guard are only exercised by tests so far.
#[cfg_attr(not(test), allow(dead_code))]
mod supervisor;

// TtsError/TtsGuard are the API surface the /v1/audio/speech proxy task
// consumes; not referenced yet outside tests.
#[allow(unused_imports)]
pub use supervisor::{TtsError, TtsGuard, TtsState, TtsSupervisor};

#[cfg(test)]
mod tests;
