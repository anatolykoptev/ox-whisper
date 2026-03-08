use serde::{Deserialize, Serialize};

use crate::words::WordTimestamp;

/// Query parameters for the WebSocket `/v1/listen` endpoint.
#[derive(Debug, Deserialize)]
pub struct WsParams {
    #[serde(default = "default_lang")]
    pub language: String,
    #[serde(default = "default_true")]
    pub vad: bool,
    #[serde(default)]
    pub interim_results: bool,
    #[serde(default)]
    pub smart_format: bool,
    #[serde(default = "default_true")]
    pub punctuate: bool,
    #[serde(default = "default_encoding")]
    pub encoding: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
}

fn default_lang() -> String { "en".to_string() }
fn default_true() -> bool { true }
fn default_encoding() -> String { "pcm_s16le".to_string() }
fn default_sample_rate() -> u32 { 16000 }

/// Control messages sent by the client as JSON text frames.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    Finalize,
    CloseStream,
    KeepAlive,
}

/// Messages sent from server to client as JSON text frames.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    Metadata {
        request_id: String,
        model: String,
        channels: u8,
    },
    Results {
        is_final: bool,
        speech_final: bool,
        from_finalize: bool,
        channel: Channel,
        #[serde(skip_serializing_if = "Option::is_none")]
        speech_started_s: Option<f64>,
    },
    SpeechStarted {
        timestamp_s: f64,
    },
    Error {
        message: String,
    },
    CloseStream,
}

#[derive(Debug, Serialize)]
pub struct Channel {
    pub alternatives: Vec<Alternative>,
}

#[derive(Debug, Serialize)]
pub struct Alternative {
    pub transcript: String,
    pub confidence: f32,
    pub words: Vec<WordTimestamp>,
}
