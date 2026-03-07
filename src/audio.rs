use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("WAV file too short (need at least 44 bytes, got {0})")]
    TooShort(usize),
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("ffmpeg failed: {0}")]
    FfmpegFailed(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct WavHeader {
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
}

pub fn parse_wav_header(data: &[u8]) -> Result<WavHeader, AudioError> {
    if data.len() < 44 {
        return Err(AudioError::TooShort(data.len()));
    }
    Ok(WavHeader {
        channels: u16::from_le_bytes([data[22], data[23]]),
        sample_rate: u32::from_le_bytes([data[24], data[25], data[26], data[27]]),
        bits_per_sample: u16::from_le_bytes([data[34], data[35]]),
    })
}

pub fn pcm_to_f32(data: &[u8], header: &WavHeader) -> Result<Vec<f32>, AudioError> {
    if header.bits_per_sample != 16 {
        return Err(AudioError::UnsupportedFormat(format!(
            "{}-bit audio not supported",
            header.bits_per_sample
        )));
    }
    if header.channels > 2 {
        return Err(AudioError::UnsupportedFormat(format!(
            "{}-channel audio not supported",
            header.channels
        )));
    }
    let pcm = &data[44..];
    let bytes_per_frame = (header.channels * 2) as usize;
    let frame_count = pcm.len() / bytes_per_frame;
    let mut samples = Vec::with_capacity(frame_count);

    for i in 0..frame_count {
        let offset = i * bytes_per_frame;
        let left = i16::from_le_bytes([pcm[offset], pcm[offset + 1]]);
        if header.channels == 1 {
            samples.push(left as f32 / 32768.0);
        } else {
            let right = i16::from_le_bytes([pcm[offset + 2], pcm[offset + 3]]);
            samples.push((left as f32 + right as f32) / 2.0 / 32768.0);
        }
    }
    Ok(samples)
}

pub fn ensure_wav(path: &Path) -> Result<(PathBuf, bool), AudioError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("wav") {
        return Ok((path.to_path_buf(), false));
    }
    let out = std::env::temp_dir().join(format!("{}.wav", uuid::Uuid::new_v4()));
    let result = std::process::Command::new("ffmpeg")
        .args(["-i"])
        .arg(path)
        .args(["-ar", "16000", "-ac", "1", "-f", "wav"])
        .arg(&out)
        .output()?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(AudioError::FfmpegFailed(stderr.into_owned()));
    }
    Ok((out, true))
}

pub fn load_wav(path: &Path) -> Result<(Vec<f32>, f64), AudioError> {
    let data = std::fs::read(path)?;
    let header = parse_wav_header(&data)?;
    let samples = pcm_to_f32(&data, &header)?;
    let duration = samples.len() as f64 / header.sample_rate as f64;
    Ok((samples, duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wav_header(channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
        let mut h = vec![0u8; 44];
        h[0..4].copy_from_slice(b"RIFF");
        h[8..12].copy_from_slice(b"WAVE");
        h[22..24].copy_from_slice(&channels.to_le_bytes());
        h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
        h[34..36].copy_from_slice(&bits.to_le_bytes());
        h
    }

    #[test]
    fn test_parse_wav_header() {
        let data = make_wav_header(1, 16000, 16);
        let hdr = parse_wav_header(&data).unwrap();
        assert_eq!(hdr.channels, 1);
        assert_eq!(hdr.sample_rate, 16000);
        assert_eq!(hdr.bits_per_sample, 16);
    }

    #[test]
    fn test_pcm_to_f32_mono() {
        let header = WavHeader { channels: 1, sample_rate: 16000, bits_per_sample: 16 };
        let mut data = make_wav_header(1, 16000, 16);
        data.extend_from_slice(&0i16.to_le_bytes());
        data.extend_from_slice(&16384i16.to_le_bytes());
        data.extend_from_slice(&(-16384i16).to_le_bytes());
        let samples = pcm_to_f32(&data, &header).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[0]).abs() < 1e-6);
        assert!((samples[1] - 0.5).abs() < 1e-4);
        assert!((samples[2] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_pcm_to_f32_stereo() {
        let header = WavHeader { channels: 2, sample_rate: 16000, bits_per_sample: 16 };
        let mut data = make_wav_header(2, 16000, 16);
        // left=16384, right=0 => avg=8192 => 8192/32768 = 0.25
        data.extend_from_slice(&16384i16.to_le_bytes());
        data.extend_from_slice(&0i16.to_le_bytes());
        let samples = pcm_to_f32(&data, &header).unwrap();
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_pcm_to_f32_unsupported_8bit() {
        let header = WavHeader { channels: 1, sample_rate: 16000, bits_per_sample: 8 };
        let data = make_wav_header(1, 16000, 8);
        let err = pcm_to_f32(&data, &header).unwrap_err();
        assert!(err.to_string().contains("8-bit"));
    }

    #[test]
    fn test_pcm_to_f32_unsupported_3ch() {
        let header = WavHeader { channels: 3, sample_rate: 16000, bits_per_sample: 16 };
        let data = make_wav_header(3, 16000, 16);
        let err = pcm_to_f32(&data, &header).unwrap_err();
        assert!(err.to_string().contains("3-channel"));
    }

    #[test]
    fn test_ensure_wav_passthrough() {
        let path = Path::new("/tmp/test.wav");
        let (result, cleanup) = ensure_wav(path).unwrap();
        assert_eq!(result, path);
        assert!(!cleanup);
    }
}
