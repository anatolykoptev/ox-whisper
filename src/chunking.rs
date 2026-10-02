/// Sanitize transcription output: remove null bytes, collapse whitespace,
/// strip leading/trailing punctuation artifacts from model output.
pub fn sanitize_utf8(text: &str) -> String {
    let mut result = text.replace('\0', "");
    // Collapse multiple spaces into one
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }
    // Remove leading punctuation (common model artifact)
    result = result
        .trim_start_matches(|c: char| c == ',' || c == '.' || c == ' ')
        .to_string();
    result = result.trim().to_string();
    result
}

/// Samples per second of the 16 kHz audio the decoder takes.
const SR: usize = 16_000;
/// Energy is measured per 10 ms hop.
const HOP: usize = SR / 100;
/// A cut point is the middle of the quietest stretch of this many hops (150 ms).
/// A stretch, not a single frame: the closure of a stop consonant is quiet for a
/// frame or two and would cut a word in half.
const SPAN: usize = 15;

/// Chunk boundaries as sample offsets, starting with `0` and ending with
/// `x.len()`: windows of at most `max_len` samples, each cut in the middle of
/// the quietest 150 ms stretch in its last fifth.
///
/// Port of `chunk_bounds` in ox-say's `ox-stt.cpp`. Consecutive boundaries are
/// strictly increasing, so every input sample lies in exactly one chunk and the
/// chunks concatenate back to the input. `max_len == 0` means "no limit".
pub fn chunk_bounds(x: &[f32], max_len: usize) -> Vec<usize> {
    let mut b = vec![0usize];
    if max_len == 0 {
        b.push(x.len());
        return b;
    }
    while x.len() - b[b.len() - 1] > max_len {
        let start = b[b.len() - 1];
        let lo = start + max_len * 4 / 5;
        let hi = start + max_len;
        let energy: Vec<f64> = x[lo..hi]
            .chunks_exact(HOP)
            .map(|h| h.iter().map(|&v| f64::from(v) * f64::from(v)).sum())
            .collect();
        let mut best = hi.saturating_sub(HOP);
        if energy.len() >= SPAN {
            let mut sum: f64 = energy[..SPAN].iter().sum();
            let mut best_sum = sum;
            let mut best_k = 0;
            for k in SPAN..energy.len() {
                sum += energy[k] - energy[k - SPAN];
                if sum < best_sum {
                    best_sum = sum;
                    best_k = k - SPAN + 1;
                }
            }
            best = lo + (best_k + SPAN / 2) * HOP;
        }
        // `best` is the start of the middle hop of the stretch (half a hop before
        // its exact centre), as in ox-say.
        // A window shorter than one hop has no room to search: still advance.
        b.push(best.clamp(start + 1, hi));
    }
    b.push(x.len());
    b
}

/// Splits `samples` into the chunks [`chunk_bounds`] describes. Nothing is
/// dropped and nothing is inserted: no VAD trimming, no zero padding.
pub fn split_at_quiet(samples: Vec<f32>, max_len: usize) -> Vec<Vec<f32>> {
    let b = chunk_bounds(&samples, max_len);
    if b.len() == 2 {
        return vec![samples];
    }
    b.windows(2).map(|w| samples[w[0]..w[1]].to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_utf8_valid() {
        assert_eq!(sanitize_utf8("hello"), "hello");
    }

    #[test]
    fn test_sanitize_utf8_null_bytes() {
        assert_eq!(sanitize_utf8("hel\0lo"), "hello");
    }

    #[test]
    fn test_sanitize_utf8_cyrillic() {
        assert_eq!(sanitize_utf8("Привет мир"), "Привет мир");
    }

    #[test]
    fn test_sanitize_utf8_emoji() {
        assert_eq!(sanitize_utf8("Hello 🌍!"), "Hello 🌍!");
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;

    const WIN: usize = 30 * SR;

    /// Loud pseudo-speech with a deterministic, never-quiet waveform.
    fn loud(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 * ((i as f32) * 0.37).sin()).collect()
    }

    fn quiet_at(x: &mut [f32], start: usize, len: usize) {
        x[start..start + len].fill(0.0001);
    }

    fn rejoin(chunks: &[Vec<f32>]) -> Vec<f32> {
        chunks.iter().flatten().copied().collect()
    }

    #[test]
    fn chunks_concatenate_to_the_exact_input() {
        let mut x = loud(WIN * 2 + 12_345);
        quiet_at(&mut x, WIN * 9 / 10, 4000);
        let chunks = split_at_quiet(x.clone(), WIN);
        assert!(chunks.len() >= 3);
        let back = rejoin(&chunks);
        assert_eq!(back.len(), x.len());
        assert!(back.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn no_chunk_exceeds_the_window_and_none_is_empty() {
        let x = loud(WIN * 4 + 777);
        for c in split_at_quiet(x, WIN) {
            assert!(c.len() <= WIN, "{}", c.len());
            assert!(!c.is_empty());
        }
    }

    /// The cut lands at the centre of the quietest 150 ms stretch of the last
    /// fifth, not at the hard window end and not on an earlier quiet stretch.
    #[test]
    fn cut_is_the_centre_of_the_quietest_stretch_in_the_last_fifth() {
        let mut x = loud(WIN + 50_000);
        let stretch = SPAN * HOP;
        let at = WIN * 9 / 10 + 7 * HOP; // inside the last fifth, hop aligned
        let at = (at / HOP) * HOP;
        quiet_at(&mut x, at, stretch);
        // A quieter stretch before the last fifth must be ignored.
        x[WIN / 2..WIN / 2 + stretch].fill(0.0);
        let b = chunk_bounds(&x, WIN);
        assert_eq!(b, vec![0, at + (SPAN / 2) * HOP, x.len()]);
    }

    #[test]
    fn short_input_is_one_chunk() {
        for n in [0, 1, WIN - 1, WIN] {
            let chunks = split_at_quiet(loud(n), WIN);
            assert_eq!(chunks.len(), 1, "n={n}");
            assert_eq!(chunks[0].len(), n);
        }
    }

    /// Digital silence: every stretch ties. The cut must still advance and the
    /// split must terminate with all samples kept.
    #[test]
    fn a_window_with_no_quiet_stretch_still_makes_progress() {
        for x in [vec![0.0f32; WIN * 3 + 5], vec![0.25f32; WIN * 3 + 5]] {
            let b = chunk_bounds(&x, WIN);
            assert!(b.windows(2).all(|w| w[0] < w[1]), "{b:?}");
            assert_eq!(*b.last().unwrap(), x.len());
        }
    }

    #[test]
    fn a_window_shorter_than_one_hop_still_terminates() {
        let x = loud(1000);
        let b = chunk_bounds(&x, 3);
        assert!(b.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(rejoin(&split_at_quiet(x.clone(), 3)).len(), x.len());
    }

    #[test]
    fn a_zero_window_means_no_limit() {
        assert_eq!(chunk_bounds(&loud(5000), 0), vec![0, 5000]);
    }

    /// The tail after the last cut is kept, not dropped.
    #[test]
    fn the_tail_remainder_is_kept() {
        let x = loud(WIN + 100);
        let chunks = split_at_quiet(x.clone(), WIN);
        assert_eq!(chunks.len(), 2);
        assert_eq!(rejoin(&chunks), x);
        assert!(chunks[1].len() >= 100);
    }
}
