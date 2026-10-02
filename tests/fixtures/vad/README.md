# VAD test fixtures

Used by `src/vad.rs` tests.

- `silero_vad.onnx` — Silero VAD (MIT, https://github.com/snakers4/silero-vad), the
  sherpa-onnx release asset `asr-models/silero_vad.onnx`, the same file the production
  container mounts (sha256 `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6`).
- `fleurs_en_a.wav`, `fleurs_en_b.wav` — two English test utterances from Google's FLEURS
  dataset (CC BY 4.0, https://huggingface.co/datasets/google/fleurs), converted to 16 kHz mono
  `pcm_s16le`. Source files `1218355507714944126.wav` and `12356293452530186109.wav`.
  Feeding `a` and then `b` through one detector without a reset produced zero speech segments
  for `b`.
