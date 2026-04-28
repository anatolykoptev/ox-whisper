# Changelog

## [0.6.0](https://github.com/anatolykoptev/ox-whisper/compare/ox-whisper-v0.5.0...ox-whisper-v0.6.0) (2026-04-28)


### Features

* add audio processing (WAV parse, PCM→f32, ffmpeg) ([ef3c21c](https://github.com/anatolykoptev/ox-whisper/commit/ef3c21ccc150062e990dbdb23bb88fb01a830607))
* add auto-paragraphs by pause detection ([314f56f](https://github.com/anatolykoptev/ox-whisper/commit/314f56fff488753fa2161eeaf43a038fef5f7682))
* add custom_spelling post-processing for domain terminology ([4982e59](https://github.com/anatolykoptev/ox-whisper/commit/4982e59959b29233aa9fd7431203c6c404a42f40))
* add Dockerfile with multi-stage build ([165a334](https://github.com/anatolykoptev/ox-whisper/commit/165a334aef62aa0d451c1686ff8bbc85b3e9ef00))
* add extra passthrough for request metadata ([263432d](https://github.com/anatolykoptev/ox-whisper/commit/263432dc6e2d54b49078f4be9789d3f9125a88c6))
* add GigaAM v2 CTC (NeMo) as primary RU model ([a623036](https://github.com/anatolykoptev/ox-whisper/commit/a623036f96bfe75ddc272713c41388bbe9bb94b2))
* add keyword boosting via fuzzy Levenshtein matching ([1697eea](https://github.com/anatolykoptev/ox-whisper/commit/1697eea8f49ab165dd1baccc70244adda37b065e))
* add language_confidence to detection and verbose_json ([b487d69](https://github.com/anatolykoptev/ox-whisper/commit/b487d69ed52b57b6de84d31c1467b1653f01e34f))
* add model loading, VAD wrapper, and punctuation ([9540e00](https://github.com/anatolykoptev/ox-whisper/commit/9540e005c4b01d59564ff23e3cdb712e6b7c883e))
* add OpenAI-compatible /v1/audio/transcriptions API (v0.2.0) ([8430310](https://github.com/anatolykoptev/ox-whisper/commit/84303105cc6c676ccd7d82967e70dd53e2f0dfa6))
* add PII redaction (regex Phase 1) — phone, email, SSN, CC, IP ([c30c80b](https://github.com/anatolykoptev/ox-whisper/commit/c30c80b01ae9d8851576123f772902cf74bd7b83))
* add recognizer pool for concurrent transcription (POOL_SIZE env var) ([68b38ba](https://github.com/anatolykoptev/ox-whisper/commit/68b38ba4ae64f0109ed391fd8d18221052e9eac3))
* add smart formatting / ITN — numbers, currency, percent (EN+RU) ([56e06c2](https://github.com/anatolykoptev/ox-whisper/commit/56e06c2f886f392ba42e2e3ebb11aa8d1959faab))
* add speaker diarization via sherpa-onnx pyannote+ERes2Net ([45f0ffb](https://github.com/anatolykoptev/ox-whisper/commit/45f0ffb0f3b67b349377e7fcd25d38726ea46718))
* add SSE streaming endpoint /transcribe/stream ([588f1b3](https://github.com/anatolykoptev/ox-whisper/commit/588f1b3e9328af45594959b2b760e4c8aa039dea))
* add text chunking with split priority ([4e04a47](https://github.com/anatolykoptev/ox-whisper/commit/4e04a470ce98bc199c21b85a7c2731b5187d3ac8))
* add transcription pipeline and HTTP handlers ([2414e10](https://github.com/anatolykoptev/ox-whisper/commit/2414e106f6fd0ddea58145a5bc208c955ea889db))
* add WebSocket streaming endpoint at /v1/listen ([3dd2f5e](https://github.com/anatolykoptev/ox-whisper/commit/3dd2f5eaae2684ae34520e5160a5be104e76f37b))
* add word timestamps, configurable VAD, and hallucination retry ([059d600](https://github.com/anatolykoptev/ox-whisper/commit/059d60047cc100fde09f5f13f7f840155941019d))
* auto-detect NeMo transducer models, add experiments doc ([a0f33ce](https://github.com/anatolykoptev/ox-whisper/commit/a0f33ce9668a973923120783abed9914087680aa))
* install.sh + docker-compose.yml + scripts/download-models.sh ([#6](https://github.com/anatolykoptev/ox-whisper/issues/6)) ([71ccd32](https://github.com/anatolykoptev/ox-whisper/commit/71ccd32bf3dd448828baa707c7efe009cd50aab8))
* Prometheus metrics on :9092 ([#4](https://github.com/anatolykoptev/ox-whisper/issues/4)) ([425785f](https://github.com/anatolykoptev/ox-whisper/commit/425785f5cf9a1fe0c6935f869fbaecc9b4100775))
* scaffold ox-whisper project with config and stub server ([aaa2722](https://github.com/anatolykoptev/ox-whisper/commit/aaa2722fbdfd8c5c579e273cfaa194b1a1de848c))
* update sherpa-onnx to v1.12.28, add confidence scores and GigaAM v3 architecture ([81577bf](https://github.com/anatolykoptev/ox-whisper/commit/81577bf6ce25fe1c974313ce400c57e6995d1207))
* upgrade to Moonshine v2 + CnnBilstm punctuation + sherpa-onnx v1.12.28 ([5fc200e](https://github.com/anatolykoptev/ox-whisper/commit/5fc200ecfa9895a1d1bc9d26555714260fecb371))
* WebSocket review fixes, test script, bump to v0.5.0 ([5881b38](https://github.com/anatolykoptev/ox-whisper/commit/5881b3849c6c510e3d4f1964689fcf548610649d))


### Bug Fixes

* add 50ms silence padding to VAD segments to reduce boundary artifacts ([277e4f6](https://github.com/anatolykoptev/ox-whisper/commit/277e4f6ed4a1f3ce11cfbde82501dd8a24a9992a))
* bug fixes + spawn_blocking optimization ([29283a1](https://github.com/anatolykoptev/ox-whisper/commit/29283a166d0832d8b064db3729f5340ecc433138))
* correct model detection, chunk size, punctuation and remove audio limit ([19b09fb](https://github.com/anatolykoptev/ox-whisper/commit/19b09fb3f57373845f759f8fde4c35feeecb9322))
* PII redaction — tighten phone regex, fix word counter, add mixed test ([07b9fab](https://github.com/anatolykoptev/ox-whisper/commit/07b9fab5864229bfaef8976e680bbffa16ae63dd))
* skip duplicate punctuation for EN, retry empty Moonshine chunks ([9d75a30](https://github.com/anatolykoptev/ox-whisper/commit/9d75a303fad674da97678f7935dda836cf054150))
* use Moonshine v1 model format compatible with sherpa-rs 0.6.8 ([f258ea0](https://github.com/anatolykoptev/ox-whisper/commit/f258ea0b04d6e48ebf5a1a767f4826b4e66a7dce))
* word timestamps, PII regex, spelling punctuation; bump to v0.4.0 ([d1e95d7](https://github.com/anatolykoptev/ox-whisper/commit/d1e95d74789109b73bf0c11b0fc7a2cae60dbdd2))


### Performance Improvements

* enable ORT_ENABLE_ALL graph optimization, batch decode, XNNPACK provider ([5822dbd](https://github.com/anatolykoptev/ox-whisper/commit/5822dbd0f6da58d700097ee1817a7ccc94fdf57c))
* set sample_rate, feature_dim and decoding_method for TransducerConfig ([61b00ff](https://github.com/anatolykoptev/ox-whisper/commit/61b00ff38c4e04ce8d1acdda985927d3cde324f0))
