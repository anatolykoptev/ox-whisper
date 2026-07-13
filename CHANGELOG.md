# Changelog

## [0.8.2](https://github.com/anatolykoptev/ox-whisper/compare/v0.8.1...v0.8.2) (2026-07-13)


### Added

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
* **docker:** add sccache + mold для signal-grade build cache ([#26](https://github.com/anatolykoptev/ox-whisper/issues/26)) ([54682c2](https://github.com/anatolykoptev/ox-whisper/commit/54682c27b37bda5eae79123f84d77c136c9b16f2))
* install.sh + docker-compose.yml + scripts/download-models.sh ([#6](https://github.com/anatolykoptev/ox-whisper/issues/6)) ([71ccd32](https://github.com/anatolykoptev/ox-whisper/commit/71ccd32bf3dd448828baa707c7efe009cd50aab8))
* **pool:** opt-in idle model eviction ([#19](https://github.com/anatolykoptev/ox-whisper/issues/19)) ([86e18fe](https://github.com/anatolykoptev/ox-whisper/commit/86e18fec36df7b2b34ee5529a75d0c8b3031bd39))
* Prometheus metrics on :9092 ([#4](https://github.com/anatolykoptev/ox-whisper/issues/4)) ([425785f](https://github.com/anatolykoptev/ox-whisper/commit/425785f5cf9a1fe0c6935f869fbaecc9b4100775))
* scaffold ox-whisper project with config and stub server ([aaa2722](https://github.com/anatolykoptev/ox-whisper/commit/aaa2722fbdfd8c5c579e273cfaa194b1a1de848c))
* **security:** add cargo-deny config + adopt nextest ([#24](https://github.com/anatolykoptev/ox-whisper/issues/24)) ([6876b0a](https://github.com/anatolykoptev/ox-whisper/commit/6876b0a4db1d4efe4ea10d39cd8f244a35f19c3f))
* update sherpa-onnx to v1.12.28, add confidence scores and GigaAM v3 architecture ([81577bf](https://github.com/anatolykoptev/ox-whisper/commit/81577bf6ce25fe1c974313ce400c57e6995d1207))
* upgrade to Moonshine v2 + CnnBilstm punctuation + sherpa-onnx v1.12.28 ([5fc200e](https://github.com/anatolykoptev/ox-whisper/commit/5fc200ecfa9895a1d1bc9d26555714260fecb371))
* WebSocket review fixes, test script, bump to v0.5.0 ([5881b38](https://github.com/anatolykoptev/ox-whisper/commit/5881b3849c6c510e3d4f1964689fcf548610649d))


### Fixed

* add 50ms silence padding to VAD segments to reduce boundary artifacts ([277e4f6](https://github.com/anatolykoptev/ox-whisper/commit/277e4f6ed4a1f3ce11cfbde82501dd8a24a9992a))
* address panic/correctness bugs from issue [#29](https://github.com/anatolykoptev/ox-whisper/issues/29) ([c82bb23](https://github.com/anatolykoptev/ox-whisper/commit/c82bb23d5c0e9401d6a8c140461dfb4de5cc8f43))
* address panic/correctness bugs from issue [#29](https://github.com/anatolykoptev/ox-whisper/issues/29) ([6d8c695](https://github.com/anatolykoptev/ox-whisper/commit/6d8c6951653e240289ed7b71e84e205c1693d7c4))
* bug fixes + spawn_blocking optimization ([29283a1](https://github.com/anatolykoptev/ox-whisper/commit/29283a166d0832d8b064db3729f5340ecc433138))
* **ci:** minimum-vendor sherpa-rs-sys (Rust files only, 220 KB) ([#15](https://github.com/anatolykoptev/ox-whisper/issues/15)) ([cd9042b](https://github.com/anatolykoptev/ox-whisper/commit/cd9042b45a54ec84f0d2d3eb054bff0394035e18))
* correct model detection, chunk size, punctuation and remove audio limit ([19b09fb](https://github.com/anatolykoptev/ox-whisper/commit/19b09fb3f57373845f759f8fde4c35feeecb9322))
* **dockerfile:** bust cargo-chef stub binary before final link ([#10](https://github.com/anatolykoptev/ox-whisper/issues/10)) ([5ba9508](https://github.com/anatolykoptev/ox-whisper/commit/5ba95086b36c1dfa6a32ff95633dd4a8d7374b13))
* **dockerfile:** drop sed patch on absent sherpa-onnx C++ source ([#16](https://github.com/anatolykoptev/ox-whisper/issues/16)) ([f0d6c0a](https://github.com/anatolykoptev/ox-whisper/commit/f0d6c0af375be78a80457bd92757b9ca5e499263))
* **dockerfile:** fetch sherpa-rs-sys from crates.io at CI build time ([#13](https://github.com/anatolykoptev/ox-whisper/issues/13)) ([3c75b22](https://github.com/anatolykoptev/ox-whisper/commit/3c75b22332e0b2a29616421cccd202d2ebee6b61))
* **docker:** remove RUSTC_WRAPPER=sccache to fix E0463 tracing_attributes ([#27](https://github.com/anatolykoptev/ox-whisper/issues/27)) ([5b9da89](https://github.com/anatolykoptev/ox-whisper/commit/5b9da8928994f8c70e84b00e84ebf3bd30f8f785))
* PII redaction — tighten phone regex, fix word counter, add mixed test ([07b9fab](https://github.com/anatolykoptev/ox-whisper/commit/07b9fab5864229bfaef8976e680bbffa16ae63dd))
* **pool:** prevent race on evicted-slot reinit ([5403353](https://github.com/anatolykoptev/ox-whisper/commit/5403353e7f77d3b88c870ad61840b646cbb83c78))
* skip duplicate punctuation for EN, retry empty Moonshine chunks ([9d75a30](https://github.com/anatolykoptev/ox-whisper/commit/9d75a303fad674da97678f7935dda836cf054150))
* use Moonshine v1 model format compatible with sherpa-rs 0.6.8 ([f258ea0](https://github.com/anatolykoptev/ox-whisper/commit/f258ea0b04d6e48ebf5a1a767f4826b4e66a7dce))
* word timestamps, PII regex, spelling punctuation; bump to v0.4.0 ([d1e95d7](https://github.com/anatolykoptev/ox-whisper/commit/d1e95d74789109b73bf0c11b0fc7a2cae60dbdd2))


### Performance

* enable ORT_ENABLE_ALL graph optimization, batch decode, XNNPACK provider ([5822dbd](https://github.com/anatolykoptev/ox-whisper/commit/5822dbd0f6da58d700097ee1817a7ccc94fdf57c))
* set sample_rate, feature_dim and decoding_method for TransducerConfig ([61b00ff](https://github.com/anatolykoptev/ox-whisper/commit/61b00ff38c4e04ce8d1acdda985927d3cde324f0))


### Changed

* extract hardcoded values to Config env vars ([8fbd01a](https://github.com/anatolykoptev/ox-whisper/commit/8fbd01a03e29b1ab5808d138e126743df4276fc0))
* extract word timestamps to words.rs, trim transcribe.rs to &lt;200 lines ([709e9d3](https://github.com/anatolykoptev/ox-whisper/commit/709e9d3ef2116cb168974692aacfea25c5747e8b))


### Documentation

* add language table and full multilingual benchmarks ([57cdbe4](https://github.com/anatolykoptev/ox-whisper/commit/57cdbe4540743a61bb87538e3c4c76d586db2a40))
* add README with API reference, configuration, and project structure ([c3464db](https://github.com/anatolykoptev/ox-whisper/commit/c3464dbf4ce1c7459e97a9a2e2b9728b61b6b276))
* add v0.3.0 benchmarks — latency, WER, throughput ([3d4a5db](https://github.com/anatolykoptev/ox-whisper/commit/3d4a5db8845f88cc695b577fbbc25a774180301f))
* mark v0.3.0 as done, add bugfix items ([b0526e5](https://github.com/anatolykoptev/ox-whisper/commit/b0526e522a83d63e5f94592b728a724dde0329cd))
* remove duplicate merge commit from release changelog ([434a4f6](https://github.com/anatolykoptev/ox-whisper/commit/434a4f65ae79562328f6eeb5e2543bcaf794999f))
* rewrite README — OpenAI API, WebSocket streaming, MIT license ([#2](https://github.com/anatolykoptev/ox-whisper/issues/2)) ([5ba5007](https://github.com/anatolykoptev/ox-whisper/commit/5ba50077fa8e9d56770b27138f2b5a8e792aebf4))
* rewrite README in best-in-class format ([#12](https://github.com/anatolykoptev/ox-whisper/issues/12)) ([ee12aa1](https://github.com/anatolykoptev/ox-whisper/commit/ee12aa1a3396edf1fe9b1b7c98b32d1bdbd97bf7))
* rewrite README with real benchmarks, cut filler ([6d14b0e](https://github.com/anatolykoptev/ox-whisper/commit/6d14b0e07a1447951723328bebdc30bf266a18ab))
* SEO-trim README — drop duplicates, lead with Whisper API alternative ([#18](https://github.com/anatolykoptev/ox-whisper/issues/18)) ([3a11fae](https://github.com/anatolykoptev/ox-whisper/commit/3a11faeef0aa7763f8831454162cadb2544a72ad))
* update roadmap — mark v0.2.0 as done ([8ef1f4b](https://github.com/anatolykoptev/ox-whisper/commit/8ef1f4b4715ccc5c94e3448f0b3a139ea595d67d))
* update roadmap with competitive insights from Deepgram/AssemblyAI/Gladia ([c159686](https://github.com/anatolykoptev/ox-whisper/commit/c159686351209b4f987a3efc8433cee0d73f75e7))

## [0.8.1](https://github.com/anatolykoptev/ox-whisper/compare/v0.8.0...v0.8.1) (2026-07-13)


### Fixed

* address panic/correctness bugs from issue [#29](https://github.com/anatolykoptev/ox-whisper/issues/29) ([6d8c695](https://github.com/anatolykoptev/ox-whisper/commit/6d8c6951653e240289ed7b71e84e205c1693d7c4))
* **pool:** prevent race on evicted-slot reinit ([5403353](https://github.com/anatolykoptev/ox-whisper/commit/5403353e7f77d3b88c870ad61840b646cbb83c78))

## [0.8.0](https://github.com/anatolykoptev/ox-whisper/compare/v0.7.0...v0.8.0) (2026-05-11)


### Features

* **docker:** add sccache + mold для signal-grade build cache ([#26](https://github.com/anatolykoptev/ox-whisper/issues/26)) ([54682c2](https://github.com/anatolykoptev/ox-whisper/commit/54682c27b37bda5eae79123f84d77c136c9b16f2))
* **security:** add cargo-deny config + adopt nextest ([#24](https://github.com/anatolykoptev/ox-whisper/issues/24)) ([6876b0a](https://github.com/anatolykoptev/ox-whisper/commit/6876b0a4db1d4efe4ea10d39cd8f244a35f19c3f))


### Bug Fixes

* **docker:** remove RUSTC_WRAPPER=sccache to fix E0463 tracing_attributes ([#27](https://github.com/anatolykoptev/ox-whisper/issues/27)) ([5b9da89](https://github.com/anatolykoptev/ox-whisper/commit/5b9da8928994f8c70e84b00e84ebf3bd30f8f785))

## [0.7.0](https://github.com/anatolykoptev/ox-whisper/compare/v0.6.1...v0.7.0) (2026-05-09)


### Features

* **pool:** opt-in idle model eviction ([#19](https://github.com/anatolykoptev/ox-whisper/issues/19)) ([86e18fe](https://github.com/anatolykoptev/ox-whisper/commit/86e18fec36df7b2b34ee5529a75d0c8b3031bd39))

## [0.6.1](https://github.com/anatolykoptev/ox-whisper/compare/v0.6.0...v0.6.1) (2026-04-28)


### Bug Fixes

* **dockerfile:** bust cargo-chef stub binary before final link ([#10](https://github.com/anatolykoptev/ox-whisper/issues/10)) ([5ba9508](https://github.com/anatolykoptev/ox-whisper/commit/5ba95086b36c1dfa6a32ff95633dd4a8d7374b13))

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
