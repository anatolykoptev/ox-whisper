# ox-whisper Roadmap

> Self-hosted speech-to-text that replaces Deepgram/AssemblyAI for teams who can't send audio to the cloud.
> Принцип: CPU-only, один контейнер, Docker < 1 GB, zero data retention.

## v0.1.0 — MVP (done)

8 языков, Silero VAD, word timestamps, пунктуация, SSE streaming. Moonshine v2 + Zipformer.

## v0.2.0 — OpenAI-Compatible API (done)

Drop-in замена `/v1/audio/transcriptions` — любой клиент OpenAI Whisper API работает без изменений.

- [x] `POST /v1/audio/transcriptions` (multipart: file, model, language, response_format, timestamp_granularities)
- [x] Форматы ответа: `json`, `verbose_json`, `text`, `srt`, `vtt`
- [x] `verbose_json`: segments с start/end/text + word-level timestamps
- [x] SRT/VTT генерация из word timestamps (8 тестов)
- [x] Авто-определение языка (EN/RU по длине выхода модели)
- [x] `GET /v1/models` — список загруженных моделей
- [x] Python OpenAI SDK совместимость проверена
- [ ] `POST /v1/audio/translations` — транскрибция + перевод в EN (через LLM proxy)

## v0.3.0 — DX + Bugfixes (done)

DX-улучшения от конкурентов + исправление багов пунктуации и пустых результатов.

- [x] `extra` passthrough — произвольные KV-пары в запросе проходят в ответ (как Deepgram)
- [x] `custom_spelling` — замена слов в выводе (`{"from": ["докер"], "to": "Docker"}`)
- [x] `language_confidence` — возвращать confidence в verbose_json при авто-определении
- [x] Авто-определение модели: GigaAM v3 > GigaAM v2 > Zipformer (реализовано в v0.1)
- [x] Fix: EN двойная пунктуация — Moonshine v2 уже ставит знаки, внешний пунктуатор пропускается
- [x] Fix: пустой результат на длинных чанках — retry с разбиением пополам
- [x] Fix: удалены unused `extract_words()`, `Pool::size()`
- [ ] GigaAM v3 RNNT — код готов, нужно подключить модель в Docker
- [ ] Бенчмарки WER на Golos/CommonVoice

## v0.4.0 — Smart Features

Фичи, которые отличают от голого Whisper. Все 3 конкурента имеют — must-have для production.

- [ ] Keyword boosting (hot words) — список терминов/фраз для улучшения распознавания (Deepgram: до 100 фраз, AssemblyAI: до 1000)
- [ ] Speaker diarization — кто говорит (Sortformer/pyannote через отдельный ONNX). `speaker` per word, `utterances[]` per turn
- [ ] PII redaction — маскирование номеров, email, имён. Regex + NER, вывод `[EMAIL_1]` / `[PHONE_2]` (как Deepgram streaming)
- [ ] Smart formatting — числа → цифры, даты, валюта, телефоны (Deepgram `smart_format`)
- [ ] Paragraphs — авто-разбивка на абзацы по паузам и смыслу

## v0.5.0 — Real-Time & Async

WebSocket streaming + async pipeline. Паттерны от Deepgram (dual flags) и AssemblyAI (turn-based).

- [ ] WebSocket endpoint `/v1/listen` — live streaming (микрофон → текст)
- [ ] Dual flags: `is_final` (текст стабилен) + `speech_final` (говорящий закончил) — как Deepgram
- [ ] Turn-based events — группировка по очередям говорящего (как AssemblyAI v3)
- [ ] `ForceEndpoint` / `Finalize` — клиент программно завершает turn
- [ ] Temporary auth tokens — `POST /v1/token` для browser-side streaming без API key
- [ ] Webhook callbacks — `callback_url` в запросе, POST результат async (retry 10x)
- [ ] Batch API — очередь файлов, статус по job_id, webhook по завершении
- [ ] `messages_config` — клиент подписывается только на нужные типы WS-событий (как Gladia)

## v0.6.0 — Multi-Engine

Выбор лучшей модели под задачу. Fallback-цепочка как AssemblyAI `speech_models[]`.

- [ ] Parakeet TDT 0.6B INT8 — SOTA качество EN (WER ~8%, 30x RT)
- [ ] SenseVoice — ZH, JA, KO с эмоциями
- [ ] `model` параметр: `moonshine`, `parakeet`, `gigaam`, `auto`
- [ ] `speech_models[]` — приоритетный список моделей с fallback (как AssemblyAI)

## v0.7.0 — Audio Intelligence

LLM post-processing поверх транскрипта. Аналог LeMUR (AssemblyAI) и audio_to_llm (Gladia).

- [ ] `audio_to_llm` — произвольные LLM-промпты к транскрипту через CLIProxyAPI
- [ ] Summarization — краткое содержание (general/bullets/concise)
- [ ] Sentiment analysis — тональность per utterance
- [ ] Entity detection — NER (имена, организации, даты, суммы)
- [ ] Auto chapters — разбивка на главы с headline + keywords (как AssemblyAI)

## v0.8.0 — Observability & Enterprise

- [ ] Prometheus metrics (`/metrics`): latency p50/p95/p99, requests, errors, model usage, queue depth
- [ ] Structured JSON logging
- [ ] API key authentication (Bearer token)
- [ ] Rate limiting per key с `X-RateLimit-*` headers
- [ ] Usage tracking (минуты аудио per key) с `tag` для cost attribution (как Deepgram)
- [ ] Health check с деталями моделей (`/v1/health`)

## Не в плане (осознанно)

| Что | Почему нет |
|-----|-----------|
| GPU inference | Наша ниша — CPU. GPU = NVIDIA lock-in, сложный deploy |
| Candle backend | Интересно, но ONNX Runtime надёжнее для production CPU |
| Whisper модели | Moonshine и Parakeet быстрее и легче на CPU |
| gRPC | HTTP + WebSocket + SSE покрывает все use cases |
| On-device SDK | Фокус на server-side, не мобильные SDK |
| Audio beep redaction | Генерация нового аудио — слишком нишево для self-hosted |
| Code switching | 100+ языков как Gladia — нереалистично без огромных моделей |
| Translation с lipsync | Видео-ориентированная фича, не наш фокус |

## Конкурентный контекст

| Feature | Deepgram | AssemblyAI | Gladia | ox-whisper | Цель |
|---------|----------|------------|--------|------------|------|
| Self-hosted | $$$$ | нет | Enterprise only | **да** | -- |
| OpenAI API compat | нет | нет | нет | **v0.2** | -- |
| SRT/VTT | да | да | да | **v0.2** | -- |
| Language detection | да + confidence | да + confidence + threshold | да + code switching | **v0.2** (basic) | v0.3 |
| Keyword boosting | `keywords` + `keyterm` (100) | `word_boost` (1000) + `keyterms_prompt` | `custom_vocabulary` + intensity | нет | v0.4 |
| Speaker diarization | per-word speaker | per-utterance speaker | per-utterance + min/max | нет | v0.4 |
| PII redaction | 50+ типов, streaming | 50+ типов + audio beep | GDPR/HIPAA groups | нет | v0.4 |
| Smart formatting | даты, $, тел., URL | format_text + punctuate | display_mode | пунктуация | v0.4 |
| WebSocket streaming | `/v1/listen` + Flux `/v2/listen` | v3 turn-based | POST init → WSS | SSE only | v0.5 |
| Webhook callbacks | retry 10x/30s | retry 10x/10s + custom auth | callback_config | нет | v0.5 |
| Temp auth tokens | нет | `POST /v2/realtime/token` | token в WSS URL | нет | v0.5 |
| LLM over transcript | нет | LeMUR (4 endpoints) | audio_to_llm (multi-prompt) | нет | v0.7 |
| Summarization | `summarize=v2` | LeMUR summary | `summarization` | нет | v0.7 |
| Sentiment | per-word + utterance | per-sentence | per-utterance + emotion | нет | v0.7 |
| Entity detection | `detect_entities` | 50+ entity types | NER | нет | v0.7 |
| Word timestamps | да | да | да | **v0.1** | -- |
| VAD | да | configurable threshold | да | **v0.1** | -- |
| Punctuation | да | да | да | **v0.1** | -- |
| CPU-only | нет | нет | нет | **v0.1** | -- |
| Docker < 1 GB | нет | нет | нет | **v0.1 (760 MB)** | -- |
| Zero data retention | нет | нет | Enterprise only | **v0.1** | -- |
