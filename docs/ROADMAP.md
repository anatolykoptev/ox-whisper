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

## v0.3.0 — RU Quality + GigaAM v3

WER 4-5% на русском (сейчас ~8-10%). Встроенная пунктуация.

- [ ] GigaAM v3 RNNT с built-in punctuation
- [ ] Авто-определение модели: GigaAM v3 > GigaAM v2 > Zipformer
- [ ] Бенчмарки WER на Golos/CommonVoice

## v0.4.0 — Smart Features

Фичи, которые отличают от голого Whisper.

- [ ] Keyword boosting (hot words) — список терминов с весами для улучшения распознавания имён, продуктов, jargon
- [ ] Speaker diarization — кто говорит (Sortformer/pyannote через отдельный ONNX)
- [ ] PII redaction — маскирование номеров, email, имён в выводе (regex + NER)
- [ ] Paragraphs / smart formatting — разбивка на абзацы, числа → цифры, "двадцать пять" → "25"

## v0.5.0 — Real-Time & Scale

- [ ] WebSocket endpoint для live streaming (микрофон → текст в реальном времени)
- [ ] Partial results с progressive refinement (interim → final)
- [ ] Webhook callbacks — `POST /v1/audio/transcriptions` с `callback_url`, результат приходит async
- [ ] Batch API — очередь файлов, статус по job_id

## v0.6.0 — Multi-Engine

Выбор лучшей модели под задачу.

- [ ] Parakeet TDT 0.6B INT8 — SOTA качество EN (WER ~8%, 30x RT)
- [ ] SenseVoice — ZH, JA, KO с эмоциями
- [ ] Авто-роутинг: язык → лучший движок
- [ ] `model` параметр: `moonshine`, `parakeet`, `gigaam`, `auto`

## v0.7.0 — Observability & Enterprise

- [ ] Prometheus metrics (`/metrics`): latency p50/p95/p99, requests, errors, model usage, queue depth
- [ ] Structured JSON logging
- [ ] API key authentication (Bearer token)
- [ ] Rate limiting per key
- [ ] Usage tracking (минуты аудио per key)
- [ ] Health check с деталями моделей (`/v1/health`)

## Не в плане (осознанно)

| Что | Почему нет |
|-----|-----------|
| GPU inference | Наша ниша — CPU. GPU = NVIDIA lock-in, сложный deploy |
| Candle backend | Интересно, но ONNX Runtime надёжнее для production CPU |
| Whisper модели | Moonshine и Parakeet быстрее и легче на CPU |
| gRPC | HTTP + WebSocket + SSE покрывает все use cases |
| On-device SDK | Фокус на server-side, не мобильные SDK |

## Конкурентный контекст

| Feature | Deepgram | AssemblyAI | Gladia | ox-whisper v0.2 | Цель |
|---------|----------|------------|--------|-----------------|------|
| Self-hosted | $$$$ | нет | нет | **да** | — |
| OpenAI API compat | нет | нет | нет | **да** | — |
| SRT/VTT | да | да | да | **да** | — |
| Language detection | да | да | да | **да** | — |
| Speaker diarization | да | да | да | нет | v0.4 |
| Keyword boosting | да | нет | нет | нет | v0.4 |
| PII redaction | да | да | да | нет | v0.4 |
| WebSocket streaming | да | да | да | SSE only | v0.5 |
| Word timestamps | да | да | да | **да** | — |
| VAD | да | да | да | **да** | — |
| Punctuation | да | да | да | **да** | — |
| CPU-only | нет | нет | нет | **да** | — |
| Docker < 1 GB | нет | нет | нет | **да (760 MB)** | — |
| Zero data retention | — | — | Gladia да | **да** | — |
