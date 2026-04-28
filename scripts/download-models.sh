#!/usr/bin/env bash
# scripts/download-models.sh — fetch ASR models for ox-whisper.
# Idempotent: skips files that already exist with non-zero size.
#
# Usage:
#   ./scripts/download-models.sh [models_dir]
# Default models_dir: ./models

set -euo pipefail

MODELS_DIR="${1:-./models}"
mkdir -p "$MODELS_DIR"/{en,ru,vad,punct-en}

log() { printf '\033[32m==>\033[0m %s\n' "$*" >&2; }

fetch() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    log "skip (exists): $dest"
    return
  fi
  log "fetch: $url"
  curl -fSL --retry 3 -o "$dest.tmp" "$url"
  mv "$dest.tmp" "$dest"
}

fetch_archive() {
  local url="$1" target_dir="$2" sentinel="$3"
  if [[ -s "$sentinel" ]]; then
    log "skip (exists): $target_dir"
    return
  fi
  local tmp
  tmp=$(mktemp -d)
  log "fetch + extract: $url"
  curl -fSL --retry 3 -o "$tmp/archive.tar.bz2" "$url"
  tar -xjf "$tmp/archive.tar.bz2" -C "$tmp"
  local inner
  inner=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -1)
  cp -r "$inner"/* "$target_dir/"
  rm -rf "$tmp"
}

# --- EN: Moonshine v2 base (HuggingFace) ---
EN_BASE="https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/resolve/main"
fetch "$EN_BASE/encoder_model.ort"          "$MODELS_DIR/en/encoder_model.ort"
fetch "$EN_BASE/decoder_model_merged.ort"   "$MODELS_DIR/en/decoder_model_merged.ort"
fetch "$EN_BASE/tokens.txt"                 "$MODELS_DIR/en/tokens.txt"

# --- RU: Zipformer INT8 (sherpa-onnx GitHub Release) ---
fetch_archive \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2" \
  "$MODELS_DIR/ru" \
  "$MODELS_DIR/ru/tokens.txt"

# --- VAD: Silero ---
fetch \
  "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx" \
  "$MODELS_DIR/vad/silero_vad.onnx"

# --- Punctuation: CNN-BiLSTM EN ---
fetch_archive \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2" \
  "$MODELS_DIR/punct-en" \
  "$MODELS_DIR/punct-en/model.int8.onnx"

log "All models downloaded to $MODELS_DIR"
log "Sizes:"
du -sh "$MODELS_DIR"/* >&2
