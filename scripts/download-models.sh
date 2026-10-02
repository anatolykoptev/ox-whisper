#!/usr/bin/env bash
# scripts/download-models.sh — fetch ASR models for ox-whisper.
# Idempotent: skips files that already exist with non-zero size.
#
# Usage:
#   ./scripts/download-models.sh [models_dir]
# Default models_dir: ./models
#
# Parakeet TDT 0.6B v3 (fp32, ~2.5 GB, ~3.2 GB RAM resident) serves ru, en and
# 23 more European languages when present. OX_WHISPER_PARAKEET=0 skips it, for
# small boxes: those languages then fall back to Zipformer (ru) and Moonshine.

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

# --- Parakeet TDT 0.6B v3, fp32 (HuggingFace), sha256-pinned ---
# fp32, not the int8 export: on 100+100 FLEURS ru/en clips the int8 export came
# out ~4 WER points worse, while fp32 matched the reference within 0.25 points.
sha256_of() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    printf 'need sha256sum or shasum to verify %s\n' "$1" >&2
    exit 1
  fi
}
# A file that exists but fails its hash (an older or damaged copy) is fetched
# again once in the same run; a fresh download that still fails stops the run.
fetch_sha() {
  local url="$1" dest="$2" want="$3"
  fetch "$url" "$dest"
  if [[ "$(sha256_of "$dest")" != "$want" ]]; then
    log "checksum mismatch, fetching again: $dest"
    rm -f "$dest"
    fetch "$url" "$dest"
    local got
    got=$(sha256_of "$dest")
    if [[ "$got" != "$want" ]]; then
      rm -f "$dest"
      printf 'checksum mismatch for %s: got %s, want %s\n' "$dest" "$got" "$want" >&2
      exit 1
    fi
  fi
}
# Created even when Parakeet is skipped: the compose file bind-mounts it, and a
# missing bind source is created by Docker as root, which a later download run
# (as a normal user) could not write into.
mkdir -p "$MODELS_DIR/parakeet"
case "${OX_WHISPER_PARAKEET:-1}" in
  0|false|no|off) want_parakeet=0 ;;
  *) want_parakeet=1 ;;
esac
if [[ "$want_parakeet" = 1 ]]; then
  # only the Parakeet files are hash-checked: fail before fetching them, not after
  command -v sha256sum >/dev/null || command -v shasum >/dev/null || {
    printf 'need sha256sum or shasum to verify the Parakeet download\n' >&2
    exit 1
  }
  # pinned to the upstream commit, so a re-push upstream cannot change what installs
  PK_BASE="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3/resolve/1a468a35cbba69418f126de829e75261dea4a4e4"
  fetch_sha "$PK_BASE/encoder.onnx"    "$MODELS_DIR/parakeet/encoder.onnx"    3eed7ce424bf8339ad09233533c687e2dbd07e74ccf5027b5e7344019ea373b0
  fetch_sha "$PK_BASE/encoder.weights" "$MODELS_DIR/parakeet/encoder.weights" 3af3f51af5f2d01dbbf5af47d42c7962a2c205f11004254bb4f2b979862f39a8
  fetch_sha "$PK_BASE/decoder.onnx"    "$MODELS_DIR/parakeet/decoder.onnx"    d593cdb0e571f5a457ec2219af9968cbf6b0e8198e8f7839b40a8754593bf68c
  fetch_sha "$PK_BASE/joiner.onnx"     "$MODELS_DIR/parakeet/joiner.onnx"     b9b0bcf88ac571902e69a6536223ed2d94885e981b85045410f1403d53121a63
  fetch_sha "$PK_BASE/tokens.txt"      "$MODELS_DIR/parakeet/tokens.txt"      d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d
else
  log "skip Parakeet (OX_WHISPER_PARAKEET=0)"
fi

log "All models downloaded to $MODELS_DIR"
log "Sizes:"
du -sh "$MODELS_DIR"/* >&2
