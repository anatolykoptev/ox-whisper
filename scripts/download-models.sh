#!/usr/bin/env bash
# scripts/download-models.sh — fetch the models ox-whisper serves with.
# Idempotent: skips files that already exist with non-zero size.
#
# Usage:
#   ./scripts/download-models.sh [models_dir]
# Default models_dir: ./models
#
# Two models: Parakeet TDT 0.6B v3 (fp32, ~2.5 GB, ~3.2 GB RAM resident) for
# every language, and the Silero VAD (0.6 MB). Parakeet's files are sha256-pinned.

set -euo pipefail

MODELS_DIR="${1:-./models}"
mkdir -p "$MODELS_DIR"/{vad,parakeet}

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

# --- VAD: Silero ---
fetch \
  "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx" \
  "$MODELS_DIR/vad/silero_vad.onnx"

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
# fail before fetching 2.5 GB, not after
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

log "All models downloaded to $MODELS_DIR"
log "Sizes:"
du -sh "$MODELS_DIR"/* >&2
