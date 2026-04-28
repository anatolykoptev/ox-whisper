#!/usr/bin/env bash
# install.sh — one-command bootstrap for ox-whisper.
#
#   curl -fsSL https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master/install.sh | bash
#
# Or with overrides:
#   curl -fsSL .../install.sh | OX_WHISPER_DIR=/opt/ox-whisper bash
#
# Env overrides:
#   OX_WHISPER_DIR        target directory (default: $HOME/ox-whisper)
#   OX_WHISPER_VERSION    image tag (default: latest)
#   OX_WHISPER_REPO_RAW   raw github base (for forks)

set -euo pipefail

OX_WHISPER_DIR="${OX_WHISPER_DIR:-$HOME/ox-whisper}"
OX_WHISPER_VERSION="${OX_WHISPER_VERSION:-latest}"
OX_WHISPER_REPO_RAW="${OX_WHISPER_REPO_RAW:-https://raw.githubusercontent.com/anatolykoptev/ox-whisper/master}"

log()  { printf '\033[32m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[31mERR\033[0m %s\n' "$*" >&2; exit 1; }

log "Target: $OX_WHISPER_DIR (version: $OX_WHISPER_VERSION)"

ARCH=$(uname -m)
case "$ARCH" in
  aarch64|arm64) ;;
  *) die "Only aarch64 / arm64 supported (sherpa-onnx vendored .so is ARM-only). Detected: $ARCH" ;;
esac

command -v curl >/dev/null || die "curl required"
command -v tar  >/dev/null || die "tar required"

if ! command -v docker >/dev/null; then
  warn "docker not found — installing..."
  curl -fsSL https://get.docker.com | sh
fi

if ! docker compose version >/dev/null 2>&1; then
  die "docker compose plugin required. Install with: apt install docker-compose-plugin"
fi

mkdir -p "$OX_WHISPER_DIR"/{models,scripts}
cd "$OX_WHISPER_DIR"

log "Fetching docker-compose.yml..."
curl -fsSL "$OX_WHISPER_REPO_RAW/docker-compose.yml" -o docker-compose.yml

log "Fetching scripts/download-models.sh..."
curl -fsSL "$OX_WHISPER_REPO_RAW/scripts/download-models.sh" -o scripts/download-models.sh
chmod +x scripts/download-models.sh

log "Downloading ASR models (~463 MB)..."
./scripts/download-models.sh ./models

log "Pulling ox-whisper image ($OX_WHISPER_VERSION)..."
OX_WHISPER_VERSION="$OX_WHISPER_VERSION" docker compose pull

log "Starting ox-whisper..."
OX_WHISPER_VERSION="$OX_WHISPER_VERSION" docker compose up -d

log "Waiting for healthcheck..."
for i in $(seq 1 60); do
  status=$(docker inspect ox-whisper --format '{{.State.Health.Status}}' 2>/dev/null || echo "starting")
  case "$status" in
    healthy) log "Healthy"; break ;;
    unhealthy) die "Container went unhealthy. Logs: docker logs ox-whisper" ;;
  esac
  sleep 2
done

log "Checking endpoints..."
curl -sf http://127.0.0.1:8092/health >/dev/null && log "  /health OK"
curl -sf http://127.0.0.1:9092/metrics | head -1 >/dev/null && log "  /metrics OK"

cat <<EOF

\033[32m✓ ox-whisper is running\033[0m

  Transcription:  http://127.0.0.1:8092
  OpenAI API:     http://127.0.0.1:8092/v1/audio/transcriptions
  WebSocket:      ws://127.0.0.1:8092/v1/listen
  Metrics:        http://127.0.0.1:9092/metrics

  Logs:    docker logs -f ox-whisper
  Stop:    cd $OX_WHISPER_DIR && docker compose down
  Update:  cd $OX_WHISPER_DIR && docker compose pull && docker compose up -d

EOF
