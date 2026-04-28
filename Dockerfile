# syntax=docker/dockerfile:1.4

# Stage 1: Chef
FROM rust:1.88-bookworm AS chef
RUN apt-get update && apt-get install -y --no-install-recommends cmake libclang-dev pkg-config && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef --locked
WORKDIR /app

# Stage 2: Planner
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Builder
FROM chef AS builder

# Copy vendored deps first (rarely changes). Minimum-vendor: only Rust
# binding files of sherpa-rs-sys are tracked; the C++ submodule is absent.
# build.rs uses pre-built libs from SHERPA_LIB_PATH and the committed
# src/bindings.rs — no C++ compile, so no sed patch needed.
COPY vendor/ vendor/

# Cook deps (cached layer)
COPY --from=planner /app/recipe.json recipe.json
COPY Cargo.toml Cargo.lock ./
ENV SHERPA_LIB_PATH=/app/vendor/sherpa-onnx
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo chef cook --release --locked --recipe-path recipe.json

# Build actual binary. Touch src/main.rs to bust cargo's fingerprint
# (cargo-chef cook left a stub binary at target/release/ox-whisper in the
# cache mount; without a source-newer-than-binary signal, cargo skips link).
COPY src/ src/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    touch src/main.rs && \
    rm -f /app/target/release/ox-whisper && \
    cargo build --release --locked --bin ox-whisper && \
    cp target/release/ox-whisper /binary && \
    test "$(stat -c %s /binary)" -gt 1000000 || (echo "ERROR: binary too small ($(stat -c %s /binary) bytes), build did not link"; exit 1)

# Stage 4: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# sherpa-onnx shared libraries from vendor
COPY vendor/sherpa-onnx/lib/libsherpa-onnx-c-api.so /usr/lib/
COPY vendor/sherpa-onnx/lib/libsherpa-onnx-cxx-api.so /usr/lib/
COPY vendor/sherpa-onnx/lib/libonnxruntime.so /usr/lib/
RUN ldconfig

COPY --from=builder /binary /usr/local/bin/ox-whisper

ENV MOONSHINE_PORT=8092
ENV MOONSHINE_MODELS_DIR=/models
ENV ZIPFORMER_RU_DIR=/ru-models
ENV SILERO_VAD_MODEL=/vad/silero_vad.onnx
ENV PUNCT_MODEL=/punct/model.int8.onnx
ENV PUNCT_VOCAB=/punct/bpe.vocab
ENV DIARIZE_SEGMENTATION_MODEL=/diarize/segmentation.onnx
ENV DIARIZE_EMBEDDING_MODEL=/diarize/embedding.onnx

EXPOSE 8092
EXPOSE 9092

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -sf http://localhost:8092/health || exit 1

ENTRYPOINT ["ox-whisper"]
