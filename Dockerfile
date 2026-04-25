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

# Copy vendored deps first (rarely changes)
COPY vendor/ vendor/

# Optimize sherpa-onnx session: enable ORT_ENABLE_ALL graph optimizations, set inter_op=1
RUN sed -i \
    -e 's/SetInterOpNumThreads(num_threads)/SetInterOpNumThreads(1)/' \
    -e 's|// sess_opts.SetGraphOptimizationLevel|sess_opts.SetGraphOptimizationLevel|' \
    -e 's/ORT_ENABLE_EXTENDED/ORT_ENABLE_ALL/' \
    vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc

# Cook deps (cached layer)
COPY --from=planner /app/recipe.json recipe.json
COPY Cargo.toml Cargo.lock ./
ENV SHERPA_LIB_PATH=/app/vendor/sherpa-onnx
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo chef cook --release --locked --recipe-path recipe.json

# Build actual binary
COPY src/ src/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked && \
    cp target/release/ox-whisper /binary

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

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -sf http://localhost:8092/health || exit 1

ENTRYPOINT ["ox-whisper"]
