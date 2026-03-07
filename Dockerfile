# Stage 1: Build
FROM rust:1.88-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake libclang-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy vendored dependencies first for layer caching
COPY vendor/ vendor/
COPY Cargo.toml Cargo.lock ./

# Optimize sherpa-onnx session: enable ORT_ENABLE_ALL graph optimizations, set inter_op=1
RUN sed -i \
    -e 's/SetInterOpNumThreads(num_threads)/SetInterOpNumThreads(1)/' \
    -e 's|// sess_opts.SetGraphOptimizationLevel|sess_opts.SetGraphOptimizationLevel|' \
    -e 's/ORT_ENABLE_EXTENDED/ORT_ENABLE_ALL/' \
    vendor/sherpa-rs-sys/sherpa-onnx/sherpa-onnx/csrc/session.cc

# Build deps only (dummy main)
RUN mkdir src && echo "fn main(){}" > src/main.rs && \
    SHERPA_LIB_PATH=/app/vendor/sherpa-onnx cargo build --release && \
    rm -rf src target/release/deps/ox_whisper* target/release/ox-whisper

# Build actual binary
COPY src/ src/
RUN SHERPA_LIB_PATH=/app/vendor/sherpa-onnx cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# sherpa-onnx shared libraries from vendor
COPY vendor/sherpa-onnx/lib/libsherpa-onnx-c-api.so /usr/lib/
COPY vendor/sherpa-onnx/lib/libsherpa-onnx-cxx-api.so /usr/lib/
COPY vendor/sherpa-onnx/lib/libonnxruntime.so /usr/lib/
RUN ldconfig

COPY --from=builder /app/target/release/ox-whisper /usr/local/bin/ox-whisper

ENV MOONSHINE_PORT=8092
ENV MOONSHINE_MODELS_DIR=/models
ENV ZIPFORMER_RU_DIR=/ru-models
ENV SILERO_VAD_MODEL=/vad/silero_vad.onnx
ENV PUNCT_MODEL=/punct/model.int8.onnx
ENV PUNCT_VOCAB=/punct/bpe.vocab

EXPOSE 8092

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -sf http://localhost:8092/health || exit 1

ENTRYPOINT ["ox-whisper"]
