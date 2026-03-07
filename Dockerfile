# Stage 1: Build
FROM rust:1.88-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake libclang-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release && rm -rf src

COPY src/ src/
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# sherpa-onnx dynamic libraries
COPY --from=builder /app/target/release/libsherpa-onnx-c-api.so /usr/lib/
COPY --from=builder /app/target/release/libsherpa-onnx-cxx-api.so /usr/lib/
COPY --from=builder /app/target/release/libonnxruntime.so /usr/lib/
RUN ldconfig

COPY --from=builder /app/target/release/ox-whisper /usr/local/bin/ox-whisper

ENV MOONSHINE_PORT=8092
ENV MOONSHINE_MODELS_DIR=/models
ENV ZIPFORMER_RU_DIR=/ru-models
ENV SILERO_VAD_MODEL=/vad/silero_vad.onnx
ENV PUNCT_MODEL=/punct/model.int8.onnx

EXPOSE 8092

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -sf http://localhost:8092/health || exit 1

ENTRYPOINT ["ox-whisper"]
