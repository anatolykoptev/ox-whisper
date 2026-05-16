# syntax=docker/dockerfile:1.4

# Stage 1: Chef
FROM rust:1.88-bookworm AS chef
RUN apt-get update && apt-get install -y --no-install-recommends cmake libclang-dev pkg-config clang mold curl && rm -rf /var/lib/apt/lists/*
# sccache: content-addressed compiler cache — hits survive BuildKit cache
# invalidation on source changes; mold replaces gold linker (3-5x faster link).
# mold is CXX-compat (sherpa-rs vendored bindings compile via clang+mold fine).
ENV SCCACHE_VERSION=0.15.0
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod +x /usr/local/bin/sccache
RUN cargo install cargo-chef --locked
WORKDIR /app

# Stage 2: Planner
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Builder
FROM chef AS builder

# mold linker via per-arch CARGO_TARGET_* — no RUSTC_WRAPPER=sccache here.
# ox-whisper is a 1-crate workspace; the only crate recompiling on source changes
# is ox-whisper itself (not cacheable by sccache). Deps are handled by cargo-chef
# cook + target/ cache mount. sccache proc-macro caching causes E0463 on
# tracing_attributes: sccache returns a metadata hit but doesn't restore the
# proc-macro .so to the path cargo expects. mold still active (3-5x faster link).
# sccache binary installed in chef stage remains available for dist-sccache use.
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=clang
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=clang
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"

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
    --mount=type=cache,target=/root/.cache/sccache,sharing=locked \
    cargo chef cook --release --locked --recipe-path recipe.json

# Build actual binary. Touch src/main.rs to bust cargo's fingerprint
# (cargo-chef cook left a stub binary at target/release/ox-whisper in the
# cache mount; without a source-newer-than-binary signal, cargo skips link).
COPY src/ src/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    --mount=type=cache,target=/root/.cache/sccache,sharing=locked \
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
