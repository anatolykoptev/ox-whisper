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

# Stage 4: qwentts-build — upstream tts-server (ServeurpersoCom/qwentts.cpp),
# pinned commit; ox-whisper spawns it as a child process at /opt/qwentts/tts-server.
FROM debian:bookworm-slim AS qwentts-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

ARG QWENTTS_COMMIT=6fae92914045cd83364d2845ceaa0f7969727319
ARG QWENTTS_GGML_COMMIT=40e16e4a814f7fe851a0c486fb9e8c722e957830

WORKDIR /src
RUN git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp qwentts && \
    cd qwentts && \
    git checkout -q "$QWENTTS_COMMIT" && \
    git submodule update --init --recursive && \
    test "$(git -C ggml rev-parse HEAD)" = "$QWENTTS_GGML_COMMIT"

# QT_N_THREADS env override: upstream divides hardware_concurrency() by 2 for
# SMT/HT, which halves throughput on SMT-less ARM cores (Ampere/Neoverse).
COPY patches/qwentts-threads.patch /tmp/qwentts-threads.patch
RUN git -C qwentts apply /tmp/qwentts-threads.patch

# Portable aarch64 build:
#   GGML_NATIVE=OFF        — no -mcpu=native; the image runs on any aarch64.
#   GGML_BACKEND_DL=ON     — CPU backends built as runtime-loaded modules
#                            (libggml-cpu-*.so), scored against host CPUID.
#   GGML_CPU_ALL_VARIANTS  — one module per ARM ISA level (armv8.0 → armv9.2),
#                            incl. a baseline for cores without dotprod.
#   BUILD_SHARED_LIBS=ON   — required by GGML_BACKEND_DL (ggml FATAL_ERRORs).
#   GGML_LLAMAFILE=ON      — sgemm fast path.
# Backend modules are dependencies of the ggml target, so building tts-server
# alone produces every libggml*.so needed at runtime.
RUN cmake -S qwentts -B qwentts/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_LLAMAFILE=ON && \
    cmake --build qwentts/build --target tts-server -j"$(nproc)" && \
    mkdir -p /opt/qwentts /usr/share/licenses/qwentts && \
    cp qwentts/build/tts-server /opt/qwentts/ && \
    cp -a qwentts/build/libggml*.so* /opt/qwentts/ && \
    test -x /opt/qwentts/tts-server && \
    ls /opt/qwentts/libggml-cpu-*.so* >/dev/null && \
    strip /opt/qwentts/tts-server && \
    find /opt/qwentts -type f -name 'libggml*.so*' -exec strip --strip-unneeded {} + && \
    cp qwentts/LICENSE /usr/share/licenses/qwentts/qwentts.cpp.LICENSE && \
    cp qwentts/ggml/LICENSE /usr/share/licenses/qwentts/ggml.LICENSE && \
    cp qwentts/vendor/cpp-httplib/LICENSE /usr/share/licenses/qwentts/cpp-httplib.LICENSE && \
    sed -n '1,\|\*/|p' qwentts/vendor/yyjson/yyjson.h > /usr/share/licenses/qwentts/yyjson.LICENSE

# Stage 5: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg curl libatomic1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# sherpa-onnx shared libraries from vendor
COPY vendor/sherpa-onnx/lib/libsherpa-onnx-c-api.so /usr/lib/
COPY vendor/sherpa-onnx/lib/libsherpa-onnx-cxx-api.so /usr/lib/
COPY vendor/sherpa-onnx/lib/libonnxruntime.so /usr/lib/
RUN ldconfig

# qwentts TTS server — spawned by ox-whisper as a child process. All
# libggml*.so must sit next to the executable: ggml_backend_load_best()
# scans the executable's directory (/proc/self/exe) for libggml-cpu-*.so
# modules and dlopen's the best-matching ARM ISA variant at startup.
# The ld.so.conf.d entry lets the same dir resolve the binary's link-time
# deps (libggml.so.0, libggml-base.so.0) without LD_LIBRARY_PATH.
COPY --from=qwentts-build /opt/qwentts/ /opt/qwentts/
COPY --from=qwentts-build /usr/share/licenses/qwentts/ /usr/share/licenses/qwentts/
RUN echo "/opt/qwentts" > /etc/ld.so.conf.d/qwentts.conf && ldconfig

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
