# Stage 1: Builder
# syntax=docker/dockerfile:1
FROM rust:1.90-slim AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

ENV CARGO_PROFILE_RELEASE_DEBUG=0

RUN cargo build --release --locked --bin rekha

# Stage 2: Runtime (must match builder's trixie for libmvec + CXXABI compat)
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 rekha && \
    useradd -u 10001 -g 10001 -r -m -d /home/rekha rekha

COPY --from=builder /app/target/release/rekha /usr/local/bin/rekha

RUN mkdir -p /data && chown 10001:10001 /data
USER 10001:10001

VOLUME /data
EXPOSE 8000

ENV REKHA_DATA_DIR=/data
ENV REKHA_HOST=0.0.0.0
ENV REKHA_PORT=8000

HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["rekha"]
CMD ["serve"]
