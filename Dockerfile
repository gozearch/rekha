# Stage 1: Builder
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

RUN cargo build --release --bin rekha

# Stage 2: Runtime (must match builder's trixie for libmvec + CXXABI compat)
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r rekha && useradd -r -g rekha rekha

COPY --from=builder /app/target/release/rekha /usr/local/bin/rekha

RUN mkdir -p /data && chown rekha:rekha /data
USER rekha

VOLUME /data
EXPOSE 8000

ENV REKHA_DATA_DIR=/data
ENV REKHA_HOST=0.0.0.0
ENV REKHA_PORT=8000

ENTRYPOINT ["rekha"]
CMD ["serve"]
