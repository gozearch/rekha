# ── Stage 1: Build ────────────────────────────────────────
FROM rust:slim-bookworm AS builder

RUN apt-get update && \
    apt-get install -y protobuf-compiler clang libclang-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN LIBCLANG_PATH=$(find /usr/lib -name 'libclang-*.so*' -type f 2>/dev/null | head -1 | xargs dirname) \
    cargo build --package rekha-cli --release && \
    strip target/release/rekha

# ── Stage 2: Runtime ──────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rekha /usr/local/bin/rekha

EXPOSE 50051
ENTRYPOINT ["rekha"]
CMD ["server", "--config", "/etc/rekha/config.yaml"]
