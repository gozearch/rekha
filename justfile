default: test

# Run all tests (excluding standalone crates)
test:
    cargo test --workspace --exclude rekha-bench

# Run lints
lint:
    cargo fmt --all --check
    cargo clippy --workspace --exclude rekha-bench -- -D warnings

# Fix formatting automatically
fix:
    cargo fmt --all

# Run coverage (requires nightly + cargo-llvm-cov)
coverage:
    cargo llvm-cov --workspace --exclude rekha-bench --lcov --output-path lcov.info

# Open coverage report in browser
coverage-html:
    cargo llvm-cov --workspace --exclude rekha-bench --html --output-path target/coverage
    open target/coverage/index.html

# Build everything (excluding standalone crates)
build:
    cargo build --workspace --exclude rekha-bench

# Build release
release-build:
    cargo build --release --workspace --exclude rekha-bench

# Check compilation (fast)
check:
    cargo check --workspace --exclude rekha-bench

# Run a specific test by name
test-name name:
    cargo test $(name) --workspace --exclude rekha-bench

# ── Docker ────────────────────────────────────────────────

# Build Docker image
docker-build:
    docker build -t rekha:latest .

# Start 3-node cluster
docker-up:
    docker compose up -d

# Stop cluster
docker-down:
    docker compose down

# Stop cluster and delete volumes
docker-down-clean:
    docker compose down -v

# Follow logs from all nodes
docker-logs:
    docker compose logs -f

# Execute a rekha command on a specific node
docker-exec node cmd:
    docker compose exec {{node}} rekha {{cmd}}
