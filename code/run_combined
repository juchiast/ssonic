#!/usr/bin/env bash
set -Eeuxo pipefail

cargo build --release --manifest-path dark/Cargo.toml --features no-assert --bin combined
LOG=trace cargo run --release --manifest-path dark/Cargo.toml --features no-assert --bin combined -- 100
LOG=trace cargo run --release --manifest-path dark/Cargo.toml --features no-assert --bin combined -- 1000
LOG=trace cargo run --release --manifest-path dark/Cargo.toml --features no-assert --bin combined -- 10000
