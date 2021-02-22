#!/usr/bin/env bash
set -Eeuxo pipefail

cargo clippy --all-targets
cargo clippy --all-targets --manifest-path common/Cargo.toml --features no-assert
cargo clippy --all-targets --manifest-path dark/Cargo.toml --features no-assert
cargo clippy --all-targets --manifest-path sonic/Cargo.toml --features no-assert
