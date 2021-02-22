#!/usr/bin/env bash
set -Eeuxo pipefail

cargo test --release -- --include-ignored -Z unstable-options
cargo test --manifest-path common/Cargo.toml --features no-assert no_assert
cargo test --manifest-path dark/Cargo.toml --features no-assert no_assert
cargo test --manifest-path sonic/Cargo.toml --features no-assert no_assert
