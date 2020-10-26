#!/usr/bin/bash
set -Eeuxo pipefail

cargo build --release --features no-assert --bin powmod
LOG=trace cargo run --release --features no-assert --bin powmod -- 8
LOG=trace cargo run --release --features no-assert --bin powmod -- 16
LOG=trace cargo run --release --features no-assert --bin powmod -- 32
LOG=trace cargo run --release --features no-assert --bin powmod -- 64
