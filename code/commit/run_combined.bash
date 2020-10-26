#!/usr/bin/env bash
set -Eeuxo pipefail

LOG=trace cargo run --release --features no-assert --bin combined -- 100
LOG=trace cargo run --release --features no-assert --bin combined -- 1000
LOG=trace cargo run --release --features no-assert --bin combined -- 10000
