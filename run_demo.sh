#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$DEMO_DIR"

if [[ ! -x "$DEMO_DIR/.venv/bin/python" ]]; then
    echo "Missing .venv. Install the project first: uv sync" >&2
    exit 1
fi

exec "$DEMO_DIR/.venv/bin/python" -m tiny_doom_defender.scripts.play_doom_visual \
    --overlay \
    --ckpt "$DEMO_DIR/tiny-doom-defender" \
    --seed 10000 \
    --episodes 1 \
    --mode argmax \
    --device cpu \
    --fps 35 \
    --hold-final
