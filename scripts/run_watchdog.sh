#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deprecated: use scripts/run/watchdog.sh --nohup instead." >&2
exec "$SCRIPT_DIR/run/watchdog.sh" --nohup "$@"
