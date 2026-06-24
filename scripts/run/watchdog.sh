#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run/watchdog.sh [--nohup|--foreground] [--env-file PATH] [--config PATH]

Options:
  --nohup          Start watchdog in the background with nohup.
  --foreground     Run watchdog in the foreground. This is the default.
  --env-file PATH  Load a session-specific env file. Defaults to .env.
  --config PATH    Watchdog config path. Defaults to config.yaml.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${GPU_WATCHDOG_ENV_FILE:-$REPO_ROOT/.env}"
CONFIG_PATH="${GPU_WATCHDOG_CONFIG:-$REPO_ROOT/config.yaml}"
MODE="foreground"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nohup)
      MODE="nohup"
      shift
      ;;
    --foreground)
      MODE="foreground"
      shift
      ;;
    --env-file)
      ENV_FILE="${2:?Missing value for --env-file}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:?Missing value for --config}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

: "${GPU_WATCHDOG_NAME:?Set GPU_WATCHDOG_NAME in the environment or env file.}"

cd "$REPO_ROOT"

SAFE_GPU_WATCHDOG_NAME="$(
  python3 - "$GPU_WATCHDOG_NAME" <<'PY'
import sys

from gpu_watchdog.config import path_safe_session_name

print(path_safe_session_name(sys.argv[1]))
PY
)"

LOG_DIR="${GPU_WATCHDOG_LOG_DIR:-$REPO_ROOT/logs}"
WATCHDOG_LOG="$LOG_DIR/$SAFE_GPU_WATCHDOG_NAME-watchdog.log"

mkdir -p "$LOG_DIR"

if [[ "$MODE" == "nohup" ]]; then
  nohup python3 main.py --config "$CONFIG_PATH" > "$WATCHDOG_LOG" 2>&1 &
  echo "Started gpu-watchdog pid=$!"
  echo "Session: $GPU_WATCHDOG_NAME"
  echo "Watchdog log: $WATCHDOG_LOG"
  exit 0
fi

echo "Running gpu-watchdog in the foreground."
echo "Session: $GPU_WATCHDOG_NAME"
echo "Use --nohup to keep it running after the shell exits."
exec python3 main.py --config "$CONFIG_PATH"
