#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="${TEST_ROOT:-/tmp/gpu-watchdog-alert-test-$$}"
CONFIG_PATH="$TEST_ROOT/config.yaml"
STATE_PATH="$TEST_ROOT/state.json"
HEARTBEAT_MARKER="$TEST_ROOT/training-heartbeat"
SUCCESS_MARKER="$TEST_ROOT/training-success"
FAILURE_MARKER="$TEST_ROOT/training-failed"
KEEPALIVE_LOG="$TEST_ROOT/keepalive.log"

TRAIN_PID=""

require_slack_webhook() {
  if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
    echo "SLACK_WEBHOOK_URL is not set."
    echo "Run: export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'"
    exit 1
  fi
}

prepare_test_root() {
  rm -rf "$TEST_ROOT"
  mkdir -p "$TEST_ROOT"
}

write_config() {
  local keepalive_mode="$1"
  local keepalive_command

  case "$keepalive_mode" in
    success)
      keepalive_command='
    - python3
    - -c
    - "import time; time.sleep(600)"'
      ;;
    failure)
      keepalive_command='
    - python3
    - -c
    - "import sys; sys.stderr.write(\"keepalive test failure\n\"); sys.exit(1)"'
      ;;
    *)
      echo "Unknown keepalive mode: $keepalive_mode"
      exit 1
      ;;
  esac

  cat > "$CONFIG_PATH" <<EOF
check_interval_seconds: 1
idle_threshold_minutes: 9999
state_path: "$STATE_PATH"
session_name: alert-test-session

training_jobs:
  - name: alert-test-training
    heartbeat_path: "$HEARTBEAT_MARKER"
    heartbeat_timeout_seconds: 2
    success_marker_path: "$SUCCESS_MARKER"
    failure_marker_path: "$FAILURE_MARKER"

gpu:
  nvidia_smi_path: nvidia-smi
  idle_utilization_threshold: 1

keepalive:
  command:$keepalive_command
  log_path: "$KEEPALIVE_LOG"
  start_grace_seconds: 1

slack:
  webhook_url:
  webhook_env_var: SLACK_WEBHOOK_URL
  request_timeout_seconds: 10
  alert_timezone: America/Toronto
  keepalive_failure_alert_interval_seconds: 3600
  notify_on_idle_keepalive: false
  notify_on_training_completed: true
EOF
}

start_fake_training() {
  python3 - "$HEARTBEAT_MARKER" <<'PY' &
import json
import os
import sys
import time
from pathlib import Path

heartbeat = Path(sys.argv[1])
heartbeat.parent.mkdir(parents=True, exist_ok=True)
step = 0
while True:
    payload = {
        "timestamp": int(time.time()),
        "epoch": None,
        "step": step,
        "status": "training",
    }
    tmp = heartbeat.with_name(f"{heartbeat.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(heartbeat)
    step += 1
    time.sleep(0.5)
PY
  TRAIN_PID="$!"
  sleep 0.3
  echo "Started fake training process: pid=$TRAIN_PID heartbeat=$HEARTBEAT_MARKER"
}

stop_fake_training() {
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
    echo "Stopped fake training process: pid=$TRAIN_PID"
  fi
}

run_watchdog_once() {
  python3 "$REPO_ROOT/main.py" --config "$CONFIG_PATH" --once
}

cleanup_test_processes() {
  stop_fake_training

  if [[ -f "$STATE_PATH" ]]; then
    local keepalive_pid
    keepalive_pid="$(
      python3 - "$STATE_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    print(json.loads(path.read_text()).get("keepalive_pid") or "")
except Exception:
    print("")
PY
    )"

    if [[ -n "$keepalive_pid" ]] && kill -0 "$keepalive_pid" 2>/dev/null; then
      kill "$keepalive_pid" 2>/dev/null || true
      wait "$keepalive_pid" 2>/dev/null || true
      echo "Stopped fake keepalive process: pid=$keepalive_pid"
    fi
  fi
}

print_done() {
  local expected="$1"
  echo
  echo "Done. Check Slack for: $expected"
  echo "Test files were written under: $TEST_ROOT"
}
