#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_alert_test_common.sh"
trap cleanup_test_processes EXIT

prepare_test_root
write_config failure

echo "Case 5: keepalive failure alert is throttled"
echo "Step 1: record the first keepalive failure alert timestamp"
run_watchdog_once

first_alert_epoch="$(
  python3 - "$STATE_PATH" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
print(state.get("last_keepalive_failure_alert_epoch") or "")
PY
)"

if [[ -z "$first_alert_epoch" ]]; then
  echo "Expected first keepalive failure alert epoch to be recorded." >&2
  exit 1
fi

echo "Step 2: run the same failure again and confirm the alert timestamp is unchanged"
run_watchdog_once

second_alert_epoch="$(
  python3 - "$STATE_PATH" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
print(state.get("last_keepalive_failure_alert_epoch") or "")
PY
)"

if [[ "$first_alert_epoch" != "$second_alert_epoch" ]]; then
  echo "Expected keepalive failure alert to be throttled." >&2
  echo "first=$first_alert_epoch second=$second_alert_epoch" >&2
  exit 1
fi

echo "Done. Keepalive failure alert throttle is working."
