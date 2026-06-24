#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_alert_test_common.sh"
trap cleanup_test_processes EXIT

require_slack_webhook
prepare_test_root
write_config success

echo "Case 1: training completed successfully"
start_fake_training

echo "Step 1: record that training is alive"
run_watchdog_once

echo "Step 2: mark training as successful and stop it"
touch "$SUCCESS_MARKER"
stop_fake_training

echo "Step 3: detect completion and send Slack alert"
run_watchdog_once

print_done "✅ [GPU Watchdog] Training Completed"
