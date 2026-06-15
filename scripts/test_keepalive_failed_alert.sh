#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_alert_test_common.sh"
trap cleanup_test_processes EXIT

require_slack_webhook
prepare_test_root
write_config failure

echo "Case 3: keepalive failed to start"
echo "Step 1: run watchdog with no training process and a failing keepalive command"
run_watchdog_once

print_done "🚨 [GPU Watchdog] Keepalive Failed"

