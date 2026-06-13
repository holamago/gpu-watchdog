#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/test_training_completed_alert.sh"
echo
"$SCRIPT_DIR/test_training_failed_alert.sh"
echo
"$SCRIPT_DIR/test_keepalive_failed_alert.sh"
echo
"$SCRIPT_DIR/test_gpu_reclaim_risk_alert.sh"

echo
echo "All alert test scripts finished. Check Slack for four alert messages."

