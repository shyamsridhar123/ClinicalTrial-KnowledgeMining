#!/usr/bin/env bash
#
# Background job runner for relation extraction
# Usage: 
#   ./scripts/run_extraction_background.sh NCT03799627      # Single trial
#   ./scripts/run_extraction_background.sh --all            # All trials
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Logging setup
LOG_DIR="logs/relation_extraction"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/extraction_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/extraction.pid"
STATUS_FILE="$LOG_DIR/extraction_status.json"

# Check if extraction is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "❌ Extraction already running with PID $OLD_PID"
        echo "   Check status: ./scripts/monitor_extraction.sh"
        echo "   View logs: tail -f $LOG_DIR/extraction_*.log"
        exit 1
    else
        echo "⚠️  Removing stale PID file"
        rm "$PID_FILE"
    fi
fi

# Parse arguments
NCT_ID=""
ALL_TRIALS=false
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_TRIALS=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        NCT*)
            NCT_ID="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [NCT_ID | --all] [--limit N]"
            exit 1
            ;;
    esac
done

# Build command
CMD="pixi run -- python scripts/extract_relations_step1.py --verbose"

if [ "$ALL_TRIALS" = true ]; then
    CMD="$CMD --all-trials"
    TASK="all trials"
elif [ -n "$NCT_ID" ]; then
    CMD="$CMD --nct-id $NCT_ID"
    TASK="$NCT_ID"
else
    echo "❌ Must specify either NCT_ID or --all"
    echo "Usage: $0 [NCT_ID | --all] [--limit N]"
    exit 1
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Initialize status file
cat > "$STATUS_FILE" <<EOF
{
  "status": "starting",
  "task": "$TASK",
  "started_at": "$(date -Iseconds)",
  "pid": null,
  "log_file": "$LOG_FILE",
  "chunks_processed": 0,
  "relations_extracted": 0
}
EOF

echo "🚀 Starting relation extraction for $TASK"
echo "   Log file: $LOG_FILE"
echo "   PID file: $PID_FILE"
echo ""
echo "Monitor progress:"
echo "  ./scripts/monitor_relation_extraction.sh"
echo ""
echo "View logs:"
echo "  tail -f $LOG_FILE"
echo ""

# Run in background with nohup
nohup bash -c "
    $CMD 2>&1 | tee '$LOG_FILE'
    EXIT_CODE=\${PIPESTATUS[0]}
    
    # Update status on completion
    if [ \$EXIT_CODE -eq 0 ]; then
        STATUS='completed'
    else
        STATUS='failed'
    fi
    
    cat > '$STATUS_FILE' <<INNER_EOF
{
  \"status\": \"\$STATUS\",
  \"task\": \"$TASK\",
  \"started_at\": \"$(date -Iseconds)\",
  \"completed_at\": \"\$(date -Iseconds)\",
  \"exit_code\": \$EXIT_CODE,
  \"log_file\": \"$LOG_FILE\"
}
INNER_EOF
    
    rm -f '$PID_FILE'
" > /dev/null 2>&1 &

# Save PID
echo $! > "$PID_FILE"

# Update status with PID
python3 -c "
import json
with open('$STATUS_FILE', 'r') as f:
    status = json.load(f)
status['status'] = 'running'
status['pid'] = $(cat "$PID_FILE")
with open('$STATUS_FILE', 'w') as f:
    json.dump(status, f, indent=2)
"

echo "✅ Background job started with PID $(cat "$PID_FILE")"
echo ""
