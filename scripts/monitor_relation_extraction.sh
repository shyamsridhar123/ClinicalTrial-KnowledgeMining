#!/usr/bin/env bash
#
# Monitor relation extraction progress in real-time
# Usage: ./scripts/monitor_relation_extraction.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs/relation_extraction"
PID_FILE="$LOG_DIR/extraction.pid"
STATUS_FILE="$LOG_DIR/extraction_status.json"

# Check if extraction is running
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No extraction job running (no PID file found)"
    echo ""
    echo "Start extraction:"
    echo "  ./scripts/run_extraction_background.sh NCT03799627"
    echo "  ./scripts/run_extraction_background.sh --all"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  Job finished or crashed (PID $PID not running)"
    
    if [ -f "$STATUS_FILE" ]; then
        echo ""
        echo "Final status:"
        cat "$STATUS_FILE" | python3 -m json.tool
    fi
    
    rm -f "$PID_FILE"
    exit 0
fi

echo "🔍 Monitoring Relation Extraction"
echo "=================================="
echo ""
echo "PID: $PID"
echo "Status file: $STATUS_FILE"
echo ""

if [ -f "$STATUS_FILE" ]; then
    python3 -c "
import json
with open('$STATUS_FILE') as f:
    status = json.load(f)
    
print(f\"Task: {status.get('task', 'unknown')}\")
print(f\"Started: {status.get('started_at', 'unknown')}\")
print(f\"Status: {status.get('status', 'unknown')}\")
print()
"
fi

# Real-time database stats
echo "📊 Database Statistics (Live)"
echo "=============================="
pixi run -- python -c "
import psycopg
from datetime import datetime, timedelta

conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()

# Total relations
cur.execute('SELECT COUNT(*) FROM docintel.relations')
total_relations = cur.fetchone()[0]

# Relations added in last hour
cur.execute('''
    SELECT COUNT(*) 
    FROM docintel.relations 
    WHERE created_at > NOW() - INTERVAL \'1 hour\'
''')
recent_relations = cur.fetchone()[0]

# Relations by predicate (top 10)
cur.execute('''
    SELECT predicate, COUNT(*) as count
    FROM docintel.relations
    GROUP BY predicate
    ORDER BY count DESC
    LIMIT 10
''')
predicates = cur.fetchall()

# Entities linked to relations
cur.execute('''
    SELECT COUNT(DISTINCT subject_entity_id) + COUNT(DISTINCT object_entity_id) as entity_count
    FROM docintel.relations
''')
linked_entities = cur.fetchone()[0]

print(f'Total Relations: {total_relations:,}')
print(f'  Added in last hour: {recent_relations:,}')
print(f'  Unique entities involved: {linked_entities:,}')
print()

if predicates:
    print('Top Predicates:')
    for pred, count in predicates:
        print(f'  {pred:30s}: {count:,}')
else:
    print('  (No relations yet)')

conn.close()
"

echo ""
echo "📝 Recent Log Entries"
echo "====================="

# Find most recent log file
LATEST_LOG=$(ls -t "$LOG_DIR"/extraction_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "Log: $LATEST_LOG"
    echo ""
    tail -n 20 "$LATEST_LOG"
else
    echo "No log files found"
fi

echo ""
echo "=================================="
echo ""
echo "Commands:"
echo "  Watch this: watch -n 5 ./scripts/monitor_relation_extraction.sh"
echo "  View logs:  tail -f $LATEST_LOG"
echo "  Stop job:   kill $PID"
echo ""
