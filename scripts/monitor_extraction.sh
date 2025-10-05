#!/bin/bash
# Monitor entity extraction progress in real-time

echo "🔍 Monitoring Entity Extraction Progress"
echo "=========================================="
echo ""

# Check if extraction process is running
EXTRACTION_PID=$(pgrep -f "docintel.extract")

if [ -z "$EXTRACTION_PID" ]; then
    echo "❌ Extraction process not running"
    exit 1
fi

echo "✅ Extraction running (PID: $EXTRACTION_PID)"
echo ""

# Monitor log file and database in a loop
while kill -0 $EXTRACTION_PID 2>/dev/null; do
    clear
    echo "🔍 Entity Extraction Progress Monitor"
    echo "=========================================="
    echo "Process ID: $EXTRACTION_PID"
    echo "Start Time: $(ps -p $EXTRACTION_PID -o lstart=)"
    echo "CPU/MEM: $(ps -p $EXTRACTION_PID -o %cpu,%mem | tail -1)"
    echo ""
    
    # Show latest log entries
    echo "📋 Latest Log Entries:"
    echo "----------------------------------------"
    tail -20 extraction_with_source_chunk_id.log 2>/dev/null | grep -E "(Processing|Extracted|Created|ERROR|✅|❌|entities|relations|📦)" || echo "Waiting for log output..."
    echo ""
    
    # Check database counts
    echo "📊 Database Status:"
    echo "----------------------------------------"
    pixi run -- python -c "
import psycopg
from psycopg.rows import dict_row
try:
    conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel', connect_timeout=2)
    cur = conn.cursor(row_factory=dict_row)
    
    # Entity counts
    cur.execute('SELECT COUNT(*) as count FROM docintel.entities')
    entity_count = cur.fetchone()['count']
    
    # Check source_chunk_id population
    cur.execute('SELECT COUNT(*) as count FROM docintel.entities WHERE source_chunk_id IS NOT NULL')
    with_source = cur.fetchone()['count']
    
    # Meta-graphs
    cur.execute('SELECT COUNT(*) as count FROM docintel.meta_graphs')
    graph_count = cur.fetchone()['count']
    
    pct = (with_source / entity_count * 100) if entity_count > 0 else 0
    
    print(f'Entities extracted: {entity_count:,}')
    print(f'With source_chunk_id: {with_source:,} ({pct:.1f}%)')
    print(f'Meta-graphs: {graph_count:,}')
    
    conn.close()
except Exception as e:
    print(f'DB check failed: {e}')
" 2>/dev/null
    
    echo ""
    echo "Refreshing in 5 seconds... (Ctrl+C to stop monitoring)"
    sleep 5
done

echo ""
echo "✅ Extraction process completed!"
echo ""
echo "📊 Final Statistics:"
echo "----------------------------------------"
pixi run -- python -c "
import psycopg
from psycopg.rows import dict_row
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor(row_factory=dict_row)

cur.execute('SELECT COUNT(*) as count FROM docintel.entities')
entity_count = cur.fetchone()['count']

cur.execute('SELECT COUNT(*) as count FROM docintel.entities WHERE source_chunk_id IS NOT NULL')
with_source = cur.fetchone()['count']

cur.execute('SELECT COUNT(*) as count FROM docintel.meta_graphs')
graph_count = cur.fetchone()['count']

cur.execute('SELECT COUNT(DISTINCT source_chunk_id) as count FROM docintel.entities WHERE source_chunk_id IS NOT NULL')
unique_chunks = cur.fetchone()['count']

pct = (with_source / entity_count * 100) if entity_count > 0 else 0

print(f'Total entities: {entity_count:,}')
print(f'With source_chunk_id: {with_source:,} ({pct:.1f}%)')
print(f'Unique chunks: {unique_chunks:,}')
print(f'Meta-graphs: {graph_count:,}')

conn.close()
"

echo ""
echo "📄 Full log: extraction_with_source_chunk_id.log"
