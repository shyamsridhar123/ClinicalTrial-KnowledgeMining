#!/bin/bash
# Quick progress check

while true; do
    clear
    echo "🔥 NORMALIZATION PROGRESS"
    echo "=================================================="
    date
    echo ""
    
    # Check process
    if ps aux | grep -q "[p]ython scripts/normalize_entities.py"; then
        echo "✅ Process: RUNNING"
    else
        echo "❌ Process: STOPPED"
        break
    fi
    
    # Get database stats
    pixi run -- python -c "
import psycopg
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM docintel.entities WHERE normalized_id IS NOT NULL')
normalized = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM docintel.entities')
total = cursor.fetchone()[0]
print(f'📊 {normalized:,} / {total:,} ({normalized/total*100:.2f}%)')
conn.close()
" 2>/dev/null
    
    echo ""
    echo "📝 Latest log entries:"
    tail -5 normalization_optimized.log | sed 's/^/   /'
    
    echo ""
    echo "Press Ctrl+C to stop monitoring..."
    sleep 10
done
