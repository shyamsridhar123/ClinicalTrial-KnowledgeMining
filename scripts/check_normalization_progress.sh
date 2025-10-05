#!/bin/bash
# Check normalization progress

echo "📊 NORMALIZATION PROGRESS CHECK"
echo "================================"
echo ""

# Check if process is running
if ps -p 54644 > /dev/null 2>&1; then
    echo "✅ Process is RUNNING (PID: 54644)"
else
    echo "⚠️  Process has stopped or completed"
fi

echo ""
echo "📈 LATEST LOG ENTRIES:"
tail -10 normalization_full.log

echo ""
echo "📊 DATABASE STATUS:"
pixi run -- python -c "
import psycopg
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cursor = conn.cursor()

cursor.execute('''
    SELECT 
        COUNT(*) as total,
        COUNT(normalized_id) as normalized,
        COUNT(*) - COUNT(normalized_id) as remaining
    FROM docintel.entities
''')
row = cursor.fetchone()

cursor.execute('''
    SELECT normalized_source, COUNT(*) as count
    FROM docintel.entities
    WHERE normalized_id IS NOT NULL
    GROUP BY normalized_source
    ORDER BY count DESC
''')
vocab_stats = cursor.fetchall()

print(f'   Total entities: {row[0]:,}')
print(f'   Normalized: {row[1]:,} ({row[1]/row[0]*100:.2f}%)')
print(f'   Remaining: {row[2]:,} ({row[2]/row[0]*100:.2f}%)')
print(f'')
print(f'   By vocabulary:')
for vocab_row in vocab_stats:
    print(f'      {vocab_row[0]:10} : {vocab_row[1]:,}')

conn.close()
"

echo ""
echo "📝 LOG FILE SIZE:"
ls -lh normalization_full.log | awk '{print "   " $5}'

echo ""
echo "⏱️  ESTIMATED TIME (assuming 1 entity/sec):"
echo "   ~45 hours for 161,902 entities"
