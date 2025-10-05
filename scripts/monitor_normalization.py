#!/usr/bin/env python3
"""Monitor normalization progress in real-time."""

import psycopg
import time
import sys

def get_stats():
    conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM docintel.entities WHERE normalized_id IS NOT NULL')
    normalized = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM docintel.entities')
    total = cursor.fetchone()[0]
    
    conn.close()
    return normalized, total

print("🔥 NORMALIZATION MONITOR")
print("=" * 60)
print("Press Ctrl+C to exit")
print("")

last_normalized = 0
last_time = time.time()

try:
    while True:
        normalized, total = get_stats()
        remaining = total - normalized
        progress_pct = (normalized / total * 100) if total > 0 else 0
        
        # Calculate rate
        current_time = time.time()
        elapsed = current_time - last_time
        entities_delta = normalized - last_normalized
        
        if elapsed > 0 and entities_delta > 0:
            rate = entities_delta / elapsed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_hours = eta_seconds / 3600
        else:
            rate = 0
            eta_hours = 0
        
        # Update display
        print(f"\r📊 {normalized:,}/{total:,} ({progress_pct:.2f}%) | "
              f"Rate: {rate:.1f} ent/s | "
              f"ETA: {eta_hours:.1f}h | "
              f"Remaining: {remaining:,}    ", end='', flush=True)
        
        last_normalized = normalized
        last_time = current_time
        
        time.sleep(5)  # Update every 5 seconds
        
except KeyboardInterrupt:
    print("\n\n✅ Monitoring stopped")
    sys.exit(0)
