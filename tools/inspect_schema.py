import sqlite3

DB_PAPER = '/home/fol2hk/.openclaw/workspace/dev/ai_quant/trading_engine.db'
DB_LIVE = '/home/fol2hk/.openclaw/workspace/dev/ai_quant/trading_engine_live.db'

def print_schema(db_path, table_name):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    rows = c.fetchall()
    print(f"Schema for {table_name} in {db_path}:")
    for row in rows:
        print(row)
    conn.close()

print_schema(DB_PAPER, 'trades')
print_schema(DB_LIVE, 'position_state')
print_schema(DB_LIVE, 'oms_intents')
