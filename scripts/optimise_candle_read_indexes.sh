#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${AIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CANDLES_DIR="${AI_QUANT_CANDLES_DB_DIR:-$ROOT_DIR/candles_dbs}"

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 is required" >&2
  exit 1
fi

if [[ ! -d "$CANDLES_DIR" ]]; then
  echo "candle DB directory not found: $CANDLES_DIR" >&2
  exit 1
fi

shopt -s nullglob
dbs=("$CANDLES_DIR"/candles_*.db)
if (( ${#dbs[@]} == 0 )); then
  echo "no candle DBs found in $CANDLES_DIR" >&2
  exit 1
fi

for db in "${dbs[@]}"; do
  case "$(basename "$db")" in
    *-wal|*-shm) continue ;;
  esac

  echo "optimising read indexes: $db"
  sqlite3 "$db" >/dev/null <<'SQL'
PRAGMA busy_timeout = 10000;
PRAGMA journal_mode = WAL;
CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_close
  ON candles(symbol, interval, COALESCE(t_close, t));
CREATE INDEX IF NOT EXISTS idx_candles_interval_close_symbol
  ON candles(interval, COALESCE(t_close, t), symbol);
ANALYZE;
PRAGMA wal_checkpoint(PASSIVE);
SQL
done
