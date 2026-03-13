#!/usr/bin/env bash
set -euo pipefail

# Apple-to-apple sweep: all intervals scoped to the shortest DB (3m) time range.

BT="./backtester/target/release/mei-backtester"
OUTDIR="sweep_full/results_aligned"
mkdir -p "$OUTDIR"

# ── Step 1: Determine the common time range from the shortest-coverage DB (3m) ──
RANGE=$(~/.local/bin/sqlite3 candles_dbs/candles_3m.db "SELECT MIN(t), MAX(t) FROM candles;")
FROM_TS=$(echo "$RANGE" | cut -d'|' -f1)
TO_TS=$(echo "$RANGE" | cut -d'|' -f2)

echo "=============================================="
echo " Apple-to-Apple Sweep"
echo " Scoped to 3m range: $FROM_TS .. $TO_TS"
FROM_S=$((FROM_TS / 1000))
TO_S=$((TO_TS / 1000))
DAYS="$(awk "BEGIN { printf \"%.1f\", (${TO_TS} - ${FROM_TS}) / 1000 / 86400 }")"
echo " Period: $(date -u -d "@${FROM_S}" '+%Y-%m-%d %H:%M') -> $(date -u -d "@${TO_S}" '+%Y-%m-%d %H:%M') (${DAYS} days)"

PHASES=(
  "p01_core_trade:3780"
  "p02_entry_thresholds:480"
  "p03_entry_toggles:256"
  "p04_execution_gates:5120"
  "p05_trailing_exits:900"
  "p06_smart_exits:144"
  "p07_market_regime:324"
  "p08_indicator_windows:972"
  "p09_pullback_drift:768"
  "p10_dynamic_sizing:1296"
  "p11_pyramiding:162"
  "p12_low_conf_exits:144"
  "p13_rsi_exit_extended:243"
  "p14_stoch_rsi:96"
  "p15_momentum_tp:144"
  "p16_vol_sizing:108"
  "p17_entry_confidence:144"
)

INTERVALS=("3m" "5m" "15m" "30m")

TOTAL_RUNS=0
START_TIME=$(date +%s)

echo " 17 phases × 4 intervals"
echo " Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="
echo ""

for interval in "${INTERVALS[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Entry/Exit interval: $interval (scoped to 3m range)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for phase_info in "${PHASES[@]}"; do
    phase="${phase_info%%:*}"
    expected="${phase_info##*:}"
    spec="sweep_full/${phase}.yaml"
    outfile="${OUTDIR}/${phase}_${interval}.jsonl"

    echo -n "  [${phase}] ${expected} combos @ ${interval} ... "
    phase_start=$(date +%s)

    $BT sweep \
      --sweep-spec "$spec" \
      --interval 1h \
      --entry-interval "$interval" \
      --exit-interval "$interval" \
      --from-ts "$FROM_TS" \
      --to-ts "$TO_TS" \
      --output "$outfile" \
      --top-n 0 \
      2>/dev/null

    phase_end=$(date +%s)
    phase_secs=$((phase_end - phase_start))
    combos=$(wc -l < "$outfile")
    rate=$(echo "scale=1; $combos / ($phase_secs + 1)" | bc 2>/dev/null || echo "?")
    echo "${combos} done in ${phase_secs}s (${rate}/s)"

    TOTAL_RUNS=$((TOTAL_RUNS + combos))
  done
  echo ""
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=============================================="
echo " Sweep complete!"
echo " Total runs:   $TOTAL_RUNS"
echo " Elapsed:      ${ELAPSED}s"
echo "=============================================="
echo ""

echo "Combined CSV export was retired with the zero-Python repository cutover."
echo "Use jq against ${OUTDIR}/*.jsonl if you need ad-hoc aggregation."
