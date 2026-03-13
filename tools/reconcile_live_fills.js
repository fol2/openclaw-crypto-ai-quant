#!/usr/bin/env node

const { execFileSync } = require("node:child_process");
const { writeFileSync, mkdtempSync } = require("node:fs");
const { join } = require("node:path");
const { tmpdir } = require("node:os");

function usage() {
  return [
    "Usage:",
    "  tools/reconcile_live_fills.js --db <sqlite> --user <address> --start <ms> --end <ms> [--apply]",
    "",
    "Examples:",
    "  tools/reconcile_live_fills.js --db trading_engine_v8_live.db --user 0x... --start 1771126743410 --end 1773401649000",
    "  tools/reconcile_live_fills.js --apply --db trading_engine_v8_live.db --user 0x... --start 1771126743410 --end 1773401649000",
  ].join("\n");
}

function parseArgs(argv) {
  const args = {
    apply: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      process.stdout.write(`${usage()}\n`);
      process.exit(0);
    }
    if (arg === "--apply") {
      args.apply = true;
      continue;
    }
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected argument: ${arg}`);
    }
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (value == null || value.startsWith("--")) {
      throw new Error(`missing value for --${key}`);
    }
    args[key] = value;
    i += 1;
  }
  for (const key of ["db", "user", "start", "end"]) {
    if (!args[key]) {
      throw new Error(`--${key} is required`);
    }
  }
  args.start = Number(args.start);
  args.end = Number(args.end);
  if (!Number.isFinite(args.start) || !Number.isFinite(args.end)) {
    throw new Error("--start/--end must be numbers");
  }
  return args;
}

function sh(command, args, options = {}) {
  return execFileSync(command, args, {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    ...options,
  });
}

function sqlJson(db, sql) {
  const out = sh("sqlite3", ["-json", db, sql]);
  return out.trim() ? JSON.parse(out) : [];
}

function sqlCount(db, sql) {
  return Number(sqlJson(db, sql)[0]?.count || 0);
}

function fetchJson(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(async (res) => {
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} from ${url}`);
    }
    return res.json();
  });
}

function keyOf(fillHash, fillTid) {
  return `${fillHash || ""}::${fillTid ?? ""}`;
}

function sqlValue(value) {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "NULL";
    return String(value);
  }
  if (typeof value === "boolean") return value ? "1" : "0";
  return `'${String(value).replaceAll("'", "''")}'`;
}

function msToIso(tsMs) {
  return new Date(tsMs).toISOString();
}

function normaliseSymbol(value) {
  return String(value || "").trim().toUpperCase();
}

function parseNum(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseFill(fill) {
  const symbol = normaliseSymbol(fill.coin || fill.symbol);
  const price = parseNum(fill.px ?? fill.price);
  const size = parseNum(fill.sz ?? fill.size);
  const tsMs = Number(fill.time ?? fill.timestamp);
  const dir = String(fill.dir || "").trim();
  const startPosition = parseNum(fill.startPosition) ?? 0;
  const oid = fill.oid == null ? null : String(fill.oid);
  const feeUsd = parseNum(fill.fee) ?? 0;
  const pnlUsd = parseNum(fill.closedPnl) ?? 0;
  if (!symbol || !Number.isFinite(price) || price <= 0 || !Number.isFinite(size) || size <= 0) {
    throw new Error(`invalid fill payload for ${JSON.stringify(fill)}`);
  }
  const parsed = classifyDirection(dir, startPosition, size);
  if (!parsed) {
    return {
      unsupported: true,
      symbol,
      price,
      size,
      tsMs,
      dir,
      startPosition,
      oid,
      feeUsd,
      pnlUsd,
      fillHash: fill.hash || null,
      fillTid: fill.tid == null ? null : Number(fill.tid),
      cloid: fill.cloid || null,
      raw: fill,
    };
  }
  return {
    unsupported: false,
    symbol,
    price,
    size,
    tsMs,
    dir,
    startPosition,
    oid,
    feeUsd,
    pnlUsd,
    fillHash: fill.hash || null,
    fillTid: fill.tid == null ? null : Number(fill.tid),
    cloid: fill.cloid || null,
    action: parsed.action,
    posType: parsed.posType,
    side: parsed.side,
    notionalUsd: price * size,
    raw: fill,
  };
}

function classifyDirection(dirRaw, startPosition, fillSize) {
  const dir = String(dirRaw || "").trim().toLowerCase();
  if (!dir) return null;
  const posType = dir.includes("long") ? "LONG" : dir.includes("short") ? "SHORT" : null;
  if (!posType) return null;
  if (dir.startsWith("open")) {
    return {
      action: Math.abs(startPosition) < 1e-9 ? "OPEN" : "ADD",
      posType,
      side: posType === "LONG" ? "BUY" : "SELL",
    };
  }
  if (dir.startsWith("close")) {
    const endingPosition =
      posType === "LONG" ? startPosition - fillSize : startPosition + fillSize;
    return {
      action:
        posType === "LONG"
          ? endingPosition <= 1e-9
            ? "CLOSE"
            : "REDUCE"
          : endingPosition >= -1e-9
            ? "CLOSE"
            : "REDUCE",
      posType,
      side: posType === "LONG" ? "SELL" : "BUY",
    };
  }
  return null;
}

function buildIntentMaps(intents) {
  const byClientOrderId = new Map();
  const byExchangeOrderId = new Map();
  const byIntentId = new Map();
  for (const intent of intents) {
    byIntentId.set(intent.intent_id, intent);
    if (intent.client_order_id) byClientOrderId.set(intent.client_order_id, intent);
    if (intent.exchange_order_id) byExchangeOrderId.set(String(intent.exchange_order_id), intent);
  }
  return { byClientOrderId, byExchangeOrderId, byIntentId };
}

function reasonCodeFromIntent(intent) {
  if (!intent || !intent.meta_json) return null;
  try {
    const meta = JSON.parse(intent.meta_json);
    return meta?.reason_code || null;
  } catch {
    return null;
  }
}

function makeManualIntentId(fill) {
  return `manual_${String(fill.fillHash).slice(0, 12)}_${fill.fillTid}`;
}

function tradeIsManual(trade) {
  if (!trade) return false;
  return (
    trade.reason === "manual_trade" ||
    trade.reason_code === "manual_trade" ||
    trade.confidence === "MANUAL"
  );
}

function cloidKind(rawCloid) {
  const value = String(rawCloid || "").toLowerCase();
  if (!value) return "none";
  if (value.startsWith("0x6d616e5f")) return "manual";
  if (value.startsWith("0x6169715f")) return "aiq";
  return "other";
}

function fillIsManual(fill, intent) {
  if (String(intent?.intent_id || "").startsWith("manual_")) {
    const kind = cloidKind(intent?.client_order_id || fill?.cloid);
    return kind === "manual" || kind === "none";
  }
  const kind = cloidKind(fill?.cloid);
  return kind === "manual" || kind === "none";
}

function buildManualTradeMeta(fill, intentId, matchedVia) {
  return JSON.stringify({
    source: "manual_trade",
    fill: fill.raw,
    oms: {
      intent_id: intentId,
      client_order_id: fill.cloid,
      exchange_order_id: fill.oid,
      matched_via: matchedVia,
    },
    repair: {
      source: "reconcile_live_fills",
    },
  });
}

function buildAutoTradeMeta(fill, intent, matchedVia) {
  return JSON.stringify({
    source: "live_fill_reconcile",
    fill: fill.raw,
    oms: {
      intent_id: intent?.intent_id || null,
      client_order_id: intent?.client_order_id || fill.cloid || null,
      exchange_order_id: intent?.exchange_order_id || fill.oid || null,
      matched_via: matchedVia,
    },
  });
}

function buildInsertTrade(fill, intent, matchedVia) {
  const manual = fillIsManual(fill, intent);
  const reason = manual ? "manual_trade" : intent?.reason || `LIVE_FILL ${fill.dir}`;
  const reasonCode = manual ? "manual_trade" : reasonCodeFromIntent(intent);
  const confidence = manual ? "MANUAL" : intent?.confidence || "N/A";
  const metaJson = manual
    ? buildManualTradeMeta(fill, intent?.intent_id || null, matchedVia)
    : buildAutoTradeMeta(fill, intent, matchedVia);
  const feeRate = fill.notionalUsd > 0 ? fill.feeUsd / fill.notionalUsd : null;
  const leverage = intent?.leverage ?? null;
  const entryAtr = intent?.entry_atr ?? null;
  const balance = null;
  const marginUsed =
    leverage && Number(leverage) > 0 ? fill.notionalUsd / Number(leverage) : null;
  return `INSERT OR IGNORE INTO trades (
    timestamp, symbol, action, type, price, size, notional, reason, reason_code,
    confidence, pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage,
    margin_used, meta_json, fill_hash, fill_tid
  ) VALUES (
    ${sqlValue(msToIso(fill.tsMs))},
    ${sqlValue(fill.symbol)},
    ${sqlValue(fill.action)},
    ${sqlValue(fill.posType)},
    ${sqlValue(fill.price)},
    ${sqlValue(fill.size)},
    ${sqlValue(fill.notionalUsd)},
    ${sqlValue(reason)},
    ${sqlValue(reasonCode)},
    ${sqlValue(confidence)},
    ${sqlValue(fill.pnlUsd)},
    ${sqlValue(fill.feeUsd)},
    NULL,
    ${sqlValue(feeRate)},
    ${sqlValue(balance)},
    ${sqlValue(entryAtr)},
    ${sqlValue(leverage)},
    ${sqlValue(marginUsed)},
    ${sqlValue(metaJson)},
    ${sqlValue(fill.fillHash)},
    ${sqlValue(fill.fillTid)}
  );`;
}

function buildInsertOmsFill(fill, intentId, matchedVia) {
  return `INSERT OR IGNORE INTO oms_fills (
    ts_ms, symbol, intent_id, order_id, action, side, pos_type, price, size, notional,
    fee_usd, fee_token, fee_rate, pnl_usd, fill_hash, fill_tid, matched_via, raw_json
  ) VALUES (
    ${sqlValue(fill.tsMs)},
    ${sqlValue(fill.symbol)},
    ${sqlValue(intentId)},
    ${sqlValue(fill.oid ? Number(fill.oid) : null)},
    ${sqlValue(fill.action)},
    ${sqlValue(fill.side)},
    ${sqlValue(fill.posType)},
    ${sqlValue(fill.price)},
    ${sqlValue(fill.size)},
    ${sqlValue(fill.notionalUsd)},
    ${sqlValue(fill.feeUsd)},
    ${sqlValue(fill.raw.feeToken || null)},
    ${sqlValue(fill.notionalUsd > 0 ? fill.feeUsd / fill.notionalUsd : null)},
    ${sqlValue(fill.pnlUsd)},
    ${sqlValue(fill.fillHash)},
    ${sqlValue(fill.fillTid)},
    ${sqlValue(matchedVia)},
    ${sqlValue(JSON.stringify(fill.raw))}
  );`;
}

function buildInsertManualIntent(fill) {
  const intentId = makeManualIntentId(fill);
  const requestedNotional = fill.notionalUsd;
  const requestedSize = fill.size;
  const metaJson = JSON.stringify({
    manual: true,
    source: "reconcile_live_fills",
    fill: fill.raw,
  });
  return {
    intentId,
    sql: `INSERT OR IGNORE INTO oms_intents (
      intent_id, created_ts_ms, sent_ts_ms, symbol, action, side, requested_size,
      requested_notional, entry_atr, leverage, decision_ts_ms, strategy_version,
      strategy_sha1, reason, confidence, status, dedupe_key, client_order_id,
      exchange_order_id, last_error, meta_json
    ) VALUES (
      ${sqlValue(intentId)},
      ${sqlValue(fill.tsMs)},
      ${sqlValue(fill.tsMs)},
      ${sqlValue(fill.symbol)},
      ${sqlValue(fill.action)},
      ${sqlValue(fill.side)},
      ${sqlValue(requestedSize)},
      ${sqlValue(requestedNotional)},
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      ${sqlValue("MANUAL_FILL")},
      ${sqlValue("n/a")},
      ${sqlValue("FILLED")},
      NULL,
      NULL,
      ${sqlValue(fill.oid)},
      NULL,
      ${sqlValue(metaJson)}
    );`,
    intent: {
      intent_id: intentId,
      symbol: fill.symbol,
      action: fill.action,
      side: fill.side,
      status: "FILLED",
      requested_size: requestedSize,
      requested_notional: requestedNotional,
      entry_atr: null,
      leverage: null,
      reason: "MANUAL_FILL",
      confidence: "n/a",
      client_order_id: null,
      exchange_order_id: fill.oid,
      meta_json: metaJson,
    },
  };
}

function buildUnknownRepair(update) {
  return `UPDATE trades
SET action = ${sqlValue(update.action)},
    type = ${sqlValue(update.posType)},
    size = ${sqlValue(update.size)},
    notional = ${sqlValue(update.notional)},
    reason = 'manual_trade',
    reason_code = 'manual_trade',
    confidence = 'MANUAL',
    meta_json = ${sqlValue(update.metaJson)}
WHERE id = ${sqlValue(update.id)};`;
}

function repairUnknownReversals(tradesById, omsFillByKey) {
  const unknownTrades = [...tradesById.values()]
    .filter((trade) => trade.action === "UNKNOWN")
    .sort((a, b) => a.id - b.id);
  if (unknownTrades.length === 0) return [];
  const repairs = [];
  for (const trade of unknownTrades) {
    const fill = omsFillByKey.get(keyOf(trade.fill_hash, trade.fill_tid));
    if (!fill) {
      throw new Error(`missing oms fill for unknown trade id=${trade.id}`);
    }
    let fillJson = {};
    try {
      fillJson = JSON.parse(fill.raw_json || "{}");
    } catch {
      throw new Error(`invalid raw_json for unknown trade id=${trade.id}`);
    }
    const rawFill = fillJson.fill || fillJson;
    const dir = String(rawFill.dir || "").trim();
    const startPosition = parseNum(rawFill.startPosition) ?? 0;
    const fillSize = parseNum(rawFill.sz) ?? trade.size;
    const price = parseNum(rawFill.px) ?? trade.price;
    if (dir === "Long > Short") {
      const closeSize = Math.abs(startPosition);
      if (!(closeSize > 0 && fillSize > closeSize)) {
        throw new Error(`could not derive close leg for trade id=${trade.id}`);
      }
      repairs.push({
        id: trade.id,
        action: "CLOSE",
        posType: "LONG",
        size: closeSize,
        notional: closeSize * price,
        metaJson: JSON.stringify({
          source: "manual_trade",
          repair: {
            source: "reconcile_live_fills",
            kind: "reversal_close_leg",
            original_fill_size: fillSize,
            start_position: startPosition,
            resulting_position: startPosition - fillSize,
          },
        }),
      });
      continue;
    }
    if (dir === "Short > Long") {
      const closeSize = Math.abs(startPosition);
      const openSize = fillSize - closeSize;
      if (!(closeSize > 0 && openSize > 0)) {
        throw new Error(`could not derive open leg for trade id=${trade.id}`);
      }
      repairs.push({
        id: trade.id,
        action: "OPEN",
        posType: "LONG",
        size: openSize,
        notional: openSize * price,
        metaJson: JSON.stringify({
          source: "manual_trade",
          repair: {
            source: "reconcile_live_fills",
            kind: "reversal_open_leg",
            original_fill_size: fillSize,
            start_position: startPosition,
            resulting_position: startPosition + fillSize,
          },
        }),
      });
      continue;
    }
    throw new Error(`unsupported UNKNOWN direction for trade id=${trade.id}: ${dir}`);
  }
  return repairs;
}

async function main() {
  const args = parseArgs(process.argv);
  const db = args.db;

  const existingOmsFills = sqlJson(
    db,
    `SELECT id, ts_ms, symbol, intent_id, order_id, action, side, pos_type, price, size, notional,
            fee_usd, fee_token, fee_rate, pnl_usd, fill_hash, fill_tid, matched_via, raw_json
     FROM oms_fills`
  );
  const existingTrades = sqlJson(
    db,
    `SELECT id, timestamp, symbol, action, type, price, size, notional, reason, reason_code,
            confidence, pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage,
            margin_used, meta_json, fill_hash, fill_tid
     FROM trades`
  );
  const intents = sqlJson(
    db,
    `SELECT intent_id, created_ts_ms, symbol, action, side, status, requested_size,
            requested_notional, entry_atr, leverage, reason, confidence, client_order_id,
            exchange_order_id, meta_json
     FROM oms_intents`
  );
  const fills = await fetchJson("https://api.hyperliquid.xyz/info", {
    type: "userFillsByTime",
    user: args.user,
    startTime: args.start,
    endTime: args.end,
    aggregateByTime: false,
  });

  const omsFillKeys = new Map(existingOmsFills.map((row) => [keyOf(row.fill_hash, row.fill_tid), row]));
  const tradeKeys = new Map(existingTrades.map((row) => [keyOf(row.fill_hash, row.fill_tid), row]));
  const tradesById = new Map(existingTrades.map((row) => [row.id, row]));
  const omsFillByKey = new Map(existingOmsFills.map((row) => [keyOf(row.fill_hash, row.fill_tid), row]));
  const intentMaps = buildIntentMaps(intents);
  const latestTradeTsMs = existingTrades.reduce((maxValue, row) => {
    const ts = row.timestamp ? Date.parse(row.timestamp) : NaN;
    return Number.isFinite(ts) ? Math.max(maxValue, ts) : maxValue;
  }, 0);

  const missing = [];
  const unsupported = [];
  for (const fillRaw of fills) {
    const fill = parseFill(fillRaw);
    const key = keyOf(fill.fillHash, fill.fillTid);
    if (omsFillKeys.has(key)) continue;
    if (fill.unsupported) {
      unsupported.push(fill);
      continue;
    }
    missing.push(fill);
  }
  if (unsupported.length > 0) {
    throw new Error(
      `unsupported missing fills encountered: ${unsupported
        .map((item) => `${item.symbol}:${item.dir}:${item.fillHash}:${item.fillTid}`)
        .join(", ")}`
    );
  }

  const plannedSql = [];
  const plannedSummary = {
    missingOmsFills: missing.length,
    insertedTrades: 0,
    insertedOmsFills: 0,
    insertedManualIntents: 0,
    updatedManualTrades: 0,
    repairedUnknownTrades: 0,
    missingBySymbol: {},
    matchedBy: {},
  };

  for (const fill of missing) {
    plannedSummary.missingBySymbol[fill.symbol] =
      (plannedSummary.missingBySymbol[fill.symbol] || 0) + 1;
    let intent = null;
    let matchedVia = null;
    if (fill.cloid && intentMaps.byClientOrderId.has(fill.cloid)) {
      intent = intentMaps.byClientOrderId.get(fill.cloid);
      matchedVia = "client_order_id";
    } else if (fill.oid && intentMaps.byExchangeOrderId.has(String(fill.oid))) {
      intent = intentMaps.byExchangeOrderId.get(String(fill.oid));
      matchedVia = "exchange_order_id";
    } else if (cloidKind(fill.cloid) === "aiq") {
      intent = null;
      matchedVia = "unmatched_aiq_fill";
    } else {
      const manualInsert = buildInsertManualIntent(fill);
      plannedSql.push(manualInsert.sql);
      intent = manualInsert.intent;
      intentMaps.byIntentId.set(intent.intent_id, intent);
      if (intent.exchange_order_id) {
        intentMaps.byExchangeOrderId.set(String(intent.exchange_order_id), intent);
      }
      matchedVia = "manual_orphan";
      plannedSummary.insertedManualIntents += 1;
    }

    plannedSummary.matchedBy[matchedVia] = (plannedSummary.matchedBy[matchedVia] || 0) + 1;
    plannedSql.push(buildInsertOmsFill(fill, intent?.intent_id || null, matchedVia));
    plannedSummary.insertedOmsFills += 1;

    const tradeKey = keyOf(fill.fillHash, fill.fillTid);
    if (!tradeKeys.has(tradeKey) && fill.tsMs > latestTradeTsMs) {
      plannedSql.push(buildInsertTrade(fill, intent, matchedVia));
      plannedSummary.insertedTrades += 1;
    }
  }

  plannedSql.push(`UPDATE trades
SET reason = 'manual_trade',
    reason_code = 'manual_trade',
    confidence = 'MANUAL'
WHERE EXISTS (
    SELECT 1
    FROM oms_fills f
    WHERE IFNULL(f.fill_hash, '') = IFNULL(trades.fill_hash, '')
      AND IFNULL(f.fill_tid, -1) = IFNULL(trades.fill_tid, -1)
      AND (json_extract(f.raw_json, '$.cloid') IS NULL OR json_extract(f.raw_json, '$.cloid') = '')
      AND (json_extract(f.raw_json, '$.fill.cloid') IS NULL OR json_extract(f.raw_json, '$.fill.cloid') = '')
)
AND NOT (
    reason = 'manual_trade' OR
    reason_code = 'manual_trade' OR
    confidence = 'MANUAL'
);`);

  const unknownRepairs = repairUnknownReversals(tradesById, omsFillByKey);
  for (const update of unknownRepairs) {
    plannedSql.push(buildUnknownRepair(update));
    plannedSummary.repairedUnknownTrades += 1;
  }

  const workDir = mkdtempSync(join(tmpdir(), "aiq-reconcile-"));
  const sqlPath = join(workDir, "reconcile.sql");
  const reportPath = join(workDir, "report.json");
  plannedSummary.updatedManualTrades = sqlCount(
    db,
    `SELECT COUNT(*) AS count
     FROM oms_fills f
     JOIN trades t
       ON IFNULL(f.fill_hash, '') = IFNULL(t.fill_hash, '')
      AND IFNULL(f.fill_tid, -1) = IFNULL(t.fill_tid, -1)
     WHERE (json_extract(f.raw_json, '$.cloid') IS NULL OR json_extract(f.raw_json, '$.cloid') = '')
       AND (json_extract(f.raw_json, '$.fill.cloid') IS NULL OR json_extract(f.raw_json, '$.fill.cloid') = '')
       AND NOT (
         t.reason = 'manual_trade' OR
         t.reason_code = 'manual_trade' OR
         t.confidence = 'MANUAL'
       )`
  );
  const sqlBody = `BEGIN IMMEDIATE;\n${plannedSql.join("\n")}\nCOMMIT;\n`;
  writeFileSync(sqlPath, sqlBody);
  writeFileSync(
    reportPath,
    JSON.stringify(
      {
        db,
        start: args.start,
        end: args.end,
        plannedSummary,
        workDir,
        sqlPath,
        reportPath,
      },
      null,
      2
    )
  );

  if (args.apply) {
    sh("sqlite3", [db, `.read ${sqlPath}`]);
  }

  process.stdout.write(
    JSON.stringify(
      {
        applied: args.apply,
        db,
        start: args.start,
        end: args.end,
        plannedSummary,
        workDir,
        sqlPath,
        reportPath,
      },
      null,
      2
    )
  );
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
