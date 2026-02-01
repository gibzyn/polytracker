# polymarket_wallet_monitor.py
import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Polymarket Wallet Monitor", page_icon="📈", layout="wide")

DATA_API_BASE = "https://data-api.polymarket.com"
USER_AGENT = "wallet-monitor/1.8"
HISTORY_DIR = "wallet_history"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ----------------------------
# Helpers
# ----------------------------
def now_utc():
    return datetime.now(timezone.utc)


def now_utc_str():
    return now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def fetch_json(url: str, params: dict | None = None, timeout: int = 15):
    r = session.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
            if x == "":
                return None
        return float(x)
    except Exception:
        return None


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def history_path(wallet: str) -> str:
    safe = wallet.strip().lower()
    return os.path.join(HISTORY_DIR, f"{safe}.csv")


def append_history(wallet: str, ts: datetime, portfolio_value: float | None, pnl: float | None, cost_basis: float | None):
    if portfolio_value is None:
        return
    ensure_dir(HISTORY_DIR)
    row = {
        "timestamp_utc": ts.isoformat(),
        "portfolio_value_usd": portfolio_value,
        "pnl_usd": pnl if pnl is not None else "",
        "cost_basis_usd": cost_basis if cost_basis is not None else "",
    }
    path = history_path(wallet)
    df_row = pd.DataFrame([row])
    if not os.path.exists(path):
        df_row.to_csv(path, index=False)
    else:
        df_row.to_csv(path, mode="a", header=False, index=False)


def load_history(wallet: str) -> pd.DataFrame:
    path = history_path(wallet)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")


# ----------------------------
# Portfolio metrics + enrichments
# ----------------------------
def compute_portfolio_metrics(df_pos: pd.DataFrame):
    if df_pos.empty:
        return None, None, None, None, 0, 0

    size_col = pick_first_existing_col(df_pos, ["size", "positionSize", "shares", "qty"])
    avg_col = pick_first_existing_col(df_pos, ["avgPrice", "averagePrice", "entryPrice"])
    cur_col = pick_first_existing_col(df_pos, ["curPrice", "currentPrice", "markPrice", "price"])
    value_col = pick_first_existing_col(df_pos, ["value", "currentValue", "positionValue", "usdValue", "marketValue"])

    work = df_pos.copy()
    work["_size"] = work[size_col].apply(safe_float) if size_col else None
    work["_avg"] = work[avg_col].apply(safe_float) if avg_col else None
    work["_cur"] = work[cur_col].apply(safe_float) if cur_col else None
    work["_val"] = work[value_col].apply(safe_float) if value_col else None

    # Treat 0 or negative "curPrice" as missing
    if "_cur" in work.columns:
        work.loc[work["_cur"].notna() & (work["_cur"] <= 0), "_cur"] = None

    # Portfolio value: prefer explicit value; else size*cur (only if cur exists)
    vals = work["_val"].dropna()
    if not vals.empty:
        portfolio_value = float(vals.sum())
    else:
        tmp = work.dropna(subset=["_size", "_cur"])
        portfolio_value = float((tmp["_size"] * tmp["_cur"]).sum()) if not tmp.empty else None

    # Cost basis: size * avg
    tmp2 = work.dropna(subset=["_size", "_avg"])
    cost_basis = float((tmp2["_size"] * tmp2["_avg"]).sum()) if not tmp2.empty else None

    pnl_est = None
    pnl_pct = None
    if portfolio_value is not None and cost_basis is not None and cost_basis != 0:
        pnl_est = portfolio_value - cost_basis
        pnl_pct = (pnl_est / cost_basis) * 100.0

    positions_count = int(len(df_pos))
    market_key = pick_first_existing_col(df_pos, ["conditionId", "condition_id", "marketId", "market"])
    markets_count = int(df_pos[market_key].nunique()) if market_key else 0

    return portfolio_value, cost_basis, pnl_est, pnl_pct, markets_count, positions_count


def add_position_cost_and_value_columns(df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - cost_paid_usd = size * avgPrice
      - current_value_usd = value/currentValue if present; else size * curPrice (only if curPrice exists and >0)
      - pnl_est_usd, pnl_est_pct computed only when current_value_usd is known
      - missing_mark_price flag when we cannot compute current value
    """
    if df_pos.empty:
        return df_pos

    size_col = pick_first_existing_col(df_pos, ["size", "positionSize", "shares", "qty"])
    avg_col = pick_first_existing_col(df_pos, ["avgPrice", "averagePrice", "entryPrice"])
    cur_col = pick_first_existing_col(df_pos, ["curPrice", "currentPrice", "markPrice", "price"])
    value_col = pick_first_existing_col(df_pos, ["value", "currentValue", "positionValue", "usdValue", "marketValue"])

    work = df_pos.copy()
    work["_size"] = work[size_col].apply(safe_float) if size_col else None
    work["_avg"] = work[avg_col].apply(safe_float) if avg_col else None
    work["_cur"] = work[cur_col].apply(safe_float) if cur_col else None
    work["_val"] = work[value_col].apply(safe_float) if value_col else None

    # Treat 0/negative current prices as missing
    if "_cur" in work.columns:
        work.loc[work["_cur"].notna() & (work["_cur"] <= 0), "_cur"] = None

    # Cost paid
    work["cost_paid_usd"] = (work["_size"] * work["_avg"]) if (size_col and avg_col) else None

    # Current value: prefer explicit value/currentValue; else size * current price if present
    if value_col:
        work["current_value_usd"] = work["_val"]
    else:
        if size_col and cur_col:
            work["current_value_usd"] = None
            m_cur = work["_size"].notna() & work["_cur"].notna()
            work.loc[m_cur, "current_value_usd"] = work.loc[m_cur, "_size"] * work.loc[m_cur, "_cur"]
        else:
            work["current_value_usd"] = None

    # Only compute PnL when current_value_usd is known
    work["pnl_est_usd"] = None
    work["pnl_est_pct"] = None
    m = work["current_value_usd"].notna() & work["cost_paid_usd"].notna() & (work["cost_paid_usd"] != 0)
    work.loc[m, "pnl_est_usd"] = work.loc[m, "current_value_usd"] - work.loc[m, "cost_paid_usd"]
    work.loc[m, "pnl_est_pct"] = (work.loc[m, "pnl_est_usd"] / work.loc[m, "cost_paid_usd"]) * 100.0

    # Flag missing marks
    work["missing_mark_price"] = work["current_value_usd"].isna()

    # Round for display
    for c in ["cost_paid_usd", "current_value_usd", "pnl_est_usd", "pnl_est_pct"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce").round(4)

    work.drop(columns=[c for c in ["_size", "_avg", "_cur", "_val"] if c in work.columns], inplace=True)
    return work


def hedged_condition_ids_from_positions(df_pos: pd.DataFrame) -> set[str]:
    if df_pos.empty:
        return set()
    cond_col = pick_first_existing_col(df_pos, ["conditionId", "condition_id", "marketId", "market"])
    out_col = pick_first_existing_col(df_pos, ["outcome", "outcomeName", "sideLabel"])
    size_col = pick_first_existing_col(df_pos, ["size", "positionSize", "shares", "qty"])
    if not cond_col or not out_col or not size_col:
        return set()

    w = df_pos.copy()
    w["_cond"] = w[cond_col].astype(str)
    w["_out"] = w[out_col].astype(str).str.strip()
    w["_size"] = w[size_col].apply(safe_float)
    w = w.dropna(subset=["_size"])
    if w.empty:
        return set()
    return set(w.groupby("_cond")["_out"].nunique()[lambda s: s >= 2].index.tolist())


def build_hedge_table(df_pos: pd.DataFrame, fee_buffer: float) -> pd.DataFrame:
    if df_pos.empty:
        return pd.DataFrame()

    cond_col = pick_first_existing_col(df_pos, ["conditionId", "condition_id", "marketId", "market"])
    out_col = pick_first_existing_col(df_pos, ["outcome", "outcomeName", "sideLabel"])
    size_col = pick_first_existing_col(df_pos, ["size", "positionSize", "shares", "qty"])
    avg_col = pick_first_existing_col(df_pos, ["avgPrice", "averagePrice", "entryPrice"])
    cur_col = pick_first_existing_col(df_pos, ["curPrice", "currentPrice", "markPrice", "price"])
    name_col = pick_first_existing_col(df_pos, ["marketName", "title", "market"])

    if not cond_col or not out_col or not size_col:
        return pd.DataFrame()

    w = df_pos.copy()
    w["_cond"] = w[cond_col].astype(str)
    w["_out"] = w[out_col].astype(str).str.strip()
    w["_size"] = w[size_col].apply(safe_float)
    w["_avg"] = w[avg_col].apply(safe_float) if avg_col else None
    w["_cur"] = w[cur_col].apply(safe_float) if cur_col else None
    w["_name"] = w[name_col].astype(str) if name_col else w["_cond"]

    # Treat 0/negative current prices as missing
    if "_cur" in w.columns:
        w.loc[w["_cur"].notna() & (w["_cur"] <= 0), "_cur"] = None

    w = w.dropna(subset=["_size"])
    if w.empty:
        return pd.DataFrame()

    rows = []
    for cond, g in w.groupby("_cond"):
        g2 = g[g["_size"].abs() > 1e-9].copy()
        outs = sorted(g2["_out"].unique().tolist())
        if len(outs) < 2:
            continue

        g2 = g2.sort_values("_size", ascending=False)
        top = g2.head(2)

        two_outcome_market = (g2["_out"].nunique() == 2) and (len(top) == 2)

        avg_sum = None
        cur_sum = None
        locked_arb_like = None
        arb_opportunity_now = None

        if two_outcome_market:
            a1 = safe_float(top["_avg"].iloc[0]) if avg_col else None
            a2 = safe_float(top["_avg"].iloc[1]) if avg_col else None
            c1 = safe_float(top["_cur"].iloc[0]) if cur_col else None
            c2 = safe_float(top["_cur"].iloc[1]) if cur_col else None

            if a1 is not None and a2 is not None:
                avg_sum = a1 + a2
                locked_arb_like = avg_sum < (1.0 - fee_buffer)

            if c1 is not None and c2 is not None:
                cur_sum = c1 + c2
                arb_opportunity_now = cur_sum < (1.0 - fee_buffer)

        rows.append(
            {
                "market_label": g2["_name"].iloc[0],
                "condition_or_market": cond,
                "outcomes_held": ", ".join(outs),
                "two_outcome_market": two_outcome_market,
                "avg_sum_top2": round(avg_sum, 6) if isinstance(avg_sum, (int, float)) else None,
                "cur_sum_top2": round(cur_sum, 6) if isinstance(cur_sum, (int, float)) else None,
                "locked_arb_like": locked_arb_like,
                "arb_opportunity_now": arb_opportunity_now,
                "fee_buffer": fee_buffer,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------
# Styling (GREEN good / RED bad)
# ----------------------------
def style_positions(df: pd.DataFrame):
    """
    Entire row:
      - GREEN if pnl_est_usd > 0
      - RED if pnl_est_usd < 0
      - NEUTRAL if pnl_est_usd is missing (e.g., missing mark price)
    """
    if df.empty or "pnl_est_usd" not in df.columns:
        return df.style

    def row_style(row):
        v = safe_float(row.get("pnl_est_usd", None))
        if v is None:
            return [""] * len(row)
        if v > 0:
            return ["background-color: #0b3d0b; color: #eaffea"] * len(row)
        if v < 0:
            return ["background-color: #4b0b0b; color: #ffecec"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


def style_hedges(df: pd.DataFrame):
    if df.empty:
        return df.style

    def row_style(row):
        locked_ok = row.get("locked_arb_like", None) is True
        now_ok = row.get("arb_opportunity_now", None) is True
        two_outcome = row.get("two_outcome_market", None) is True

        if locked_ok or now_ok:
            return ["background-color: #0b3d0b; color: #eaffea"] * len(row)
        if two_outcome and (row.get("locked_arb_like", None) is False) and (row.get("arb_opportunity_now", None) is False):
            return ["background-color: #4b0b0b; color: #ffecec"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


# ----------------------------
# UI
# ----------------------------
st.title("📈 Polymarket Wallet Monitor")
st.caption("Includes Hedge Verification table to prove whether multi-outcome positions exist.")

with st.sidebar:
    wallet = st.text_input("Wallet address (0x...)", value="", placeholder="0x1234...")
    refresh_seconds = st.slider("Auto refresh (seconds)", 2, 60, 5)
    trades_limit = st.slider("Recent trades to show", 10, 500, 150)
    trades_param_mode = st.selectbox("Trades API param", ["user", "proxyWallet"], index=0)
    fee_buffer = st.slider("Hedge/arb fee buffer", 0.0, 0.10, 0.02, step=0.005)
    show_raw = st.toggle("Show raw JSON (debug)", value=False)
    run = st.toggle("Run", value=False)

if not run:
    st.info("Turn on **Run** in the sidebar.")
    st.stop()

wallet = wallet.strip()
if not wallet.startswith("0x") or len(wallet) < 10:
    st.warning("Enter a valid wallet address like `0x...`")
    st.stop()

# Auto-refresh without while True
try:
    st.autorefresh(interval=refresh_seconds * 1000, key="pm_refresh")
except Exception:
    pass

ts = now_utc()

# ----------------------------
# Positions fetch
# ----------------------------
positions_payload = []
positions_error = None
try:
    positions_payload = fetch_json(f"{DATA_API_BASE}/positions", params={"user": wallet})
except Exception as e:
    positions_error = str(e)
    positions_payload = []

if isinstance(positions_payload, dict) and "data" in positions_payload:
    positions_payload = positions_payload["data"]

df_pos_raw = pd.DataFrame(positions_payload) if positions_payload else pd.DataFrame()
df_pos = add_position_cost_and_value_columns(df_pos_raw)

portfolio_value, cost_basis, pnl_est, pnl_pct, markets_count, positions_count = compute_portfolio_metrics(df_pos_raw)

append_history(wallet, ts, portfolio_value, pnl_est, cost_basis)
hist = load_history(wallet)

# ----------------------------
# Overview
# ----------------------------
st.markdown("## Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio Value", f"${portfolio_value:,.2f}" if portfolio_value is not None else "N/A")
c2.metric("Est. PnL", f"${pnl_est:,.2f}" if pnl_est is not None else "N/A")
c3.metric("Est. PnL %", f"{pnl_pct:,.2f}%" if pnl_pct is not None else "N/A")
c4.metric("Markets", f"{markets_count}")
c5.metric("Last Refresh", now_utc_str())

if positions_error:
    st.warning(f"Positions fetch issue: {positions_error}")

# ----------------------------
# Historical
# ----------------------------
st.markdown("## Historical")
if not hist.empty and "portfolio_value_usd" in hist.columns:
    hp = hist.copy()
    hp["portfolio_value_usd"] = pd.to_numeric(hp["portfolio_value_usd"], errors="coerce")
    if "pnl_usd" in hp.columns:
        hp["pnl_usd"] = pd.to_numeric(hp["pnl_usd"], errors="coerce")
    hp = hp.dropna(subset=["timestamp_utc", "portfolio_value_usd"]).set_index("timestamp_utc")
    st.line_chart(hp[["portfolio_value_usd"]], height=210)
    if "pnl_usd" in hp.columns and hp["pnl_usd"].notna().any():
        st.line_chart(hp[["pnl_usd"]], height=210)
else:
    st.caption("History charts will fill after a few refreshes (stored in wallet_history/).")

# ----------------------------
# Hedge Verification (proof)
# ----------------------------
st.markdown("## Hedge Verification (Proof)")
if df_pos_raw.empty:
    st.info("No positions returned, nothing to verify.")
else:
    cond_col_v = pick_first_existing_col(df_pos_raw, ["conditionId", "condition_id", "marketId", "market"])
    out_col_v = pick_first_existing_col(df_pos_raw, ["outcome", "outcomeName", "sideLabel"])
    size_col_v = pick_first_existing_col(df_pos_raw, ["size", "positionSize", "shares", "qty"])

    st.write("Detected columns:", {"condition": cond_col_v, "outcome": out_col_v, "size": size_col_v})

    if not cond_col_v or not out_col_v:
        st.warning("Can't verify hedges: missing condition/market id and/or outcome fields.")
    else:
        v = df_pos_raw.copy()
        v["_cond"] = v[cond_col_v].astype(str)
        v["_out"] = v[out_col_v].astype(str).str.strip()
        if size_col_v:
            v["_size"] = v[size_col_v].apply(safe_float)
        else:
            v["_size"] = 1.0

        summary = (
            v.groupby("_cond")
            .agg(
                outcomes_held=("_out", lambda s: sorted(set(s))),
                outcome_count=("_out", "nunique"),
                total_rows=("_out", "size"),
                total_abs_size=("_size", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).abs().sum())),
            )
            .reset_index()
            .rename(columns={"_cond": "condition_or_market"})
            .sort_values(["outcome_count", "total_rows"], ascending=False)
        )

        st.write("Top conditions by outcome_count:")
        st.dataframe(summary.head(50), use_container_width=True, hide_index=True)

        hedged = summary[summary["outcome_count"] >= 2].copy()
        if hedged.empty:
            st.success("✅ Verified: no condition/market has 2+ distinct outcomes in CURRENT positions snapshot.")
        else:
            st.error("⚠️ Found possible hedges (2+ outcomes held in same condition/market):")
            st.dataframe(hedged, use_container_width=True, hide_index=True)

# ----------------------------
# Hedges + arb-like
# ----------------------------
st.markdown("## Hedges / Arb-like (2-outcome markets)")
hedge_df = build_hedge_table(df_pos_raw, fee_buffer=fee_buffer)
if hedge_df.empty:
    st.info("No hedge structures detected (no markets with multiple outcomes held).")
else:
    show_cols = [
        "market_label",
        "outcomes_held",
        "two_outcome_market",
        "avg_sum_top2",
        "cur_sum_top2",
        "locked_arb_like",
        "arb_opportunity_now",
        "fee_buffer",
    ]
    show_cols = [c for c in show_cols if c in hedge_df.columns]
    view = hedge_df[show_cols].copy()
    st.dataframe(style_hedges(view), use_container_width=True, hide_index=True)

# ----------------------------
# Positions
# ----------------------------
st.markdown("## Positions (Cost vs Value)")
if df_pos.empty:
    st.warning("No positions returned.")
else:
    preferred_cols = [
        "marketName",
        "title",
        "market",
        "conditionId",
        "asset",
        "outcome",
        "size",
        "avgPrice",
        "curPrice",
        "cost_paid_usd",
        "current_value_usd",
        "pnl_est_usd",
        "pnl_est_pct",
        "missing_mark_price",
        "updatedAt",
    ]
    cols = [c for c in preferred_cols if c in df_pos.columns]
    pos_view = df_pos[cols].copy() if cols else df_pos.copy()
    st.dataframe(style_positions(pos_view), use_container_width=True, hide_index=True)

    if "missing_mark_price" in pos_view.columns:
        missing = int(pos_view["missing_mark_price"].sum())
        if missing:
            st.info(f"{missing} position rows have no mark price/value from the API, so PnL is not scored (neutral).")

if show_raw:
    with st.expander("Raw positions JSON"):
        st.json(positions_payload)

# ----------------------------
# Trades
# ----------------------------
st.markdown("## Trades")
trades_payload = []
trades_error = None
try:
    trades_payload = fetch_json(
        f"{DATA_API_BASE}/trades",
        params={trades_param_mode: wallet, "limit": trades_limit},
    )
except Exception as e:
    trades_error = str(e)
    trades_payload = []

if isinstance(trades_payload, dict) and "data" in trades_payload:
    trades_payload = trades_payload["data"]

if trades_error:
    st.warning(f"Trades fetch issue: {trades_error}")

if not trades_payload:
    st.warning("No trades returned (try switching Trades API param in the sidebar).")
else:
    df_trades = pd.DataFrame(trades_payload)

    hedged_conds = hedged_condition_ids_from_positions(df_pos_raw)
    t_cond = pick_first_existing_col(df_trades, ["conditionId", "condition_id", "marketId", "market"])
    if t_cond:
        df_trades["_cond"] = df_trades[t_cond].astype(str)
        df_trades["hedged_market_now"] = df_trades["_cond"].apply(lambda x: x in hedged_conds)
    else:
        df_trades["hedged_market_now"] = None

    for ts_col in ["timestamp", "createdAt"]:
        if ts_col in df_trades.columns:
            df_trades = df_trades.sort_values(by=ts_col, ascending=False)
            break

    preferred_trade_cols = [
        "timestamp",
        "createdAt",
        "marketName",
        "title",
        "market",
        "conditionId",
        "asset",
        "outcome",
        "side",
        "price",
        "size",
        "hedged_market_now",
        "txHash",
    ]
    tcols = [c for c in preferred_trade_cols if c in df_trades.columns]

    tab1, tab2 = st.tabs(["All Trades", "Non-Hedged Trades"])
    with tab1:
        st.dataframe(df_trades[tcols] if tcols else df_trades, use_container_width=True, hide_index=True)
    with tab2:
        if "hedged_market_now" in df_trades.columns:
            non = df_trades[df_trades["hedged_market_now"] != True]
            st.dataframe(non[tcols] if tcols else non, use_container_width=True, hide_index=True)
        else:
            st.info("Cannot tag hedged/non-hedged (missing condition/market id field).")

if show_raw:
    with st.expander("Raw trades JSON"):
        st.json(trades_payload)

st.caption("Reached end of script ✅")
