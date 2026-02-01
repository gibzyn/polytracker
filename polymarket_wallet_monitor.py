import os
import re
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Polymarket Wallet Monitor", page_icon="📈", layout="wide")

DATA_API_BASE = "https://data-api.polymarket.com"
USER_AGENT = "wallet-monitor/2.1"
HISTORY_DIR = os.path.join(os.getcwd(), "wallet_history")  # cloud-friendly-ish (still ephemeral on redeploy)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# ----------------------------
# BTC/ETH ONLY FILTER
# ----------------------------
BTC_PATTERNS = [r"\bBTC\b", r"\bBITCOIN\b", r"\bXBT\b"]
ETH_PATTERNS = [r"\bETH\b", r"\bETHEREUM\b"]
_ALLOWED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (BTC_PATTERNS + ETH_PATTERNS)]


def _looks_like_btc_or_eth(text: str) -> bool:
    if not text:
        return False
    s = str(text)
    return any(p.search(s) for p in _ALLOWED_PATTERNS)


def is_btc_or_eth_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False

    fields_to_check = [
        item.get("marketName"),
        item.get("title"),
        item.get("asset"),
        item.get("question"),
        item.get("description"),
        item.get("eventTitle"),
        item.get("seriesTitle"),
        item.get("slug"),
    ]

    m = item.get("market")
    if isinstance(m, str):
        fields_to_check.append(m)
    elif isinstance(m, dict):
        fields_to_check.extend(
            [m.get("marketName"), m.get("title"), m.get("name"), m.get("ticker"), m.get("symbol"), m.get("slug")]
        )

    return any(_looks_like_btc_or_eth(v) for v in fields_to_check if v is not None)


def filter_payload_btc_eth(payload_list: list) -> list:
    if not isinstance(payload_list, list):
        return []
    return [x for x in payload_list if is_btc_or_eth_item(x)]


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


# Streamlit-cache wrappers (short TTL; reduces rerun spam)
@st.cache_data(ttl=7)
def cached_positions(wallet: str):
    return fetch_json(f"{DATA_API_BASE}/positions", params={"user": wallet})


@st.cache_data(ttl=7)
def cached_trades(wallet: str, trades_param_mode: str, limit: int):
    return fetch_json(f"{DATA_API_BASE}/trades", params={trades_param_mode: wallet, "limit": limit})


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


def append_history(wallet: str, ts: datetime, portfolio_value: float | None):
    if portfolio_value is None:
        return
    ensure_dir(HISTORY_DIR)
    row = {
        "timestamp_utc": ts.isoformat(),
        "portfolio_value_usd": portfolio_value,
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
    df["portfolio_value_usd"] = pd.to_numeric(df.get("portfolio_value_usd"), errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "portfolio_value_usd"]).sort_values("timestamp_utc")
    return df


# ----------------------------
# Portfolio value (robust)
# ----------------------------
def compute_positions_value(df_pos_raw: pd.DataFrame) -> float | None:
    """
    Best-effort open positions value in USD.
    Prefer explicit "value/currentValue/usdValue"; else size*curPrice if available.
    """
    if df_pos_raw.empty:
        return 0.0

    size_col = pick_first_existing_col(df_pos_raw, ["size", "positionSize", "shares", "qty"])
    cur_col = pick_first_existing_col(df_pos_raw, ["curPrice", "currentPrice", "markPrice", "price"])
    value_col = pick_first_existing_col(df_pos_raw, ["value", "currentValue", "positionValue", "usdValue", "marketValue"])

    w = df_pos_raw.copy()

    if value_col and value_col in w.columns:
        vals = w[value_col].apply(safe_float).dropna()
        if not vals.empty:
            return float(vals.sum())

    if size_col and cur_col and (size_col in w.columns) and (cur_col in w.columns):
        w["_size"] = w[size_col].apply(safe_float)
        w["_cur"] = w[cur_col].apply(safe_float)
        w.loc[w["_cur"].notna() & (w["_cur"] <= 0), "_cur"] = None
        tmp = w.dropna(subset=["_size", "_cur"])
        if not tmp.empty:
            return float((tmp["_size"] * tmp["_cur"]).sum())

    return None


def enrich_positions_table(df_pos_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Display-only: adds current_value_usd when possible and a missing flag.
    No cost-basis / scary PnL math.
    """
    if df_pos_raw.empty:
        return df_pos_raw

    size_col = pick_first_existing_col(df_pos_raw, ["size", "positionSize", "shares", "qty"])
    cur_col = pick_first_existing_col(df_pos_raw, ["curPrice", "currentPrice", "markPrice", "price"])
    value_col = pick_first_existing_col(df_pos_raw, ["value", "currentValue", "positionValue", "usdValue", "marketValue"])

    w = df_pos_raw.copy()

    if value_col:
        w["current_value_usd"] = w[value_col].apply(safe_float)
    else:
        w["current_value_usd"] = None
        if size_col and cur_col:
            w["_size"] = w[size_col].apply(safe_float)
            w["_cur"] = w[cur_col].apply(safe_float)
            w.loc[w["_cur"].notna() & (w["_cur"] <= 0), "_cur"] = None
            m = w["_size"].notna() & w["_cur"].notna()
            w.loc[m, "current_value_usd"] = w.loc[m, "_size"] * w.loc[m, "_cur"]
            w.drop(columns=["_size", "_cur"], inplace=True, errors="ignore")

    w["missing_mark_or_value"] = w["current_value_usd"].isna()
    w["current_value_usd"] = pd.to_numeric(w["current_value_usd"], errors="coerce").round(4)
    return w


# ----------------------------
# History-based P/L (Option 1)
# ----------------------------
def timeframe_start(ts_now: datetime, tf: str) -> datetime | None:
    if tf == "1D":
        return ts_now - timedelta(days=1)
    if tf == "1W":
        return ts_now - timedelta(weeks=1)
    if tf == "1M":
        return ts_now - timedelta(days=30)
    if tf == "ALL":
        return None
    return ts_now - timedelta(days=1)


def compute_pl_from_history(hist: pd.DataFrame, now_value: float | None, ts_now: datetime, tf: str):
    """
    P/L over timeframe = now_value - value_at_or_before(start_time).
    Uses nearest snapshot at/before start_time; else earliest available.
    """
    if now_value is None or hist.empty:
        return None, None

    start = timeframe_start(ts_now, tf)
    if start is None:
        anchor_row = hist.iloc[0]
        anchor_value = float(anchor_row["portfolio_value_usd"])
        return now_value - anchor_value, anchor_value

    subset = hist[hist["timestamp_utc"] <= start]
    if not subset.empty:
        anchor_row = subset.iloc[-1]
    else:
        anchor_row = hist.iloc[0]

    anchor_value = float(anchor_row["portfolio_value_usd"])
    return now_value - anchor_value, anchor_value


def biggest_win_from_history(hist: pd.DataFrame) -> float | None:
    """
    Proxy: biggest positive jump between consecutive snapshots.
    """
    if hist is None or hist.empty or len(hist) < 2:
        return None
    diffs = hist["portfolio_value_usd"].diff()
    if diffs.notna().any():
        mx = diffs.max()
        if pd.isna(mx):
            return None
        return float(mx) if mx > 0 else 0.0
    return None


# ----------------------------
# UI
# ----------------------------
st.title("📈 Polymarket Wallet Monitor (BTC/ETH Only)")
st.caption("Accurate tracking via portfolio value change (no lifetime PnL guesses).")

with st.sidebar:
    wallet = st.text_input("Wallet address (0x...)", value="", placeholder="0x1234...")
    trades_limit = st.slider("Recent trades to show", 10, 500, 150)
    trades_param_mode = st.selectbox("Trades API param", ["user", "proxyWallet"], index=0)

    auto_refresh = st.toggle("Auto refresh", value=False)
    refresh_seconds = st.slider("Auto refresh (seconds)", 5, 120, 15)
    show_raw = st.toggle("Show raw JSON (debug)", value=False)

    fetch_clicked = st.button("Fetch / Refresh")

# Gate fetch: either user clicks, or auto refresh tick fires
if auto_refresh:
    try:
        st.autorefresh(interval=refresh_seconds * 1000, key="pm_refresh")
        should_run = True
    except Exception:
        should_run = fetch_clicked
else:
    should_run = fetch_clicked

if not should_run:
    st.info("Click **Fetch / Refresh** to load the wallet.")
    st.stop()

wallet = wallet.strip()
if not wallet.startswith("0x") or len(wallet) < 10:
    st.warning("Enter a valid wallet address like `0x...`")
    st.stop()

ts = now_utc()

# ----------------------------
# Fetch + filter
# ----------------------------
positions_error = None
trades_error = None

try:
    positions_payload = cached_positions(wallet)
except Exception as e:
    positions_error = str(e)
    positions_payload = []

if isinstance(positions_payload, dict) and "data" in positions_payload:
    positions_payload = positions_payload["data"]

positions_payload = filter_payload_btc_eth(positions_payload)

try:
    trades_payload = cached_trades(wallet, trades_param_mode, trades_limit)
except Exception as e:
    trades_error = str(e)
    trades_payload = []

if isinstance(trades_payload, dict) and "data" in trades_payload:
    trades_payload = trades_payload["data"]

trades_payload = filter_payload_btc_eth(trades_payload)

df_pos_raw = pd.DataFrame(positions_payload) if positions_payload else pd.DataFrame()
df_pos_view = enrich_positions_table(df_pos_raw)

# Portfolio value NOW (positions value)
positions_value_now = compute_positions_value(df_pos_raw)
if positions_value_now is not None:
    append_history(wallet, ts, positions_value_now)

hist = load_history(wallet)

# ----------------------------
# Header metrics (screenshot vibe)
# ----------------------------
predictions_count = 0
if trades_payload:
    df_t_tmp = pd.DataFrame(trades_payload)
    mk = pick_first_existing_col(df_t_tmp, ["conditionId", "condition_id", "marketId", "market"])
    if mk:
        predictions_count = int(df_t_tmp[mk].astype(str).nunique())
    else:
        predictions_count = int(len(df_t_tmp))

biggest_win = biggest_win_from_history(hist)

# timeframe buttons
tf_cols = st.columns([3, 1, 1, 1, 1])
with tf_cols[0]:
    st.subheader("Dashboard")
with tf_cols[1]:
    tf_1d = st.button("1D")
with tf_cols[2]:
    tf_1w = st.button("1W")
with tf_cols[3]:
    tf_1m = st.button("1M")
with tf_cols[4]:
    tf_all = st.button("ALL")

if "tf" not in st.session_state:
    st.session_state["tf"] = "1D"
if tf_1d:
    st.session_state["tf"] = "1D"
elif tf_1w:
    st.session_state["tf"] = "1W"
elif tf_1m:
    st.session_state["tf"] = "1M"
elif tf_all:
    st.session_state["tf"] = "ALL"

tf = st.session_state["tf"]

pl_usd, anchor_value = compute_pl_from_history(hist, positions_value_now, ts, tf)

# Two main cards
left, right = st.columns([1.2, 1.8])  # ✅ no vertical_alignment (older streamlit-safe)

with left:
    st.markdown("### Wallet")
    st.code(wallet, language=None)
    st.caption(f"Last refresh: {now_utc_str()}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Positions Value", f"${positions_value_now:,.2f}" if positions_value_now is not None else "N/A")
    m2.metric("Biggest Win", f"${biggest_win:,.2f}" if biggest_win is not None else "N/A")
    m3.metric("Predictions", f"{predictions_count}")

    if positions_error:
        st.warning(f"Positions fetch issue: {positions_error}")
    if trades_error:
        st.warning(f"Trades fetch issue: {trades_error}")

with right:
    st.markdown(f"### Profit / Loss ({tf})")
    if pl_usd is None:
        st.metric("P/L", "N/A")
        st.caption("Not enough history yet. Refresh over time to build snapshots.")
    else:
        st.metric("P/L", f"${pl_usd:,.2f}")
        if anchor_value is not None:
            st.caption(f"Computed as: value now − value at start of window (anchor ${anchor_value:,.2f}).")

    if not hist.empty:
        chart = hist.copy().set_index("timestamp_utc")
        start = timeframe_start(ts, tf)
        if start is not None:
            chart = chart[chart.index >= start]
        st.line_chart(chart[["portfolio_value_usd"]], height=220)
    else:
        st.caption("History will appear after first snapshot is saved.")

st.divider()

# ----------------------------
# Positions table (no PnL)
# ----------------------------
st.markdown("## Positions (BTC/ETH only)")
if df_pos_view.empty:
    st.info("No BTC/ETH positions returned.")
else:
    preferred_cols = [
        "marketName",
        "title",
        "market",
        "conditionId",
        "asset",
        "outcome",
        "size",
        "curPrice",
        "current_value_usd",
        "missing_mark_or_value",
        "updatedAt",
    ]
    cols = [c for c in preferred_cols if c in df_pos_view.columns]
    view = df_pos_view[cols].copy() if cols else df_pos_view.copy()
    st.dataframe(view, use_container_width=True, hide_index=True)

    if "missing_mark_or_value" in view.columns:
        missing = int(view["missing_mark_or_value"].sum())
        if missing:
            st.info(f"{missing} position rows have no mark/value from the API, so they’re shown without valuation.")

if show_raw:
    with st.expander("Raw positions JSON (BTC/ETH filtered)"):
        st.json(positions_payload)

# ----------------------------
# Trades table
# ----------------------------
st.markdown("## Activity / Trades (BTC/ETH only)")
if not trades_payload:
    st.info("No BTC/ETH trades returned (try switching Trades API param in the sidebar).")
else:
    df_trades = pd.DataFrame(trades_payload)

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
        "txHash",
    ]
    tcols = [c for c in preferred_trade_cols if c in df_trades.columns]
    st.dataframe(df_trades[tcols] if tcols else df_trades, use_container_width=True, hide_index=True)

if show_raw:
    with st.expander("Raw trades JSON (BTC/ETH filtered)"):
        st.json(trades_payload)

st.caption("Reached end of script ✅")
