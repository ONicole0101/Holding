import argparse
import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    import config  # type: ignore
except Exception:
    config = None

API_URL = "https://api.finmindtrade.com/api/v4/data"
USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"
TOKEN = os.getenv("FINMIND_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

DEFAULT_OUTPUT_FILE = "AllStatic_Chip.csv"

# FinMind official note: TaiwanStockTradingDailyReport is too large for ranges.
# Request one day at a time. Known missing dates are skipped to avoid false errors.
MISSING_DATE_RANGES = [
    (dt.date(2022, 10, 31), dt.date(2022, 11, 3)),
    (dt.date(2023, 1, 11), dt.date(2023, 1, 17)),
]

STATIC_CHIP_COLUMNS = [
    "stock_id",
    "name",
    "chip_trend_days",
    "chip_concentration_threshold",
    "chip_latest_date",
    "chip_available_days",
    "chip_concentration_pct",
    "chip_concentration_score",
    "main_force_net",
    "main_force_score",
    "broker_diff",
    "broker_diff_score",
    "chip_signal_state",
    "chip_signal_text",
    "chips_updated_at",
    "chips_status",
    "chips_reason",
    "finmind_token_status",
    "finmind_token_source",
    "finmind_token_masked",
    "finmind_user_count",
    "finmind_api_request_limit",
    "finmind_remain",
    "finmind_usage_checked_at",
]


def utc_now_str() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def mask_token(token: Optional[str] = None) -> str:
    token = token if token is not None else TOKEN
    if not token:
        return ""
    token = str(token)
    if len(token) <= 8:
        return "*" * len(token)
    return token[:4] + "..." + token[-4:]


def compact_text(text: Any, max_len: int = 220) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def safe_json(res: requests.Response) -> Dict[str, Any]:
    try:
        data = res.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_finmind_meta() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "finmind_token_status": "missing_token" if not TOKEN else "unknown",
        "finmind_token_source": "FINMIND_TOKEN" if TOKEN else "",
        "finmind_token_masked": mask_token(),
        "finmind_user_count": None,
        "finmind_api_request_limit": None,
        "finmind_remain": None,
        "finmind_usage_checked_at": utc_now_str(),
    }
    if not TOKEN:
        return info
    try:
        res = requests.get(USER_INFO_URL, headers=HEADERS, timeout=60)
        data = safe_json(res)
        used = data.get("user_count")
        limit = data.get("api_request_limit")
        used_int = int(used or 0) if str(used or "").strip() != "" else None
        limit_int = int(limit or 0) if str(limit or "").strip() != "" else None
        remain = max(limit_int - used_int, 0) if used_int is not None and limit_int else None
        info.update({
            "finmind_token_status": "ok" if res.status_code == 200 and not data.get("error") else "error",
            "finmind_user_count": used_int,
            "finmind_api_request_limit": limit_int,
            "finmind_remain": remain,
        })
    except Exception as exc:
        info["finmind_token_status"] = "error"
        print(f"⚠️ cannot check FinMind user_info: {exc}", flush=True)
    return info


def score_by_ratio(ratio: float) -> float:
    if ratio >= 0.999:
        return 1
    if ratio > 0:
        return 0.5
    if ratio <= -0.999:
        return -1
    if ratio < 0:
        return -0.5
    return 0


def read_int_env(name: str, default: int, min_value: int = 1, max_value: int = 20) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(min_value, min(int(value), max_value))


def read_float_env(name: str, default: float, min_value: float = 0, max_value: float = 100) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(min_value, min(float(value), max_value))


def resolve_csv_file(csv_file: Optional[str] = None) -> str:
    if csv_file:
        return csv_file
    if os.getenv("CSV_FILE"):
        return os.getenv("CSV_FILE", "stocks.csv")
    if config is not None and getattr(config, "CSV_FILE", None):
        return getattr(config, "CSV_FILE")
    return "stocks.csv"


def resolve_output(output_file: Optional[str] = None) -> str:
    if output_file:
        return output_file
    if os.getenv("STATIC_CHIP_FILE"):
        return os.getenv("STATIC_CHIP_FILE", DEFAULT_OUTPUT_FILE)
    if os.getenv("STATIC_CHIPS_FILE"):
        return os.getenv("STATIC_CHIPS_FILE", DEFAULT_OUTPUT_FILE)
    if config is not None:
        for attr in ("STATIC_CHIP_OUTPUT_FILE", "STATIC_CHIPS_OUTPUT_FILE"):
            if getattr(config, attr, None):
                return getattr(config, attr)
    return DEFAULT_OUTPUT_FILE


def load_stock_list(csv_file: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"stock csv not found: {csv_file}")
    df = pd.read_csv(csv_file, sep="\t", encoding="utf-8-sig", dtype=str)
    if len(df.columns) == 1:
        # tolerate comma separated files too
        df = pd.read_csv(csv_file, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()
    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "stock_id"})
    if "Name" in df.columns:
        df = df.rename(columns={"Name": "name"})
    if "stock_id" not in df.columns:
        raise ValueError(f"{csv_file} missing Ticker/stock_id column")
    if "name" not in df.columns:
        df["name"] = ""
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df = df[df["stock_id"] != ""]
    return df[["stock_id", "name"]].to_dict(orient="records")


def is_known_missing_date(day: dt.date) -> bool:
    return any(start <= day <= end for start, end in MISSING_DATE_RANGES)


def iter_candidate_dates(start_date: dt.date, end_date: dt.date) -> List[dt.date]:
    days: List[dt.date] = []
    cur = start_date
    while cur <= end_date:
        if not is_known_missing_date(cur):
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def fetch_trading_daily_report_one_day(stock_id: str, day: dt.date, verbose: bool = True) -> pd.DataFrame:
    # Do NOT send end_date. FinMind returns 400 for this dataset when end_date is present.
    params = {
        "dataset": "TaiwanStockTradingDailyReport",
        "data_id": str(stock_id),
        "start_date": day.strftime("%Y-%m-%d"),
        "token": TOKEN,
    }
    if verbose:
        print(
            "🔎 chip analysis request: "
            f"dataset={params['dataset']} stock_id={stock_id} "
            f"date={params['start_date']} token_present={bool(TOKEN)}",
            flush=True,
        )
    res = requests.get(API_URL, params=params, headers=HEADERS, timeout=300)
    if verbose:
        print(f"🔄 chip analysis response status: {res.status_code}", flush=True)
    data = safe_json(res)
    if res.status_code != 200:
        msg = data.get("msg") or data.get("message") or res.text[:200]
        raise RuntimeError(f"FinMind TaiwanStockTradingDailyReport error {stock_id} {day}: {res.status_code} {msg}")
    rows = data.get("data", [])
    return pd.DataFrame(rows)


def fetch_trading_daily_report_days(stock_id: str, days: List[dt.date], workers: int = 1, verbose: bool = True) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    errors: List[str] = []
    if workers <= 1:
        for day in days:
            try:
                df = fetch_trading_daily_report_one_day(stock_id, day, verbose=verbose)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                errors.append(str(exc))
                print(f"⚠️ chip day failed {stock_id} {day}: {exc}", flush=True)
        if errors and not frames:
            print(f"⚠️ all chip day requests failed {stock_id}; first={errors[0]}", flush=True)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(fetch_trading_daily_report_one_day, stock_id, day, False): day for day in days}
        for future in as_completed(future_map):
            day = future_map[future]
            try:
                df = future.result()
                if verbose:
                    print(f"🔄 chip day done stock_id={stock_id} date={day} rows={len(df)}", flush=True)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                errors.append(str(exc))
                print(f"⚠️ chip day failed {stock_id} {day}: {exc}", flush=True)
    if errors and not frames:
        print(f"⚠️ all chip day requests failed {stock_id}; first={errors[0]}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_chip_source(df: pd.DataFrame, stock_id: str, end_date: dt.date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.str.strip()
    broker_col = next(
        (c for c in ["broker", "securities_trader", "securities_trader_id", "securities_trader_name"] if c in df.columns),
        None,
    )
    if broker_col is None:
        print(f"⚠️ chip data has no broker-like column {stock_id}: cols={list(df.columns)}", flush=True)
        return pd.DataFrame()
    if "stock_id" in df.columns:
        df = df[df["stock_id"].astype(str).str.strip() == str(stock_id)]
    if df.empty:
        return pd.DataFrame()
    required = {"date", "buy", "sell"}
    if not required.issubset(df.columns):
        print(f"⚠️ chip data missing required columns {stock_id}: cols={list(df.columns)}", flush=True)
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0)
    df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date"])
    df = df[df["date"] <= end_date]
    if df.empty:
        return pd.DataFrame()
    df["broker_key"] = df[broker_col].astype(str).str.strip()
    df["net_buy"] = df["buy"] - df["sell"]
    return df


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    daily_rows: List[Dict[str, Any]] = []
    for date_value, group in df.groupby("date"):
        group = group.copy()
        active_buyers = group.loc[group["buy"] > 0, "broker_key"].nunique()
        active_sellers = group.loc[group["sell"] > 0, "broker_key"].nunique()
        broker_diff = int(active_buyers - active_sellers)
        sorted_group = group.sort_values("net_buy", ascending=False)
        top_buy = float(sorted_group.head(15)["net_buy"].sum())
        top_sell = float(sorted_group.tail(15)["net_buy"].sum())
        main_force_net = top_buy + top_sell
        total_turnover = float((group["buy"] + group["sell"]).sum())
        concentration_pct = abs(main_force_net) / total_turnover * 100 if total_turnover else None
        daily_rows.append({
            "date": date_value,
            "main_force_net": main_force_net,
            "broker_diff": broker_diff,
            "chip_concentration_pct": concentration_pct,
        })
    return pd.DataFrame(daily_rows).sort_values("date", ascending=False)


def no_data_result(trend_days: int, concentration_threshold: float, reason: str = "籌碼資料不足") -> Dict[str, Any]:
    return {
        "chip_trend_days": trend_days,
        "chip_concentration_threshold": concentration_threshold,
        "chip_latest_date": None,
        "chip_available_days": 0,
        "chip_concentration_pct": None,
        "chip_concentration_score": None,
        "main_force_net": None,
        "main_force_score": None,
        "broker_diff": None,
        "broker_diff_score": None,
        "chip_signal_state": "no_data",
        "chip_signal_text": reason,
    }


def analyze_chip(stock_id: str, trend_days: int, concentration_threshold: float, workers: int = 1, lookback_days: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback_days or max(trend_days * 7, 21))
    days = iter_candidate_dates(start_date, end_date)
    raw = fetch_trading_daily_report_days(stock_id, days, workers=workers, verbose=(workers <= 1))
    df = normalize_chip_source(raw, stock_id, end_date)
    report = summarize_daily(df)
    if report.empty:
        return pd.DataFrame(), no_data_result(
            trend_days,
            concentration_threshold,
            "未抓取到籌碼資料；請確認 TaiwanStockTradingDailyReport 單日請求、資料更新時間與 API 權限。",
        )
    recent = report.head(trend_days).copy()
    available_days = len(recent)
    if available_days == 0:
        return pd.DataFrame(), no_data_result(trend_days, concentration_threshold)

    main_pos = int((recent["main_force_net"] > 0).sum())
    main_neg = int((recent["main_force_net"] < 0).sum())
    diff_pos = int((recent["broker_diff"] > 0).sum())
    diff_neg = int((recent["broker_diff"] < 0).sum())
    conc_ok = recent["chip_concentration_pct"].fillna(0) >= concentration_threshold
    conc_pos = int(((recent["main_force_net"] > 0) & conc_ok).sum())
    conc_neg = int(((recent["main_force_net"] < 0) & conc_ok).sum())

    denom = float(available_days)
    main_score = score_by_ratio((main_pos - main_neg) / denom)
    broker_score = score_by_ratio((diff_pos - diff_neg) / denom)
    concentration_score = score_by_ratio((conc_pos - conc_neg) / denom)
    latest = recent.iloc[0]

    state = "neutral"
    text = "籌碼震盪，方向未定"
    enough_days = available_days >= trend_days
    if enough_days and main_pos == available_days and diff_neg == available_days and conc_pos >= 1:
        state = "bullish_concentrated"
        text = f"主力連{trend_days}買、買賣家數差連{trend_days}負，籌碼偏集中"
    elif enough_days and main_pos == available_days and diff_pos >= 1:
        state = "bullish_distributed"
        text = f"主力連{trend_days}買但買賣家數差偏正，可能偏分散"
    elif enough_days and main_neg == available_days and diff_pos == available_days:
        state = "bearish_distributed"
        text = f"主力連{trend_days}賣、買賣家數差連{trend_days}正，籌碼流向散戶風險高"
    elif enough_days and main_neg == available_days:
        state = "bearish"
        text = f"主力連{trend_days}賣，籌碼偏弱"
    elif not enough_days:
        state = "partial"
        text = f"僅取得{available_days}個交易日籌碼資料，暫列觀察"

    result = {
        "chip_trend_days": trend_days,
        "chip_concentration_threshold": concentration_threshold,
        "chip_latest_date": str(latest["date"]),
        "chip_available_days": int(available_days),
        "chip_concentration_pct": round(float(latest["chip_concentration_pct"]), 2) if pd.notna(latest["chip_concentration_pct"]) else None,
        "chip_concentration_score": concentration_score,
        "main_force_net": int(round(float(latest["main_force_net"]))),
        "main_force_score": main_score,
        "broker_diff": int(latest["broker_diff"]),
        "broker_diff_score": broker_score,
        "chip_signal_state": state,
        "chip_signal_text": text,
    }
    display = recent.rename(columns={
        "date": "Date",
        "main_force_net": "主力買賣超",
        "broker_diff": "買賣家數差",
        "chip_concentration_pct": "籌碼集中度%",
    })[["Date", "主力買賣超", "買賣家數差", "籌碼集中度%"]]
    display["主力買賣超"] = display["主力買賣超"].round(0).astype(int)
    display["籌碼集中度%"] = display["籌碼集中度%"].round(2)
    return display, result


def build_row(stock: Dict[str, str], chip_result: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    row = {column: None for column in STATIC_CHIP_COLUMNS}
    row.update({
        "stock_id": str(stock.get("stock_id", "")).strip(),
        "name": stock.get("name", ""),
        "chips_updated_at": utc_now_str(),
    })
    for key in [
        "chip_trend_days",
        "chip_concentration_threshold",
        "chip_latest_date",
        "chip_available_days",
        "chip_concentration_pct",
        "chip_concentration_score",
        "main_force_net",
        "main_force_score",
        "broker_diff",
        "broker_diff_score",
        "chip_signal_state",
        "chip_signal_text",
    ]:
        row[key] = chip_result.get(key)
    state = str(chip_result.get("chip_signal_state") or "").strip().lower()
    row["chips_status"] = "ok" if state not in {"", "no_data", "error"} else state or "no_data"
    row["chips_reason"] = "" if row["chips_status"] == "ok" else compact_text(chip_result.get("chip_signal_text"))
    row.update(meta)
    return row


def save_rows(rows: List[Dict[str, Any]], output_file: str) -> None:
    df = pd.DataFrame(rows)
    for column in STATIC_CHIP_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[STATIC_CHIP_COLUMNS]
    tmp = output_file + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, output_file)
    print(f"✅ Static chip saved: {output_file}, rows={len(df)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AllStatic_Chip.csv from FinMind broker chip data")
    parser.add_argument("--stock-id", default=os.getenv("CHIP_STOCK_ID"), help="Analyze one stock only. If omitted, stocks.csv is used.")
    parser.add_argument("--csv-file", default=None, help="Stock list TSV/CSV path. Default: config.CSV_FILE or stocks.csv")
    parser.add_argument("--output", default=None, help=f"Output CSV. Default: {DEFAULT_OUTPUT_FILE}")
    parser.add_argument("--trend-days", type=int, default=read_int_env("CHIP_TREND_DAYS", 3))
    parser.add_argument("--concentration-threshold", type=float, default=read_float_env("CHIP_CONCENTRATION_THRESHOLD", 15))
    parser.add_argument("--lookback-days", type=int, default=read_int_env("CHIP_LOOKBACK_DAYS", 21, min_value=3, max_value=90))
    parser.add_argument("--sleep-sec", type=float, default=float(os.getenv("CHIP_SLEEP_SEC", "0.05")))
    parser.add_argument("--workers", type=int, default=read_int_env("CHIP_WORKERS", 1, min_value=1, max_value=16), help="Parallel day requests per stock. Keep low if API rate limit is tight.")
    parser.add_argument("--no-prompt", action="store_true", help="Kept for workflow compatibility; no prompts are used in batch mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trend_days = max(1, min(int(args.trend_days), 20))
    concentration_threshold = max(0.0, min(float(args.concentration_threshold), 100.0))
    workers = max(1, min(int(args.workers), 16))
    output_file = resolve_output(args.output)
    meta = get_finmind_meta()
    print(f"FINMIND token present={bool(TOKEN)} status={meta.get('finmind_token_status')} remain={meta.get('finmind_remain')}", flush=True)
    print("FinMind note: TaiwanStockTradingDailyReport only supports one day per request; no end_date is sent.", flush=True)
    print("FinMind note: data usually updates Mon-Fri 21:00 Taiwan time; actual availability follows API data.", flush=True)
    print(f"籌碼參數：天數={trend_days}，集中度門檻={concentration_threshold:g}%，lookback={args.lookback_days}天，workers={workers}", flush=True)

    if args.stock_id:
        stocks = [{"stock_id": str(args.stock_id).strip(), "name": ""}]
    else:
        csv_file = resolve_csv_file(args.csv_file)
        stocks = load_stock_list(csv_file)
        print(f"📄 使用股票清單：{csv_file}, stocks={len(stocks)}", flush=True)

    rows: List[Dict[str, Any]] = []
    for idx, stock in enumerate(stocks, 1):
        sid = str(stock.get("stock_id", "")).strip()
        print(f"正在分析 {idx}/{len(stocks)} {sid} {stock.get('name', '')}...", flush=True)
        try:
            recent_df, chip_result = analyze_chip(sid, trend_days, concentration_threshold, workers=workers, lookback_days=args.lookback_days)
        except Exception as exc:
            print(f"❌ chip analysis failed {sid}: {exc}", flush=True)
            recent_df = pd.DataFrame()
            chip_result = no_data_result(trend_days, concentration_threshold, f"chip analysis error: {exc}")
            chip_result["chip_signal_state"] = "error"
        if args.stock_id:
            print(f"\n【最近 {trend_days} 個交易日籌碼數據】")
            print(recent_df.to_string(index=False) if not recent_df.empty else "無足夠籌碼資料可供顯示")
            print("-" * 40)
            print(chip_result.get("chip_signal_text"))
        rows.append(build_row(stock, chip_result, meta))
        if args.sleep_sec and args.sleep_sec > 0 and idx < len(stocks):
            time.sleep(args.sleep_sec)

    save_rows(rows, output_file)
    final_df = pd.DataFrame(rows)
    status_counts = final_df["chips_status"].astype(str).value_counts().to_dict() if not final_df.empty else {}
    print(f"AllStatic_Chip status summary: {status_counts}", flush=True)
    preview_cols = ["stock_id", "name", "chip_latest_date", "main_force_net", "broker_diff", "chip_signal_state", "chips_status"]
    print(final_df[preview_cols].head(30).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
