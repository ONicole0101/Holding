from __future__ import annotations

import argparse
import inspect
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd

try:
    import config  # type: ignore
except Exception:
    config = None

from data_sources import (
    get_chip_analysis,
    get_finmind_token_status,
    get_finmind_user_info,
    log_finmind_static_event,
)

DEFAULT_OUTPUT_FILE = "AllStatic_Chips.csv"

CHIP_DATA_COLS = [
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
]

FINMIND_META_COLS = [
    "finmind_token_status",
    "finmind_token_source",
    "finmind_token_masked",
    "finmind_user_count",
    "finmind_api_request_limit",
    "finmind_remain",
    "finmind_usage_checked_at",
]

ORDERED_COLS = [
    "stock_id",
    "name",
] + CHIP_DATA_COLS + [
    "chips_updated_at",
    "chips_status",
    "chips_reason",
] + FINMIND_META_COLS

_LAST_FINMIND_USAGE_INFO = None


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def cfg(name: str, default: Any = None) -> Any:
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    if config is not None and getattr(config, name, None) not in (None, ""):
        return getattr(config, name)
    return default


def resolve_csv_file(csv_file: str | None = None) -> str:
    return str(csv_file or cfg("CSV_FILE", "stocks.csv"))


def resolve_output_file(output_file: str | None = None) -> str:
    return str(
        output_file
        or os.getenv("STATIC_CHIP_FILE")
        or os.getenv("STATIC_CHIPS_FILE")
        or cfg("STATIC_CHIP_OUTPUT_FILE")
        or cfg("STATIC_CHIPS_OUTPUT_FILE")
        or DEFAULT_OUTPUT_FILE
    )


def compact_text(text: Any, max_len: int = 180) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def normalize_finmind_usage_info(info: dict | None) -> dict:
    info = info or {}
    return {
        "finmind_token_status": info.get("login_status") or ("ok" if info.get("token_present") else "missing_token"),
        "finmind_token_source": info.get("token_source") or "",
        "finmind_token_masked": info.get("token_masked") or "",
        "finmind_user_count": info.get("user_count"),
        "finmind_api_request_limit": info.get("api_request_limit"),
        "finmind_remain": info.get("remain"),
        "finmind_usage_checked_at": now_utc_str(),
    }


def apply_finmind_usage_to_row(row: dict, info: dict | None = None) -> dict:
    global _LAST_FINMIND_USAGE_INFO
    if info is None:
        info = _LAST_FINMIND_USAGE_INFO or get_finmind_token_status()
    row.update(normalize_finmind_usage_info(info))
    return row


def get_finmind_usage():
    global _LAST_FINMIND_USAGE_INFO
    info = get_finmind_user_info(write_log=True, source="generate_static_chips")
    _LAST_FINMIND_USAGE_INFO = info
    used = int(info.get("user_count") or 0)
    limit = int(info.get("api_request_limit") or 0)
    remain = info.get("remain")
    remain = int(remain or 0) if remain is not None else 0
    print(
        "FinMind token: "
        f"token_present={info.get('token_present')}, "
        f"source={info.get('token_source')}, "
        f"token={info.get('token_masked')}, "
        f"login={info.get('login_status')}",
        flush=True,
    )
    print(f"FinMind usage: {used}/{limit}, remain={remain}", flush=True)
    if not info.get("ok"):
        print(f"WARNING FinMind token/user_info check failed: {info.get('message')}", flush=True)
    return used, limit, remain


def normalize_chips_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ORDERED_COLS)
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in ORDERED_COLS:
        if col not in df.columns:
            df[col] = None
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    return df[ORDERED_COLS]


def atomic_write_csv(df: pd.DataFrame, path: str):
    tmp_path = path + ".tmp"
    df = normalize_chips_df(df)
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, path)


def empty_chip_row(stock: dict) -> dict:
    row = {col: None for col in ORDERED_COLS}
    row["stock_id"] = str(stock.get("stock_id", "")).strip()
    row["name"] = stock.get("name", "")
    row["chips_updated_at"] = now_utc_str()
    row["chips_status"] = "incomplete"
    row["chips_reason"] = "not processed yet"
    apply_finmind_usage_to_row(row)
    return row


def call_get_chip_analysis(stock_id: str, trend_days=None, concentration_threshold=None, lookback_days=None, workers=None) -> dict:
    kwargs = {
        "trend_days": trend_days,
        "concentration_threshold": concentration_threshold,
        "lookback_days": lookback_days,
        "workers": workers,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        sig = inspect.signature(get_chip_analysis)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return get_chip_analysis(stock_id, **supported) or {}
    except (TypeError, ValueError):
        # Backward compatible fallback for older data_sources.get_chip_analysis.
        return get_chip_analysis(stock_id, trend_days=trend_days, concentration_threshold=concentration_threshold) or {}


def build_chip_row(stock: dict, trend_days=None, concentration_threshold=None, lookback_days=None, workers=None) -> dict:
    stock_id = str(stock.get("stock_id", "")).strip()
    row = empty_chip_row(stock)
    row["chips_updated_at"] = now_utc_str()

    try:
        chip = call_get_chip_analysis(
            stock_id,
            trend_days=trend_days,
            concentration_threshold=concentration_threshold,
            lookback_days=lookback_days,
            workers=workers,
        )
        for col in CHIP_DATA_COLS:
            row[col] = chip.get(col)

        state = str(row.get("chip_signal_state") or "").strip().lower()
        if state and state not in {"no_data", "error"}:
            row["chips_status"] = "ok"
            row["chips_reason"] = ""
        elif state == "error":
            row["chips_status"] = "error"
            row["chips_reason"] = compact_text(row.get("chip_signal_text") or "籌碼資料錯誤")
        else:
            row["chips_status"] = "no_data"
            row["chips_reason"] = compact_text(row.get("chip_signal_text") or "籌碼資料不足")
    except Exception as exc:
        row["chips_status"] = "error"
        row["chips_reason"] = compact_text(str(exc))

    apply_finmind_usage_to_row(row)
    return row


def load_stock_list(csv_file: str | None = None) -> list[dict]:
    csv_file = resolve_csv_file(csv_file)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"stock csv not found: {csv_file}")

    df = pd.read_csv(csv_file, sep="\t", encoding="utf-8-sig", dtype=str)
    if len(df.columns) == 1:
        df = pd.read_csv(csv_file, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()

    rename_map = {}
    if "Ticker" in df.columns:
        rename_map["Ticker"] = "stock_id"
    if "Name" in df.columns:
        rename_map["Name"] = "name"
    df = df.rename(columns=rename_map)

    if "stock_id" not in df.columns:
        raise ValueError(f"{csv_file} missing Ticker or stock_id column")
    if "name" not in df.columns:
        df["name"] = ""

    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df = df[df["stock_id"] != ""]
    return df[["stock_id", "name"]].to_dict(orient="records")


def build_static_chips(
    stock_list: list[dict],
    output_file: str,
    trend_days=None,
    concentration_threshold=None,
    lookback_days=None,
    workers=None,
    sleep_sec: float = 0.2,
):
    token_status = get_finmind_token_status()
    log_finmind_static_event(
        "generate_static_chips_start",
        source="generate_static_chips",
        status=token_status.get("login_status"),
        message=f"output={output_file}, token={token_status.get('token_masked')}",
    )

    try:
        get_finmind_usage()
    except Exception as exc:
        print(f"Cannot check FinMind usage, continue chip build: {exc}", flush=True)

    rows = []
    for idx, stock in enumerate(stock_list, 1):
        sid = str(stock.get("stock_id", "")).strip()
        print(f"Processing chips {idx}/{len(stock_list)}: {sid} {stock.get('name')}", flush=True)
        rows.append(
            build_chip_row(
                stock,
                trend_days=trend_days,
                concentration_threshold=concentration_threshold,
                lookback_days=lookback_days,
                workers=workers,
            )
        )
        if sleep_sec and sleep_sec > 0:
            time.sleep(float(sleep_sec))

    final_df = normalize_chips_df(pd.DataFrame(rows))
    atomic_write_csv(final_df, output_file)
    status_counts = final_df["chips_status"].astype(str).str.lower().value_counts().to_dict() if not final_df.empty else {}

    log_finmind_static_event(
        "generate_static_chips_end",
        source="generate_static_chips",
        status="completed",
        message=f"updated={len(final_df)}, output={output_file}, status={status_counts}",
    )
    print(f"Static chips rebuild: {status_counts}, total={len(final_df)}, output={output_file}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild AllStatic_Chips.csv for broker chip data.")
    parser.add_argument("--no-prompt", action="store_true", help="Accepted for workflow compatibility; no interactive prompts are used.")
    parser.add_argument("--csv-file", default=None, help="Stock list file. Supports Ticker/Name or stock_id/name columns.")
    parser.add_argument("--output", default=None, help="Chip static output file.")
    parser.add_argument("--trend-days", type=int, default=None, help="Override CHIP_TREND_DAYS.")
    parser.add_argument("--concentration-threshold", type=float, default=None, help="Override CHIP_CONCENTRATION_THRESHOLD.")
    parser.add_argument("--lookback-days", type=int, default=None, help="Optional lookback window for implementations that support it.")
    parser.add_argument("--workers", type=int, default=None, help="Optional parallel worker count for implementations that support it.")
    parser.add_argument("--sleep-sec", type=float, default=None, help="Sleep between stocks.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_file = resolve_csv_file(args.csv_file)
    output_file = resolve_output_file(args.output)
    sleep_sec = args.sleep_sec if args.sleep_sec is not None else float(cfg("CHIP_SLEEP_SEC", 0.2) or 0.2)
    trend_days = args.trend_days if args.trend_days is not None else int(float(cfg("CHIP_TREND_DAYS", 3) or 3))
    concentration_threshold = (
        args.concentration_threshold
        if args.concentration_threshold is not None
        else float(cfg("CHIP_CONCENTRATION_THRESHOLD", 15) or 15)
    )
    lookback_days = args.lookback_days if args.lookback_days is not None else int(float(cfg("CHIP_LOOKBACK_DAYS", 21) or 21))
    workers = args.workers if args.workers is not None else int(float(cfg("CHIP_WORKERS", 1) or 1))

    print(
        "Chip static config: "
        f"csv_file={csv_file}, output={output_file}, trend_days={trend_days}, "
        f"threshold={concentration_threshold}, lookback_days={lookback_days}, "
        f"workers={workers}, sleep_sec={sleep_sec}",
        flush=True,
    )

    try:
        stock_list = load_stock_list(csv_file)
    except Exception as exc:
        print(f"Failed to read source CSV/config: {exc}", flush=True)
        raise

    build_static_chips(
        stock_list=stock_list,
        output_file=output_file,
        trend_days=trend_days,
        concentration_threshold=concentration_threshold,
        lookback_days=lookback_days,
        workers=workers,
        sleep_sec=sleep_sec,
    )


if __name__ == "__main__":
    main()
