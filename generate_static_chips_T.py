from datetime import datetime
import argparse
import os
import time

import pandas as pd

import config
from data_sources import (
    get_chip_analysis,
    get_finmind_token_status,
    get_finmind_user_info,
    log_finmind_static_event,
)

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

# t0 = latest/current available trading day, t1 = previous trading day, t2 = two trading days ago.
CHIP_DAILY_COLS = [
    "chip_date_t0",
    "chip_date_t1",
    "chip_date_t2",
    "chip_concentration_pct_t0",
    "chip_concentration_pct_t1",
    "chip_concentration_pct_t2",
    "main_force_net_t0",
    "main_force_net_t1",
    "main_force_net_t2",
    "broker_diff_t0",
    "broker_diff_t1",
    "broker_diff_t2",
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
] + CHIP_DATA_COLS + CHIP_DAILY_COLS + [
    "chips_updated_at",
    "chips_status",
    "chips_reason",
] + FINMIND_META_COLS

_LAST_FINMIND_USAGE_INFO = None


def now_utc_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def is_blank_value(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def compact_text(text: str, max_len: int = 180) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _first_present(mapping: dict, *keys: str):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_chip_daily_fields(chip: dict) -> dict:
    out = {col: None for col in CHIP_DAILY_COLS}
    rows = chip.get("chip_recent_rows") or chip.get("recent_rows") or chip.get("daily_rows") or chip.get("chip_daily_rows") or []
    if isinstance(rows, list):
        for idx, row in enumerate(rows[:3]):
            if not isinstance(row, dict):
                continue
            suffix = f"t{idx}"
            out[f"chip_date_{suffix}"] = _first_present(row, "date", "Date", "chip_date", "chip_latest_date")
            out[f"chip_concentration_pct_{suffix}"] = _first_present(row, "chip_concentration_pct", "籌碼集中度%", "concentration_pct")
            out[f"main_force_net_{suffix}"] = _first_present(row, "main_force_net", "主力買賣超")
            out[f"broker_diff_{suffix}"] = _first_present(row, "broker_diff", "買賣家數差")
    for idx in range(3):
        suffix = f"t{idx}"
        out[f"chip_date_{suffix}"] = _first_present(chip, f"chip_date_{suffix}", f"chip_latest_date_{suffix}", f"chip_day{idx}_date", f"chip_date_{idx}") or out[f"chip_date_{suffix}"]
        out[f"chip_concentration_pct_{suffix}"] = _first_present(chip, f"chip_concentration_pct_{suffix}", f"chip_day{idx}_concentration_pct", f"chip_concentration_pct_{idx}") or out[f"chip_concentration_pct_{suffix}"]
        out[f"main_force_net_{suffix}"] = _first_present(chip, f"main_force_net_{suffix}", f"chip_day{idx}_main_force_net", f"main_force_net_{idx}") or out[f"main_force_net_{suffix}"]
        out[f"broker_diff_{suffix}"] = _first_present(chip, f"broker_diff_{suffix}", f"chip_day{idx}_broker_diff", f"broker_diff_{idx}") or out[f"broker_diff_{suffix}"]
    out["chip_date_t0"] = out["chip_date_t0"] or chip.get("chip_latest_date")
    out["chip_concentration_pct_t0"] = out["chip_concentration_pct_t0"] or chip.get("chip_concentration_pct")
    out["main_force_net_t0"] = out["main_force_net_t0"] or chip.get("main_force_net")
    out["broker_diff_t0"] = out["broker_diff_t0"] or chip.get("broker_diff")
    return out


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
        print(f"⚠️ FinMind token/user_info check failed: {info.get('message')}", flush=True)
    return used, limit, remain


def normalize_chips_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ORDERED_COLS)
    df = df.copy()
    df.columns = df.columns.str.strip()
    for c in ORDERED_COLS:
        if c not in df.columns:
            df[c] = None
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    return df[ORDERED_COLS]


def atomic_write_csv(df: pd.DataFrame, path: str):
    tmp_path = path + ".tmp"
    df = normalize_chips_df(df)
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, path)


def empty_chip_row(s: dict) -> dict:
    row = {c: None for c in ORDERED_COLS}
    row["stock_id"] = str(s["stock_id"]).strip()
    row["name"] = s.get("name")
    row["chips_updated_at"] = now_utc_str()
    row["chips_status"] = "incomplete"
    row["chips_reason"] = "not processed yet"
    apply_finmind_usage_to_row(row)
    return row


def build_chip_row(s: dict, trend_days=None, concentration_threshold=None) -> dict:
    stock_id = str(s["stock_id"]).strip()
    row = empty_chip_row(s)
    row["chips_updated_at"] = now_utc_str()

    try:
        chip = get_chip_analysis(
            stock_id,
            trend_days=trend_days,
            concentration_threshold=concentration_threshold,
        ) or {}
        for c in CHIP_DATA_COLS:
            row[c] = chip.get(c)

        row.update(_normalize_chip_daily_fields(chip))

        state = str(row.get("chip_signal_state") or "").strip().lower()
        if state and state != "no_data":
            row["chips_status"] = "ok"
            row["chips_reason"] = ""
        else:
            row["chips_status"] = "no_data"
            row["chips_reason"] = compact_text(row.get("chip_signal_text") or "籌碼資料不足")
    except Exception as e:
        row["chips_status"] = "error"
        row["chips_reason"] = compact_text(str(e))

    apply_finmind_usage_to_row(row)
    return row


def load_stock_list():
    csv_file = config.CSV_FILE
    src_df = pd.read_csv(csv_file, sep="\t", encoding="utf-8-sig", dtype=str)
    src_df.columns = src_df.columns.str.strip()
    src_df = src_df.rename(columns={"Ticker": "stock_id", "Name": "name"})
    src_df["stock_id"] = src_df["stock_id"].astype(str).str.strip()
    return src_df.to_dict(orient="records")


def build_static_chips(stock_list, output_file, trend_days=None, concentration_threshold=None, sleep_sec=0.2):
    token_status = get_finmind_token_status()
    log_finmind_static_event(
        "generate_static_chips_start",
        source="generate_static_chips",
        status=token_status.get("login_status"),
        message=f"output={output_file}, token={token_status.get('token_masked')}",
    )

    try:
        get_finmind_usage()
    except Exception as e:
        print(f"Cannot check FinMind usage, continue chip build: {e}", flush=True)

    rows = []
    for i, s in enumerate(stock_list, 1):
        sid = str(s.get("stock_id", "")).strip()
        print(f"Processing chips {i}/{len(stock_list)}: {sid} {s.get('name')}", flush=True)
        rows.append(build_chip_row(s, trend_days=trend_days, concentration_threshold=concentration_threshold))
        if sleep_sec and sleep_sec > 0:
            time.sleep(sleep_sec)

    final_df = normalize_chips_df(pd.DataFrame(rows))
    atomic_write_csv(final_df, output_file)
    status_counts = final_df["chips_status"].astype(str).str.lower().value_counts().to_dict() if not final_df.empty else {}

    log_finmind_static_event(
        "generate_static_chips_end",
        source="generate_static_chips",
        status="completed",
        message=f"updated={len(final_df)}, output={output_file}, status={status_counts}",
    )
    print(f"Static_Chips rebuild: {status_counts}, total={len(final_df)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Rebuild AllStatic_Chips.csv for broker chip data.")
    parser.add_argument("--output", default=os.getenv("STATIC_CHIPS_FILE", getattr(config, "STATIC_CHIPS_OUTPUT_FILE", "AllStatic_Chips.csv")))
    parser.add_argument("--trend-days", type=int, default=None, help="Override CHIP_TREND_DAYS.")
    parser.add_argument("--concentration-threshold", type=float, default=None, help="Override CHIP_CONCENTRATION_THRESHOLD.")
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep between stocks.")
    args = parser.parse_args()

    try:
        stock_list = load_stock_list()
    except Exception as e:
        print(f"Failed to read source CSV/config: {e}", flush=True)
        return

    build_static_chips(
        stock_list=stock_list,
        output_file=args.output,
        trend_days=args.trend_days,
        concentration_threshold=args.concentration_threshold,
        sleep_sec=args.sleep_sec,
    )


if __name__ == "__main__":
    main()
