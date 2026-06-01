from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pandas as pd

# Reduce noisy third-party logs as early as possible.
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
except Exception:
    pass

try:
    import config  # type: ignore
except Exception:
    config = None

from data_sources import (  # noqa: E402
    get_chip_analysis,
    get_finmind_token_status,
    get_finmind_user_info,
)

SCRIPT_VERSION = "chips-static-v20260601-quiet-no-action-log-002"
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

ORDERED_COLS = (
    ["stock_id", "name"]
    + CHIP_DATA_COLS
    + ["chips_updated_at", "chips_status", "chips_reason"]
    + FINMIND_META_COLS
)

_LAST_FINMIND_USAGE_INFO: dict | None = None

NOISY_LOG_PATTERNS = (
    "chip analysis request:",
    "chip analysis response status:",
    "FinMind API 起始剩餘次數",
    "Login success",
)


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def cfg(name: str, default: Any = None) -> Any:
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    if config is not None and getattr(config, name, None) not in (None, ""):
        return getattr(config, name)
    return default


def read_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def read_int(value: Any, default: int, min_value: int = 1, max_value: int = 999) -> int:
    try:
        num = int(float(str(value).strip()))
    except Exception:
        num = int(default)
    return max(min_value, min(num, max_value))


def read_float(
    value: Any,
    default: float,
    min_value: float = 0.0,
    max_value: float = 100.0,
) -> float:
    try:
        num = float(str(value).strip())
    except Exception:
        num = float(default)
    return max(min_value, min(num, max_value))


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
        "finmind_token_status": info.get("login_status")
        or ("ok" if info.get("token_present") else "missing_token"),
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


def get_finmind_usage(write_log: bool = False) -> dict:
    """
    Check token/quota once.

    write_log defaults to False so this script does not create chips_analysis_action.log
    or other FinMind usage audit files.
    """
    global _LAST_FINMIND_USAGE_INFO
    info = get_finmind_user_info(
        write_log=write_log, source="generate_static_chips")
    _LAST_FINMIND_USAGE_INFO = info

    print(
        "FinMind: "
        f"token={bool(info.get('token_present'))}, "
        f"login={info.get('login_status')}, "
        f"usage={info.get('user_count')}/{info.get('api_request_limit')}, "
        f"remain={info.get('remain')}",
        flush=True,
    )

    if not info.get("ok"):
        print(
            f"WARNING FinMind user_info check failed: {info.get('message')}", flush=True)

    return info


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


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
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


def _filter_noisy_output(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        if any(pattern in line for pattern in NOISY_LOG_PATTERNS):
            continue
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield

    filtered = _filter_noisy_output(buf.getvalue())
    if filtered:
        print(filtered, flush=True)


def call_get_chip_analysis(
    stock_id: str,
    trend_days: int,
    concentration_threshold: float,
    lookback_days: int | None = None,
    day_workers: int | None = None,
    suppress_api_logs: bool = True,
) -> dict:
    kwargs = {
        "trend_days": trend_days,
        "concentration_threshold": concentration_threshold,
        "lookback_days": lookback_days,
        "workers": day_workers,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    with maybe_suppress_stdout(suppress_api_logs):
        try:
            sig = inspect.signature(get_chip_analysis)
            supported = {
                key: value for key, value in kwargs.items() if key in sig.parameters
            }
            return get_chip_analysis(stock_id, **supported) or {}
        except (TypeError, ValueError):
            return (
                get_chip_analysis(
                    stock_id,
                    trend_days=trend_days,
                    concentration_threshold=concentration_threshold,
                )
                or {}
            )


def build_chip_row(
    stock: dict,
    trend_days: int,
    concentration_threshold: float,
    lookback_days: int | None = None,
    day_workers: int | None = None,
    suppress_api_logs: bool = True,
) -> dict:
    row = empty_chip_row(stock)
    row["chips_updated_at"] = now_utc_str()
    stock_id = str(stock.get("stock_id", "")).strip()

    try:
        chip = call_get_chip_analysis(
            stock_id,
            trend_days=trend_days,
            concentration_threshold=concentration_threshold,
            lookback_days=lookback_days,
            day_workers=day_workers,
            suppress_api_logs=suppress_api_logs,
        )

        for col in CHIP_DATA_COLS:
            row[col] = chip.get(col)

        state = str(row.get("chip_signal_state") or "").strip().lower()

        if state and state not in {"no_data", "error"}:
            row["chips_status"] = "ok"
            row["chips_reason"] = ""
        elif state == "error":
            row["chips_status"] = "error"
            row["chips_reason"] = compact_text(
                row.get("chip_signal_text") or "籌碼資料錯誤"
            )
        else:
            row["chips_status"] = "no_data"
            row["chips_reason"] = compact_text(
                row.get("chip_signal_text") or "籌碼資料不足"
            )

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


def should_log_progress(index: int, total: int, log_every: int) -> bool:
    return index == 1 or index == total or (log_every > 0 and index % log_every == 0)


def summarize_row(row: dict) -> str:
    return (
        f"{row.get('stock_id')} {row.get('name') or ''} "
        f"status={row.get('chips_status')} "
        f"date={row.get('chip_latest_date') or '-'} "
        f"main={row.get('main_force_net') if row.get('main_force_net') is not None else '-'} "
        f"diff={row.get('broker_diff') if row.get('broker_diff') is not None else '-'} "
        f"state={row.get('chip_signal_state') or '-'}"
    )


def build_static_chips(
    stock_list: list[dict],
    output_file: str,
    trend_days: int,
    concentration_threshold: float,
    lookback_days: int | None = None,
    workers: int = 1,
    day_workers: int | None = None,
    sleep_sec: float = 0.0,
    log_every: int = 25,
    verbose: bool = False,
    suppress_api_logs: bool = True,
) -> pd.DataFrame:
    started = datetime.utcnow()

    try:
        get_finmind_usage(write_log=False)
    except Exception as exc:
        print(f"Cannot check FinMind usage, continue: {exc}", flush=True)

    total = len(stock_list)
    workers = max(1, min(int(workers or 1), max(total, 1)))

    rows_by_index: dict[int, dict] = {}
    notable: list[str] = []

    print(
        f"Build chips: total={total}, workers={workers}, trend_days={trend_days}, "
        f"threshold={concentration_threshold:g}, lookback_days={lookback_days}, output={output_file}",
        flush=True,
    )

    def task(item: tuple[int, dict]) -> tuple[int, dict]:
        idx, stock = item

        if sleep_sec and sleep_sec > 0:
            import time

            time.sleep(float(sleep_sec) * ((idx - 1) % workers))

        row = build_chip_row(
            stock,
            trend_days=trend_days,
            concentration_threshold=concentration_threshold,
            lookback_days=lookback_days,
            day_workers=day_workers,
            suppress_api_logs=suppress_api_logs and not verbose,
        )
        return idx, row

    if workers <= 1:
        for idx, stock in enumerate(stock_list, 1):
            idx, row = task((idx, stock))
            rows_by_index[idx] = row

            status = str(row.get("chips_status") or "other").lower()
            if status in {"error", "no_data"}:
                notable.append(summarize_row(row) +
                               f" reason={row.get('chips_reason') or ''}")

            if verbose or status in {"error", "no_data"} or should_log_progress(idx, total, log_every):
                print(f"[{idx}/{total}] {summarize_row(row)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(task, (idx, stock)): idx
                for idx, stock in enumerate(stock_list, 1)
            }

            completed = 0
            for future in as_completed(futures):
                idx = futures[future]

                try:
                    idx, row = future.result()
                except Exception as exc:
                    stock = stock_list[idx - 1]
                    row = empty_chip_row(stock)
                    row["chips_status"] = "error"
                    row["chips_reason"] = compact_text(str(exc))

                rows_by_index[idx] = row
                completed += 1

                status = str(row.get("chips_status") or "other").lower()
                if status in {"error", "no_data"}:
                    notable.append(summarize_row(row) +
                                   f" reason={row.get('chips_reason') or ''}")

                if verbose or status in {"error", "no_data"} or should_log_progress(completed, total, log_every):
                    print(
                        f"[{completed}/{total}] {summarize_row(row)}", flush=True)

    rows = [rows_by_index[idx]
            for idx in range(1, total + 1) if idx in rows_by_index]

    final_df = normalize_chips_df(pd.DataFrame(rows))
    atomic_write_csv(final_df, output_file)

    elapsed = (datetime.utcnow() - started).total_seconds()
    status_counts = (
        final_df["chips_status"].astype(
            str).str.lower().value_counts().to_dict()
        if not final_df.empty
        else {}
    )

    print(
        f"Done: rows={len(final_df)}, status={status_counts}, elapsed={elapsed:.1f}s, output={output_file}",
        flush=True,
    )

    if notable and not verbose:
        print("Notable rows:", flush=True)
        for line in notable[:10]:
            print("- " + line, flush=True)
        if len(notable) > 10:
            print(f"- ... {len(notable) - 10} more", flush=True)

    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild AllStatic_Chips.csv for broker chip data."
    )

    parser.add_argument("--version", action="store_true",
                        help="Print script version and exit.")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Accepted for workflow compatibility; no prompts are used.")
    parser.add_argument("--csv-file", default=None,
                        help="Stock list file. Supports Ticker/Name or stock_id/name columns.")
    parser.add_argument("--output", default=None,
                        help="Chip static output file.")
    parser.add_argument("--trend-days", type=int,
                        default=None, help="Override CHIP_TREND_DAYS.")
    parser.add_argument("--concentration-threshold", type=float,
                        default=None, help="Override CHIP_CONCENTRATION_THRESHOLD.")
    parser.add_argument("--lookback-days", type=int, default=None,
                        help="Optional lookback window for implementations that support it.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel stock workers. Use a small value to avoid API limits.")
    parser.add_argument("--day-workers", type=int, default=None,
                        help="Optional per-stock day workers if get_chip_analysis supports it.")
    parser.add_argument("--sleep-sec", type=float, default=None,
                        help="Small request stagger between parallel tasks.")
    parser.add_argument("--log-every", type=int, default=None,
                        help="Print one progress line every N completed stocks. 0 = only errors/no_data and final summary.")
    parser.add_argument("--verbose", action="store_true", default=read_bool_env(
        "CHIP_VERBOSE", False), help="Print every stock/API result.")
    parser.add_argument(
        "--suppress-api-logs",
        dest="suppress_api_logs",
        action="store_true",
        default=read_bool_env("CHIP_SUPPRESS_API_LOGS", True),
        help="Suppress noisy API request/status logs from data_sources.",
    )
    parser.add_argument(
        "--no-suppress-api-logs",
        dest="suppress_api_logs",
        action="store_false",
        help="Show noisy API request/status logs from data_sources.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.version:
        print(SCRIPT_VERSION)
        return

    csv_file = resolve_csv_file(args.csv_file)
    output_file = resolve_output_file(args.output)

    sleep_sec = (
        args.sleep_sec
        if args.sleep_sec is not None
        else read_float(cfg("CHIP_SLEEP_SEC", 0.0), 0.0, 0.0, 10.0)
    )

    trend_days = read_int(
        args.trend_days if args.trend_days is not None else cfg(
            "CHIP_TREND_DAYS", 3),
        3,
        1,
        20,
    )

    concentration_threshold = read_float(
        args.concentration_threshold
        if args.concentration_threshold is not None
        else cfg("CHIP_CONCENTRATION_THRESHOLD", 15),
        15.0,
        0.0,
        100.0,
    )

    lookback_days = read_int(
        args.lookback_days if args.lookback_days is not None else cfg(
            "CHIP_LOOKBACK_DAYS", 21),
        21,
        3,
        120,
    )

    workers = read_int(
        args.workers if args.workers is not None else cfg("CHIP_WORKERS", 4),
        4,
        1,
        12,
    )

    day_workers = None if args.day_workers is None else read_int(
        args.day_workers, 1, 1, 16)

    log_every = read_int(
        args.log_every if args.log_every is not None else cfg(
            "CHIP_LOG_EVERY", 25),
        25,
        0,
        10000,
    )

    stock_list = load_stock_list(csv_file)

    build_static_chips(
        stock_list=stock_list,
        output_file=output_file,
        trend_days=trend_days,
        concentration_threshold=concentration_threshold,
        lookback_days=lookback_days,
        workers=workers,
        day_workers=day_workers,
        sleep_sec=sleep_sec,
        log_every=log_every,
        verbose=bool(args.verbose),
        suppress_api_logs=bool(args.suppress_api_logs),
    )


if __name__ == "__main__":
    main()
