import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from FinMind.data import DataLoader
from loguru import logger

# Standardize on FINMIND_TOKEN only.
FINMIND_token = os.getenv("FINMIND_TOKEN")
FINMIND_TOKEN_SOURCE = "FINMIND_TOKEN" if FINMIND_token else ""
headers = {"Authorization": f"Bearer {FINMIND_token}"} if FINMIND_token else {}
API_URL = 'https://api.finmindtrade.com/api/v4/data'
USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"
FINMIND_USAGE_LOG_FILE = os.getenv(
    "FINMIND_USAGE_LOG_FILE", "finmind_token_usage_log.csv")

api = DataLoader()
FINMIND_TOKEN_LOGIN_STATUS = "missing_token"
FINMIND_TOKEN_LOGIN_MESSAGE = "FINMIND_TOKEN is not set"
if FINMIND_token:
    try:
        api.login_by_token(api_token=FINMIND_token)
        FINMIND_TOKEN_LOGIN_STATUS = "ok"
        FINMIND_TOKEN_LOGIN_MESSAGE = "DataLoader.login_by_token succeeded"
    except Exception as e:
        FINMIND_TOKEN_LOGIN_STATUS = "error"
        FINMIND_TOKEN_LOGIN_MESSAGE = str(e)

_INITIAL_QUOTA_PRINTED = False
FINMIND_API_CALL_COUNT = 0
FINMIND_DATASET_CALL_COUNTS = {}

# 停用所有來自 FinMind 的 Log 訊息
logger.remove()
logging.getLogger('FinMind').setLevel(logging.WARNING)


def _mask_token(token=None):
    """Return a safe token display value, never the full token."""
    token = token if token is not None else FINMIND_token
    if not token:
        return ""
    token = str(token)
    if len(token) <= 8:
        return "*" * len(token)
    return token[:4] + "..." + token[-4:]


def _append_finmind_usage_event(
    event,
    source="",
    stock_id="",
    dataset="",
    status="",
    status_code="",
    user_count=None,
    api_request_limit=None,
    remain=None,
    message="",
):
    """Append token/login/quota/API usage evidence to a CSV audit log."""
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "source": source,
        "stock_id": stock_id,
        "dataset": dataset,
        "token_present": bool(FINMIND_token),
        "token_source": FINMIND_TOKEN_SOURCE,
        "token_masked": _mask_token(),
        "login_status": FINMIND_TOKEN_LOGIN_STATUS,
        "login_message": FINMIND_TOKEN_LOGIN_MESSAGE,
        "request_count": FINMIND_API_CALL_COUNT,
        "dataset_request_count": FINMIND_DATASET_CALL_COUNTS.get(dataset, 0),
        "user_count": user_count,
        "api_request_limit": api_request_limit,
        "remain": remain,
        "status": status,
        "status_code": status_code,
        "message": str(message or "")[:300],
    }

    try:
        log_path = FINMIND_USAGE_LOG_FILE
        exists = os.path.exists(log_path)
        pd.DataFrame([row]).to_csv(
            log_path,
            mode="a",
            header=not exists,
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as e:
        print(f"⚠️ cannot write FinMind usage log: {e}", flush=True)


def log_finmind_static_event(event, message="", **kwargs):
    """Public helper for generate_static_csv.py to write the same audit log."""
    _append_finmind_usage_event(event=event, message=message, **kwargs)


def _record_finmind_request(source, stock_id="", dataset=""):
    """Count and log each FinMind API/DataLoader call in one place."""
    global FINMIND_API_CALL_COUNT
    FINMIND_API_CALL_COUNT += 1
    FINMIND_DATASET_CALL_COUNTS[dataset] = FINMIND_DATASET_CALL_COUNTS.get(
        dataset, 0) + 1

    _append_finmind_usage_event(
        event="api_call",
        source=source,
        stock_id=stock_id,
        dataset=dataset,
        status="sent",
        message="FinMind request sent with token in query params and/or Authorization header",
    )


def get_finmind_token_status():
    """Return local evidence that the FinMind token was loaded and DataLoader login ran."""
    return {
        "token_present": bool(FINMIND_token),
        "token_source": FINMIND_TOKEN_SOURCE,
        "token_masked": _mask_token(),
        "login_status": FINMIND_TOKEN_LOGIN_STATUS,
        "login_message": FINMIND_TOKEN_LOGIN_MESSAGE,
        "request_count": FINMIND_API_CALL_COUNT,
        "dataset_call_counts": dict(FINMIND_DATASET_CALL_COUNTS),
        "usage_log_file": FINMIND_USAGE_LOG_FILE,
    }


def get_finmind_user_info(write_log=True, source="user_info"):
    """
    Validate the token against FinMind user_info and return usage information.

    This is the strongest runtime check that the token is accepted by FinMind:
    - token_present/token_source proves the environment variable was loaded
    - HTTP status/body proves FinMind accepted or rejected it
    - user_count/api_request_limit/remain shows actual account quota usage
    """
    if not FINMIND_token:
        info = {
            "ok": False,
            "token_present": False,
            "token_source": "",
            "token_masked": "",
            "login_status": FINMIND_TOKEN_LOGIN_STATUS,
            "login_message": FINMIND_TOKEN_LOGIN_MESSAGE,
            "user_count": None,
            "api_request_limit": None,
            "remain": None,
            "status_code": None,
            "message": "FINMIND_TOKEN is not set",
        }
        if write_log:
            _append_finmind_usage_event(
                event="token_check",
                source=source,
                status="missing_token",
                message=info["message"],
            )
        return info

    try:
        res = requests.get(USER_INFO_URL, headers=headers, timeout=300)
        data = _safe_response_json(res)
        used = data.get("user_count")
        limit = data.get("api_request_limit")

        try:
            used_int = int(used or 0)
            limit_int = int(limit or 0)
            remain = max(limit_int - used_int, 0) if limit_int else None
        except Exception:
            used_int = used
            limit_int = limit
            remain = None

        ok = res.status_code == 200 and not data.get("error")
        msg = data.get("msg") or data.get(
            "message") or data.get("status") or res.text[:200]
        info = {
            "ok": ok,
            "token_present": True,
            "token_source": FINMIND_TOKEN_SOURCE,
            "token_masked": _mask_token(),
            "login_status": FINMIND_TOKEN_LOGIN_STATUS,
            "login_message": FINMIND_TOKEN_LOGIN_MESSAGE,
            "user_count": used_int,
            "api_request_limit": limit_int,
            "remain": remain,
            "status_code": res.status_code,
            "message": msg,
        }

        if write_log:
            _append_finmind_usage_event(
                event="token_check",
                source=source,
                status="ok" if ok else "error",
                status_code=res.status_code,
                user_count=used_int,
                api_request_limit=limit_int,
                remain=remain,
                message=msg,
            )
        return info

    except Exception as e:
        info = {
            "ok": False,
            "token_present": True,
            "token_source": FINMIND_TOKEN_SOURCE,
            "token_masked": _mask_token(),
            "login_status": FINMIND_TOKEN_LOGIN_STATUS,
            "login_message": FINMIND_TOKEN_LOGIN_MESSAGE,
            "user_count": None,
            "api_request_limit": None,
            "remain": None,
            "status_code": None,
            "message": str(e),
        }
        if write_log:
            _append_finmind_usage_event(
                event="token_check",
                source=source,
                status="error",
                message=str(e),
            )
        return info


def _safe_response_json(res):
    """避免 API 回傳非 JSON 時，印錯誤又讓程式中斷。"""
    try:
        return res.json()
    except Exception:
        return {}


def _extract_remaining_quota(data, res=None):
    """從 FinMind 回傳 body/header 中盡量抓出剩餘次數/額度資訊。"""
    if res is not None:
        for key in [
            "X-RateLimit-Remaining", "X-Rate-Limit-Remaining",
            "RateLimit-Remaining", "x-ratelimit-remaining"
        ]:
            value = res.headers.get(key)
            if value not in [None, ""]:
                return f"header {key}={value}"

    for key in [
        "api_usage", "api_remaining", "remaining", "remaining_count",
        "quota", "limit", "msg", "message", "status"
    ]:
        value = data.get(key) if isinstance(data, dict) else None
        if value not in [None, ""]:
            text = str(value)
            if any(word in text.lower() for word in [
                "remaining", "quota", "limit", "api", "剩餘", "次數", "額度"
            ]):
                return text
    return None


def _print_initial_quota_once(data, res=None):
    """第一次收到 API 回應時，印出起始剩餘次數。"""
    global _INITIAL_QUOTA_PRINTED
    if _INITIAL_QUOTA_PRINTED:
        return

    quota_msg = _extract_remaining_quota(data, res)
    if quota_msg is None:
        quota_msg = "API 回應未提供剩餘次數欄位"

    print(f"🔢 FinMind API 起始剩餘次數: {quota_msg}")
    _INITIAL_QUOTA_PRINTED = True


def _print_api_status_error(source, stock_id, res, data=None):
    """非 200/異常 API 狀態時，統一印出 status code 與訊息。"""
    if data is None:
        data = _safe_response_json(res)

    msg = data.get("msg") or data.get(
        "message") or data.get("status") or res.text[:200]
    print(
        f"❌ {source} API error {stock_id}: "
        f"status_code={res.status_code}, msg={msg}"
    )


def get_stock_data(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': str(stock_id),
            'start_date': '2023-01-01',
            'token': FINMIND_token,
        }
        _record_finmind_request("get_stock_data", stock_id, "TaiwanStockPrice")
        res = requests.get(API_URL, params=params,
                           headers=headers, timeout=300)
        data = _safe_response_json(res)
        _print_initial_quota_once(data, res)

        if res.status_code == 402:
            _print_api_status_error('get_stock_data', stock_id, res, data)
            raise RuntimeError(
                f"FinMind quota exceeded for {stock_id}: {data.get('msg')}")

        if res.status_code != 200:
            _print_api_status_error('get_stock_data', stock_id, res, data)
            return pd.DataFrame()

        if 'data' not in data or len(data['data']) == 0:
            print(
                f"⚠️ get_stock_data empty {stock_id}: status={res.status_code}, msg={data.get('msg')}")
            return pd.DataFrame()

        df = pd.DataFrame(data['data'])

        volume_col = None
        for c in ['Trading_Volume', 'trading_volume', 'Trading_Volume_1000']:
            if c in df.columns:
                volume_col = c
                break

        required_cols = ['date', 'open', 'close', 'max', 'min']
        if volume_col:
            required_cols.append(volume_col)

        df = df[required_cols].copy()
        df['date'] = pd.to_datetime(df['date'])

        if volume_col:
            df['volume'] = pd.to_numeric(df[volume_col], errors='coerce')
            if df['volume'].max() > 100000:
                df['volume'] = df['volume'] / 1000
        else:
            df['volume'] = None

        df = df.dropna(subset=['open', 'close', 'max',
                       'min']).sort_values('date')

        return df
    except RuntimeError:
        raise
    except Exception as e:
        print(f'❌ get_stock_data error {stock_id}: {e}')
        return pd.DataFrame()


def get_revenue_raw(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockMonthRevenue',  # 🔥 月營收
            'data_id': stock_id,
            'start_date': '2022-01-01',
            'token': FINMIND_token,
        }

        _record_finmind_request(
            "revenue source", stock_id, "TaiwanStockMonthRevenue")
        res = requests.get(API_URL, params=params,
                           headers=headers, timeout=300)
        res_data = _safe_response_json(res)
        _print_initial_quota_once(res_data, res)

        if res.status_code != 200:
            _print_api_status_error('revenue source', stock_id, res, res_data)
            return []

        data = res_data.get('data', [])
        return data

    except Exception as e:
        print(f'❌ revenue source error {stock_id}: {e}')
        return []


def get_profit_ratio(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockFinancialStatements',
            'data_id': stock_id,
            'start_date': '2020-01-01',
            'token': FINMIND_token,
        }
        _record_finmind_request("profit source", stock_id, "TaiwanStockFinancialStatements")
        res = requests.get(API_URL, params=params, headers=headers, timeout=300)
        data = _safe_response_json(res)
        _print_initial_quota_once(data, res)

        if res.status_code != 200:
            _print_api_status_error('profit source', stock_id, res, data)
            return pd.DataFrame()

        return pd.DataFrame(data.get('data', []))
    except Exception as e:
        print(f'❌ profit source error {stock_id}: {e}')
        return pd.DataFrame()


def get_eps_raw(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockFinancialStatements',
            'data_id': stock_id,
            'start_date': '2020-01-01',
            'token': FINMIND_token,
        }
        _record_finmind_request("EPS source", stock_id,
                                "TaiwanStockFinancialStatements")
        res = requests.get(API_URL, params=params,
                           headers=headers, timeout=300)
        data = _safe_response_json(res)
        _print_initial_quota_once(data, res)

        if res.status_code != 200:
            _print_api_status_error('EPS source', stock_id, res, data)
            return []

        return data.get('data', [])
    except Exception as e:
        print(f'❌ EPS source error {stock_id}: {e}')
        return []


def get_dividend_raw(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockDividend',
            'data_id': stock_id,
            'start_date': '2020-01-01',
            'token': FINMIND_token,
        }
        _record_finmind_request(
            "dividend source", stock_id, "TaiwanStockDividend")
        res = requests.get(API_URL, params=params,
                           headers=headers, timeout=300)
        data = _safe_response_json(res)
        _print_initial_quota_once(data, res)

        if res.status_code != 200:
            _print_api_status_error('dividend source', stock_id, res, data)
            return []
        return data.get('data', [])
    except Exception as e:
        print(f'❌ dividend source error {stock_id}: {e}')
        return []


def get_per_raw(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockPER',
            'data_id': stock_id,
            'start_date': '2023-01-01',
            'token': FINMIND_token,
        }
        _record_finmind_request("PER source", stock_id, "TaiwanStockPER")
        res = requests.get(API_URL, params=params,
                           headers=headers, timeout=300)
        data = _safe_response_json(res)
        _print_initial_quota_once(data, res)

        if res.status_code != 200:
            _print_api_status_error('PER source', stock_id, res, data)
            return []

        return data.get('data', [])
    except Exception as e:
        print(f'❌ PER source error {stock_id}: {e}')
        return []


def get_per_pbr_90d_stats(stock_id, days=90):
    """
    Latest valid PER/PBR plus rolling high/low.
    If the newest FinMind row has blank PER/PBR, walk backward to the newest valid value.
    """
    def safe_round(x):
        try:
            if pd.isna(x):
                return None
            return round(float(x), 2)
        except Exception:
            return None

    empty = {
        "per": None, "per_90d_high": None, "per_90d_low": None, "per_is_prev": False,
        "pbr": None, "pbr_90d_high": None, "pbr_90d_low": None, "pbr_is_prev": False,
    }

    try:
        params = {
            "dataset": "TaiwanStockPER",
            "data_id": stock_id,
            "start_date": (datetime.today() - timedelta(days=max(days * 3, 240))).strftime("%Y-%m-%d"),
            "token": FINMIND_token,
        }
        _record_finmind_request("PER/PBR 90D", stock_id, "TaiwanStockPER")
        res = requests.get(API_URL, params=params, headers=headers, timeout=300)
        res_data = _safe_response_json(res)
        _print_initial_quota_once(res_data, res)

        if res.status_code != 200:
            _print_api_status_error('PER/PBR 90D', stock_id, res, res_data)
            return empty

        data = res_data.get("data", [])
        if not data:
            return empty

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            return empty
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            return empty

        latest_row_date = df["date"].max()
        cutoff = latest_row_date - pd.Timedelta(days=days)
        df_win = df[df["date"] >= cutoff].copy()
        if df_win.empty:
            df_win = df.copy()

        per_col = next((c for c in ["price_earning_ratio", "PER", "per"] if c in df_win.columns), None)
        pbr_col = next((c for c in ["price_book_ratio", "PBR", "pbr"] if c in df_win.columns), None)

        def latest_valid(col):
            if not col:
                return None, None, False
            s = pd.to_numeric(df_win[col], errors="coerce")
            valid = df_win.loc[s.notna(), ["date", col]].copy()
            if valid.empty:
                return None, None, False
            latest_valid_date = valid["date"].max()
            latest_value = valid.loc[valid["date"] == latest_valid_date, col].iloc[-1]
            return safe_round(latest_value), latest_valid_date, bool(latest_valid_date < latest_row_date)

        per, per_date, per_is_prev = latest_valid(per_col)
        pbr, pbr_date, pbr_is_prev = latest_valid(pbr_col)

        if per_col:
            per_s = pd.to_numeric(df_win[per_col], errors="coerce").dropna()
            per_high = safe_round(per_s.max()) if not per_s.empty else per
            per_low = safe_round(per_s.min()) if not per_s.empty else per
        else:
            per_high = per_low = None

        if pbr_col:
            pbr_s = pd.to_numeric(df_win[pbr_col], errors="coerce").dropna()
            pbr_high = safe_round(pbr_s.max()) if not pbr_s.empty else pbr
            pbr_low = safe_round(pbr_s.min()) if not pbr_s.empty else pbr
        else:
            pbr_high = pbr_low = None

        return {
            "per": per,
            "per_90d_high": per_high if per_high is not None else per,
            "per_90d_low": per_low if per_low is not None else per,
            "per_is_prev": per_is_prev,
            "pbr": pbr,
            "pbr_90d_high": pbr_high if pbr_high is not None else pbr,
            "pbr_90d_low": pbr_low if pbr_low is not None else pbr,
            "pbr_is_prev": pbr_is_prev,
        }
    except Exception as e:
        print(f"❌ PER/PBR 90D error {stock_id}: {e}")
        return empty

