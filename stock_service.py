import os
import pandas as pd
import numpy as np

from data_sources import get_stock_data
from financial_analysis import (
    calc_eps_score,
    calc_margin_score,
    calc_trend_score,
    get_dividend_yield,
)
from signals import get_tech_signal
from technical_indicators import add_indicators, get_kd_trend, get_bb_trend, get_MABias


STATIC_CSV_PATH = os.getenv("STATIC_CSV_FILE", "AllStatic.csv")
_STATIC_MAP_CACHE = None
_STATIC_MAP_MTIME = None


def get_price_90d_high_low(df):
    df_90 = df.tail(90)
    max_price90 = pd.to_numeric(df_90["max"], errors="coerce").max()
    min_price90 = pd.to_numeric(df_90["min"], errors="coerce").min()

    if pd.isna(max_price90) or pd.isna(min_price90):
        return {
            "price_90d_high": None,
            "price_90d_low": None,
        }

    return {
        "price_90d_high": float(max_price90),
        "price_90d_low": float(min_price90),
    }


def load_static_map(static_csv_path=STATIC_CSV_PATH, force_reload=False):
    global _STATIC_MAP_CACHE, _STATIC_MAP_MTIME

    try:
        if not os.path.exists(static_csv_path):
            print(f"⚠️ 找不到靜態資料檔: {static_csv_path}")
            return {}

        mtime = os.path.getmtime(static_csv_path)
        if (not force_reload) and _STATIC_MAP_CACHE is not None and _STATIC_MAP_MTIME == mtime:
            return _STATIC_MAP_CACHE

        df = pd.read_csv(static_csv_path, encoding="utf-8-sig",
                         dtype={"stock_id": str})
        df.columns = df.columns.str.strip()

        if "stock_id" not in df.columns:
            print(f"⚠️ AllStatic.csv 缺少 stock_id 欄位: {static_csv_path}")
            return {}

        # 將 NaN 轉為 None，方便後續使用
        df = df.where(pd.notna(df), None)

        static_map = {}
        for _, row in df.iterrows():
            stock_id = str(row["stock_id"]).strip()
            static_map[stock_id] = row.to_dict()

        _STATIC_MAP_CACHE = static_map
        _STATIC_MAP_MTIME = mtime
        print(f"✅ 已載入靜態資料: {static_csv_path}, 筆數={len(static_map)}")
        return static_map

    except Exception as e:
        print(f"❌ 讀取 AllStatic.csv 失敗: {e}")
        return {}


def to_float_or_none(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def to_int_or_none(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return None


def process_stock(s, static_map=None):
    stock_id = str(s["stock_id"])
    name = s["name"]

    base = {
        "name": name,
        "code": stock_id,
        "price": None,
        "chg": None,
        "chgPct": None,
        "amp": None,
        "sig": 0,
        "signal": "資料異常",
        "score": 0,
        "signal_text": "資料異常",
        "reason": "",
        "entry_note": "",
    }

    static_row = (static_map or {}).get(stock_id, {})

    try:
        df = get_stock_data(stock_id)

        if df is None:
            x = base.copy()
            x.update({
                "signal": "無資料",
                "signal_text": "查無資料",
                "reason": "get_stock_data 回傳 None",
            })
            x.update(_build_static_fields(static_row))
            return x

        if df.empty:
            x = base.copy()
            x.update({
                "signal": "無資料",
                "signal_text": "查無資料",
                "reason": "股價資料為空",
            })
            x.update(_build_static_fields(static_row))
            return x

        if len(df) < 90:
            x = base.copy()
            x.update({
                "signal": "資料不足",
                "signal_text": "資料不足",
                "reason": f"歷史資料不足90日，僅有 {len(df)} 筆",
            })
            x.update(_build_static_fields(static_row))
            return x

        df = add_indicators(df)
        latest, prev = df.iloc[-1], df.iloc[-2]
        price_stats = get_price_90d_high_low(df)
        max_price = latest["max"]
        min_price = latest["min"]
        chg = latest["close"] - prev["close"]

        chgPct = round((chg / prev["close"]) * 100, 2)
        chgamp = latest["max"] - latest["min"]
        amp = round((chgamp / prev["close"]) * 100, 2)

        try:
            yield_raw = get_dividend_yield(stock_id, latest["close"])
        except Exception as e:
            print(f"❌ dividend error {stock_id}: {e}")
            yield_raw = None

        dividend = None
        yield_value = None
        if isinstance(yield_raw, dict):
            dividend = yield_raw.get("dividend")
            yield_value = yield_raw.get("yield")
        elif isinstance(yield_raw, (int, float)):
            yield_value = float(yield_raw)

        try:
            ma_stats = get_MABias(df) or {}
        except Exception as e:
            print(f"❌ ma bias error {stock_id}: {e}")
            ma_stats = {}

        safe_ma_stats = {}
        for k2, v2 in ma_stats.items():
            if v2 is None or pd.isna(v2):
                safe_ma_stats[k2.lower()] = None
            else:
                safe_ma_stats[k2.lower()] = float(v2)

        k = float(latest["K"]) if pd.notna(latest["K"]) else None
        d = float(latest["D"]) if pd.notna(latest["D"]) else None
        prev_k = float(prev["K"]) if pd.notna(prev["K"]) else None
        prev_d = float(prev["D"]) if pd.notna(prev["D"]) else None

        kd_score = 0
        if None not in (k, d, prev_k, prev_d):
            if k > d and prev_k <= prev_d:
                kd_score = 1
            elif k < d and prev_k >= prev_d:
                kd_score = -1
            elif k > d:
                kd_score = 0.5
            elif k < d:
                kd_score = -0.5

        kd_trend = get_kd_trend(
            df) or {"kd_3d_up": None, "kd_trend": None, "kd_score": None}
        bb_trend = get_bb_trend(
            df) or {"bb_3d_up": None, "bb_trend": None, "bb_score": None}
        k_trend = kd_trend.get("kd_trend")
        d_trend = None

        ma18 = latest["MA18"] if pd.notna(latest["MA18"]) else None
        prev_ma18 = prev["MA18"] if pd.notna(prev["MA18"]) else None
        close = latest["close"]
        prev_close = prev["close"]

        volume = latest.get("volume", None)
        prev_volume = prev.get("volume", None)
        prev2 = df.iloc[-3]
        prev2_volume = prev2.get("volume", None)
        volume_ratio = None
        volume_add = None

        if pd.notna(volume) and pd.notna(prev_volume) and prev_volume > 0:
            volume_ratio = round((volume / prev_volume - 1) * 100, 2)
            volume_add = int(volume - prev_volume)

        bb_upper = latest["BB_upper"] if "BB_upper" in latest else None
        bb_lower = latest["BB_lower"] if "BB_lower" in latest else None
        bb_pct = None
        if pd.notna(bb_upper) and pd.notna(bb_lower) and bb_upper != bb_lower:
            bb_pct = round((close - bb_lower) / (bb_upper - bb_lower) * 100, 1)
            bb_pct = float(bb_pct)

        bias6 = safe_ma_stats.get("bias6")
        bias18 = safe_ma_stats.get("bias18")
        bias50 = safe_ma_stats.get("bias50")
        bias6_min = safe_ma_stats.get(
            "bias6_90d_low") or safe_ma_stats.get("bias6_min")
        bias6_max = safe_ma_stats.get(
            "bias6_90d_high") or safe_ma_stats.get("bias6_max")
        bias18_min = safe_ma_stats.get(
            "bias18_90d_low") or safe_ma_stats.get("bias18_min")
        bias18_max = safe_ma_stats.get(
            "bias18_90d_high") or safe_ma_stats.get("bias18_max")
        bias50_min = safe_ma_stats.get(
            "bias50_90d_low") or safe_ma_stats.get("bias50_min")
        bias50_max = safe_ma_stats.get(
            "bias50_90d_high") or safe_ma_stats.get("bias50_max")

        try:
            signal_res = get_tech_signal(
                close=close,
                chgPct=chgPct,
                amp=amp,
                volume=volume,
                prev_volume=prev_volume,
                prev2_volume=prev2_volume,
                k=k,
                d=d,
                prev_k=prev_k,
                prev_d=prev_d,
                k_trend=k_trend,
                d_trend=d_trend,
                bb_pct=bb_pct,
                bias6=bias6,
                bias18=bias18,
                bias50=bias50,
                bias6_min=bias6_min,
                bias6_max=bias6_max,
                bias18_min=bias18_min,
                bias18_max=bias18_max,
                bias50_min=bias50_min,
                bias50_max=bias50_max,
                ma18=ma18,
                prev_ma18=prev_ma18,
                prev_close=prev_close,
            ) or {"signal": "等待觀察", "reason": "", "signal_text": "等待觀察"}
        except Exception as e:
            print(f"❌ signal error {stock_id}: {e}")
            signal_res = {"signal": "等待觀察",
                          "reason": f"signal error: {e}", "signal_text": "等待觀察"}

        signal = signal_res.get("signal", "等待觀察")
        reason = signal_res.get("reason", "")
        signal_text = signal_res.get("signal_text", "等待觀察")

        sig = 1 if signal == "買進" else -1 if signal == "賣出" else 0

        kd_buy = bool(None not in (k, d, prev_k, prev_d)
                      and (prev_k <= prev_d) and (k > d))
        ma18_break = bool(
            ma18 is not None and prev_ma18 is not None and prev_close <= prev_ma18 and close > ma18
        )

        entry_note = ""
        if "短線過熱" in reason or "不宜追價" in reason:
            entry_note = "不追價"
        elif signal == "買進" and kd_buy and ma18_break and k is not None and k < 35:
            entry_note = "抄底"
        elif signal == "買進" and ma18_break and chgPct >= 3:
            entry_note = "追漲"

        static_fields = _build_static_fields(static_row)

        margin_score = calc_margin_score(
            static_fields.get("gross_margin"),
            static_fields.get("operating_margin"),
            static_fields.get("net_margin"),
        )
        eps_score = calc_eps_score(
            static_fields.get("eps_Y"),
            static_fields.get("eps_ttm"),
        )
        trend_score = calc_trend_score(
            static_fields.get("gross_margin_qoq"),
            static_fields.get("gross_margin_yoy_diff"),
            static_fields.get("net_margin_qoq"),
            static_fields.get("net_margin_yoy_diff"),
        )
        score = round(margin_score * 0.4 + eps_score *
                      0.3 + trend_score * 0.3, 2)

        def to_py(v):
            if isinstance(v, np.bool_):
                return bool(v)
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            if pd.isna(v):
                return None
            return v

        result = {
            "name": name,
            "code": stock_id,
            "price": float(round(close, 2)),
            "price_max": float(round(max_price, 2)),
            "price_min": float(round(min_price, 2)),
            "price_90d_high": price_stats.get("price_90d_high"),
            "price_90d_low": price_stats.get("price_90d_low"),
            "chg": float(round(chg, 2)),
            "chgPct": float(chgPct),
            "amp": float(amp),

            **static_fields,

            "dividend": float(dividend) if dividend is not None else None,
            "yield_value": float(yield_value) if yield_value is not None and not pd.isna(yield_value) else None,
            "k": float(round(k, 1)) if k is not None else None,
            "d": float(round(d, 1)) if d is not None else None,
            "kd_3d_up": kd_trend.get("kd_3d_up"),
            "kd_trend": kd_trend.get("kd_trend"),
            "k_trend": k_trend,
            "d_trend": d_trend,
            "kd_score": float(kd_score),
            "ma18": float(round(ma18, 2)) if ma18 is not None else None,
            "ma18_break": bool(ma18_break),
            "kd_buy": bool(kd_buy),
            "bb_pct": float(bb_pct) if bb_pct is not None else None,
            "bb_upper": float(round(bb_upper, 2)) if bb_upper is not None and pd.notna(bb_upper) else None,
            "bb_lower": float(round(bb_lower, 2)) if bb_lower is not None and pd.notna(bb_lower) else None,
            "bb_3d_up": bb_trend.get("bb_3d_up"),
            "bb_trend": bb_trend.get("bb_trend"),
            "bb_score": bb_trend.get("bb_score"),
            "volume": int(round(volume, 0)) if pd.notna(volume) else None,
            "prev_volume": int(round(prev_volume, 0)) if pd.notna(prev_volume) else None,
            "prev2_volume": int(round(prev2_volume, 0)) if pd.notna(prev2_volume) else None,
            "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
            "volume_add": volume_add if volume_add is not None else None,

            "ma6": safe_ma_stats.get("ma6"),
            "bias6": bias6,
            "bias6_min": bias6_min,
            "bias6_max": bias6_max,
            "bias18": bias18,
            "bias18_min": bias18_min,
            "bias18_max": bias18_max,
            "ma50": safe_ma_stats.get("ma50"),
            "bias50": bias50,
            "bias50_min": bias50_min,
            "bias50_max": bias50_max,

            "sig": int(sig),
            "signal": signal,
            "score": float(score),
            "signal_text": signal_text,
            "reason": reason,
            "entry_note": entry_note,
        }
        return {k: to_py(v) for k, v in result.items()}

    except RuntimeError:
        raise
    except Exception as e:
        print(f"❌ process error {stock_id}: {e}")
        x = base.copy()
        x.update(_build_static_fields(static_row))
        x.update({
            "signal": "資料異常",
            "signal_text": "資料異常",
            "reason": f"process error: {e}",
        })
        return x


def _build_static_fields(static_row):
    return {
        "eps_Y": to_float_or_none(static_row.get("eps_Y")),
        "eps_ttm": to_float_or_none(static_row.get("eps_ttm")),
        "per_Y": to_float_or_none(static_row.get("per_Y")),
        "per_ttm": to_float_or_none(static_row.get("per_ttm")),
        "rev": to_float_or_none(static_row.get("rev")),
        "rev_mom": to_float_or_none(static_row.get("rev_mom")),
        "rev_qoq": to_float_or_none(static_row.get("rev_qoq")),
        "rev_yoy": to_float_or_none(static_row.get("rev_yoy")),

        "gross_margin": to_float_or_none(static_row.get("gross_margin")),
        "gross_margin_qoq": to_float_or_none(static_row.get("gross_margin_qoq")),
        "gross_margin_yoy_diff": to_float_or_none(static_row.get("gross_margin_yoy_diff")),

        "operating_margin": to_float_or_none(static_row.get("operating_margin")),
        "operating_margin_qoq": to_float_or_none(static_row.get("operating_margin_qoq")),
        "operating_margin_yoy_diff": to_float_or_none(static_row.get("operating_margin_yoy_diff")),

        "net_margin": to_float_or_none(static_row.get("net_margin")),
        "net_margin_qoq": to_float_or_none(static_row.get("net_margin_qoq")),
        "net_margin_yoy_diff": to_float_or_none(static_row.get("net_margin_yoy_diff")),

        "per_latest": to_float_or_none(static_row.get("per_latest")),
        "per_90d_high": to_float_or_none(static_row.get("per_90d_high")),
        "per_90d_low": to_float_or_none(static_row.get("per_90d_low")),
        "pbr_latest": to_float_or_none(static_row.get("pbr_latest")),
        "pbr_90d_high": to_float_or_none(static_row.get("pbr_90d_high")),
        "pbr_90d_low": to_float_or_none(static_row.get("pbr_90d_low")),
    }


def get_full_stock_analysis(stock_list, static_map=None):
    results = []
    if static_map is None:
        static_map = load_static_map()

    for i, s in enumerate(stock_list, 1):
        #   print(f"處理中 {i}/{len(stock_list)}: {s}")
        data = process_stock(s, static_map=static_map)
        results.append(data)

        if data.get("signal") in ("無資料", "資料不足", "資料異常"):
            print(f"⚠️ 保留異常資料: {s} -> {data.get('reason')}")

    return results
