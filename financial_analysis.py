from datetime import datetime

import pandas as pd

from data_sources import (
    get_dividend_raw,
    get_eps_raw,
    get_per_raw,
    get_profit_ratio as get_profit_ratio_raw,
    get_revenue_raw,
)


def safe_margin(num, denom):
    if num is None or denom is None or denom <= 0:
        return None
    return round(num / denom * 100, 2)


def calc_diff(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 2)


def fmt(v):
    return '-' if v is None else v


def build_output(result):
    cur = result['current']
    prev = result['prev']
    yoy = result['yoy']
    qoq = result['qoq']
    yoy_diff = result['yoy_diff']

    return {
        'gross_margin': cur['gross'],
        'gross_margin_prev': prev['gross'],
        'gross_margin_yoy': yoy['gross'],
        'gross_margin_qoq': qoq['gross'],
        'gross_margin_yoy_diff': yoy_diff['gross'],
        'gross_margin_combined': f"{fmt(cur['gross'])} / {fmt(prev['gross'])} / {fmt(yoy['gross'])}",
        'operating_margin': cur['op'],
        'operating_margin_prev': prev['op'],
        'operating_margin_yoy': yoy['op'],
        'operating_margin_qoq': qoq['op'],
        'operating_margin_yoy_diff': yoy_diff['op'],
        'operating_margin_combined': f"{fmt(cur['op'])} / {fmt(prev['op'])} / {fmt(yoy['op'])}",
        'net_margin': cur['net'],
        'net_margin_prev': prev['net'],
        'net_margin_yoy': yoy['net'],
        'net_margin_qoq': qoq['net'],
        'net_margin_yoy_diff': yoy_diff['net'],
        'net_margin_combined': f"{fmt(cur['net'])} / {fmt(prev['net'])} / {fmt(yoy['net'])}",
    }



def _metric_name_mask(df: pd.DataFrame, names: list[str]):
    """Match FinMind metric by type/name/origin_name, tolerant of Chinese labels."""
    mask = pd.Series(False, index=df.index)
    for col in ("type", "name", "origin_name"):
        if col in df.columns:
            s = df[col].astype(str)
            for name in names:
                mask = mask | s.str.fullmatch(name, case=False, na=False) | s.str.contains(name, case=False, na=False, regex=False)
    return mask


def _standardize_financial_df(data) -> pd.DataFrame:
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data or [])
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date")


def get_profit_ratio(stock_id):
    """
    Return latest available gross/operating/net margins.

    Important: do NOT drop the whole quarter just because one metric is missing.
    Each margin finds its own latest valid Revenue + numerator pair. This prevents
    FinMind having data for one ratio while another incomplete field makes all
    margins blank.
    """
    try:
        df = _standardize_financial_df(get_profit_ratio_raw(stock_id))
        if df.empty:
            return None

        metric_aliases = {
            "Revenue": ["Revenue", "營業收入", "營業收入合計", "收益", "收入"],
            "GrossProfit": ["GrossProfit", "營業毛利", "營業毛利（毛損）", "毛利"],
            "OperatingIncome": ["OperatingIncome", "營業利益", "營業利益（損失）", "營益", "營業淨利"],
            "IncomeAfterTaxes": ["IncomeAfterTaxes", "本期淨利", "稅後淨利", "本期淨利（淨損）", "淨利"],
        }

        parts = []
        for std_name, aliases in metric_aliases.items():
            m = df.loc[_metric_name_mask(df, aliases), ["date", "value"]].copy()
            if m.empty:
                continue
            m["metric"] = std_name
            parts.append(m)

        if not parts:
            return None

        long_df = pd.concat(parts, ignore_index=True)
        pivot = long_df.pivot_table(index="date", columns="metric", values="value", aggfunc="last").sort_index()
        if "Revenue" not in pivot.columns:
            return None

        latest_statement_date = pivot.index.max()

        def margin_series(numerator: str):
            if numerator not in pivot.columns:
                return pd.DataFrame(columns=["margin"])
            tmp = pivot[["Revenue", numerator]].copy()
            tmp["Revenue"] = pd.to_numeric(tmp["Revenue"], errors="coerce")
            tmp[numerator] = pd.to_numeric(tmp[numerator], errors="coerce")
            tmp = tmp.dropna(subset=["Revenue", numerator])
            tmp = tmp[tmp["Revenue"] > 0]
            if tmp.empty:
                return pd.DataFrame(columns=["margin"])
            tmp["margin"] = (tmp[numerator] / tmp["Revenue"] * 100).round(2)
            return tmp[["margin"]]

        def calc_one(numerator: str):
            ms = margin_series(numerator)
            if ms.empty:
                return {"current": None, "prev": None, "yoy": None, "qoq": None, "yoy_diff": None, "is_prev": False}

            current_date = ms.index.max()
            current = float(ms.loc[current_date, "margin"])

            prev = None
            if len(ms) >= 2:
                prev = float(ms.iloc[-2]["margin"])

            same_q_last_year = current_date - pd.DateOffset(years=1)
            yoy = None
            if same_q_last_year in ms.index:
                yoy = float(ms.loc[same_q_last_year, "margin"])
            elif len(ms) >= 5:
                yoy = float(ms.iloc[-5]["margin"])

            return {
                "current": round(current, 2),
                "prev": round(prev, 2) if prev is not None else None,
                "yoy": round(yoy, 2) if yoy is not None else None,
                "qoq": calc_diff(current, prev),
                "yoy_diff": calc_diff(current, yoy),
                "is_prev": bool(current_date < latest_statement_date),
            }

        gross = calc_one("GrossProfit")
        op = calc_one("OperatingIncome")
        net = calc_one("IncomeAfterTaxes")

        return {
            "current": {"gross": gross["current"], "op": op["current"], "net": net["current"]},
            "prev": {"gross": gross["prev"], "op": op["prev"], "net": net["prev"]},
            "yoy": {"gross": gross["yoy"], "op": op["yoy"], "net": net["yoy"]},
            "qoq": {"gross": gross["qoq"], "op": op["qoq"], "net": net["qoq"]},
            "yoy_diff": {"gross": gross["yoy_diff"], "op": op["yoy_diff"], "net": net["yoy_diff"]},
            "is_prev": {"gross": gross["is_prev"], "op": op["is_prev"], "net": net["is_prev"]},
        }
    except Exception as e:
        print(f'❌ profit error {stock_id}: {e}')
        return None


def extract_metric(res, key):
    if not res:
        return None, None, None
    return (
        res.get('current', {}).get(key),
        res.get('qoq', {}).get(key),
        res.get('yoy_diff', {}).get(key),
    )


def extract_metric_is_prev(res, key):
    if not res:
        return False
    return bool(res.get('is_prev', {}).get(key, False))


def get_eps_analysis(stock_id, current_price=None):
    """
    EPS from FinMind TaiwanStockFinancialStatements.

    Uses latest valid data instead of hard-coding current calendar year/quarter.
    Returns: eps_Y, eps_ttm, per_Y, per_ttm, eps_Y_is_prev, eps_ttm_is_prev
    """
    try:
        df = _standardize_financial_df(get_eps_raw(stock_id))
        if df.empty:
            return (None, None, None, None, False, False)

        eps_mask = _metric_name_mask(df, ["EPS", "基本每股盈餘", "每股盈餘"])
        eps_df = df.loc[eps_mask, ["date", "value"]].copy()
        if eps_df.empty:
            return (None, None, None, None, False, False)

        eps_df["year"] = eps_df["date"].dt.year.astype(int)
        eps_df["season"] = eps_df["date"].dt.quarter.astype(int)
        eps_df = (
            eps_df.sort_values("date")
                  .drop_duplicates(["year", "season"], keep="last")
                  .dropna(subset=["value"])
                  .reset_index(drop=True)
        )
        if eps_df.empty:
            return (None, None, None, None, False, False)

        latest_eps_date = eps_df["date"].max()

        annual = (
            eps_df.groupby("year")
                  .agg(eps_sum=("value", "sum"), q_count=("season", "nunique"), last_date=("date", "max"))
                  .reset_index()
                  .sort_values("year")
        )
        annual_full = annual[annual["q_count"] >= 4]
        if not annual_full.empty:
            yrow = annual_full.iloc[-1]
        else:
            # Fallback for newly listed or incomplete years: use latest available year sum.
            yrow = annual.iloc[-1]
        eps_y = round(float(yrow["eps_sum"]), 2) if pd.notna(yrow["eps_sum"]) else None
        eps_y_is_prev = bool(pd.to_datetime(yrow["last_date"]) < latest_eps_date)

        latest4 = eps_df.sort_values("date").tail(4)
        if len(latest4) >= 4:
            eps_ttm = round(float(latest4["value"].sum()), 2)
            eps_ttm_is_prev = bool(latest4["date"].max() < latest_eps_date)
        else:
            eps_ttm = eps_y
            eps_ttm_is_prev = eps_y_is_prev

        def calc_per(price, eps):
            try:
                price = float(price)
                eps = float(eps)
            except (TypeError, ValueError):
                return None
            return round(price / eps, 2) if price > 0 and eps > 0 else None

        per_y = calc_per(current_price, eps_y)
        per_ttm = calc_per(current_price, eps_ttm)

        return eps_y, eps_ttm, per_y, per_ttm, eps_y_is_prev, eps_ttm_is_prev

    except Exception as e:
        print(f"❌ EPS error {stock_id}: {e}")
        return (None, None, None, None, False, False)

def get_dividend_yield(stock_id, current_price=None):
    try:
        data = get_dividend_raw(stock_id)
        if not data:
            return {'dividend': None, 'yield': None}

        df = pd.DataFrame(data)
        cash_cols = ['CashEarningsDistribution', 'CashStatutorySurplus']
        exist_cols = [c for c in cash_cols if c in df.columns]
        if not exist_cols:
            return {'dividend': None, 'yield': None}

        df[exist_cols] = df[exist_cols].apply(pd.to_numeric, errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        df_group = (
            df.groupby('year')[exist_cols]
            .sum()
            .sum(axis=1)
            .reset_index(name='cash_dividend')
            .sort_values('year', ascending=False)
        )

        dividend = None
        for val in df_group['cash_dividend']:
            if val and val > 0:
                dividend = round(val, 2)
                break

        yield_pct = None
        per_data = get_per_raw(stock_id)
        if per_data:
            df2 = pd.DataFrame(per_data)
            df2['date'] = pd.to_datetime(df2['date'])
            latest = df2.sort_values('date').iloc[-1]
            yield_pct = latest.get('dividend_yield')
            if yield_pct is not None:
                yield_pct = round(float(yield_pct), 2)

        if yield_pct is None and dividend and current_price and current_price > 0:
            yield_pct = round(dividend / current_price * 100, 2)

        return {'dividend': dividend, 'yield': yield_pct}
    except Exception as e:
        print(f'❌ 股利/殖利率錯誤 {stock_id}: {e}')
        return {'dividend': None, 'yield': None}


def calc_margin_score(gross, op, net):
    score = 0
    if gross is not None:
        score += gross * 0.4
    if op is not None:
        score += op * 0.3
    if net is not None:
        score += net * 0.3
    return round(score, 2)


def calc_eps_score(eps_last, eps_ttm):
    if eps_last is None or eps_ttm is None or eps_last <= 0:
        return 0
    growth = (eps_ttm - eps_last) / eps_last * 100
    return round(growth, 2)


def calc_trend_score(qoq_g, yoy_g, qoq_n, yoy_n):
    vals = [qoq_g, yoy_g, qoq_n, yoy_n]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0
    return round(sum(vals) / len(vals), 2)
