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


def _normalize_metric_name(value) -> str:
    """Normalize FinMind financial-statement metric labels for exact matching."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    replacements = {
        "（": "(",
        "）": ")",
        "％": "%",
        " ": "",
        "\u3000": "",
        "－": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _metric_name_mask(df: pd.DataFrame, names: list[str]):
    """Exact-match FinMind metric by type/name/origin_name after normalization.

    Do not use contains(). Labels such as 營業收入, 毛利, 毛利率, 淨利 and
    淨利率 overlap; substring matching can select the wrong row and explode
    EPS/margins/QoQ/YoY.
    """
    mask = pd.Series(False, index=df.index)
    clean_names = {_normalize_metric_name(name) for name in names}
    for col in ("type", "name", "origin_name"):
        if col in df.columns:
            s = df[col].map(_normalize_metric_name)
            mask = mask | s.isin(clean_names)
    return mask


def _standardize_financial_df(data) -> pd.DataFrame:
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data or [])
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date")


def _series_by_metric(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    """Return one value per statement date for a strictly matched metric."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    m = df.loc[_metric_name_mask(df, aliases), ["date", "value"]].copy()
    if m.empty:
        return pd.Series(dtype="float64")
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    m = m.dropna(subset=["date", "value"]).sort_values("date")
    if m.empty:
        return pd.Series(dtype="float64")
    return (
        m.drop_duplicates("date", keep="last")
         .set_index("date")["value"]
         .sort_index()
         .astype(float)
    )


def _normalize_percent_series(s: pd.Series, max_abs: float = 200.0) -> pd.Series:
    """Normalize percentage values and remove impossible polluted rows."""
    if s is None or s.empty:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if s.empty:
        return s

    # Some APIs store ratios as decimals, for example 0.2332 means 23.32%.
    median_abs = s.abs().median()
    if pd.notna(median_abs) and median_abs <= 1.5:
        s = s * 100

    s = s.round(2)
    return s[s.between(-max_abs, max_abs)]


def _calc_current_qoq_yoy(s: pd.Series, latest_statement_date=None) -> dict:
    """Calculate current value, QoQ and YoY as percentage-point differences."""
    s = _normalize_percent_series(s)
    if s.empty:
        return {"current": None, "prev": None, "yoy": None, "qoq": None, "yoy_diff": None, "is_prev": False}

    current_date = s.index.max()
    current = float(s.loc[current_date])

    prev = float(s.iloc[-2]) if len(s) >= 2 else None

    # Prefer same year/quarter matching instead of exact date - 1 year.
    current_year = int(current_date.year)
    current_quarter = int(current_date.quarter)
    yoy_candidates = [
        idx for idx in s.index
        if int(idx.year) == current_year - 1 and int(idx.quarter) == current_quarter
    ]
    if yoy_candidates:
        yoy_date = max(yoy_candidates)
        yoy = float(s.loc[yoy_date])
    elif len(s) >= 5:
        yoy = float(s.iloc[-5])
    else:
        yoy = None

    if latest_statement_date is None:
        latest_statement_date = current_date

    return {
        "current": round(current, 2),
        "prev": round(prev, 2) if prev is not None else None,
        "yoy": round(yoy, 2) if yoy is not None else None,
        "qoq": calc_diff(current, prev),
        "yoy_diff": calc_diff(current, yoy),
        "is_prev": bool(current_date < latest_statement_date),
    }


def get_profit_ratio(stock_id):
    """
    Return latest gross/operating/net margins.

    Project rule:
    1. The displayed margins are percentage points, for example 23.32.
    2. QoQ/YoY are percentage-point differences, for example +3.75.
    3. Prefer amount-account calculation from FinMind financial statements:
       Revenue / GrossProfit / OperatingIncome / IncomeAfterTaxes.
    4. Use precomputed ratio rows only as fallback.
    5. Never substring-match metric names.
    """
    try:
        df = _standardize_financial_df(get_profit_ratio_raw(stock_id))
        if df.empty:
            return None

        latest_statement_date = df["date"].max()

        amount_aliases = {
            "Revenue": [
                "Revenue",
                "營業收入",
                "營業收入合計",
                "營業收入淨額",
                "營業收益",
                "收益",
                "收入",
            ],
            "GrossProfit": [
                "GrossProfit",
                "營業毛利",
                "營業毛利(毛損)",
                "營業毛利（毛損）",
            ],
            "OperatingIncome": [
                "OperatingIncome",
                "營業利益",
                "營業利益(損失)",
                "營業利益（損失）",
            ],
            "IncomeAfterTaxes": [
                "IncomeAfterTaxes",
                "本期淨利",
                "本期淨利(淨損)",
                "本期淨利（淨損）",
                "稅後淨利",
                "繼續營業單位稅後淨利(淨損)",
                "繼續營業單位稅後淨利（淨損）",
            ],
        }

        amount_series = {k: _series_by_metric(
            df, v) for k, v in amount_aliases.items()}
        revenue = amount_series["Revenue"]

        def calc_margin_from_amount(numerator_key: str) -> pd.Series:
            numerator = amount_series.get(
                numerator_key, pd.Series(dtype="float64"))
            if revenue.empty or numerator.empty:
                return pd.Series(dtype="float64")

            merged = pd.concat(
                [revenue.rename("revenue"), numerator.rename("numerator")],
                axis=1,
                join="inner",
            ).dropna()
            if merged.empty:
                return pd.Series(dtype="float64")

            merged = merged[merged["revenue"] > 0]
            if merged.empty:
                return pd.Series(dtype="float64")

            s = (merged["numerator"] / merged["revenue"] * 100).round(2)
            return s[s.between(-200.0, 200.0)]

        # Amount-based calculation is the primary path. Units cancel out.
        gross_s = calc_margin_from_amount("GrossProfit")
        op_s = calc_margin_from_amount("OperatingIncome")
        net_s = calc_margin_from_amount("IncomeAfterTaxes")

        # Ratio rows are fallback only, and only for the specific metric that is missing.
        ratio_aliases = {
            "gross": [
                "GrossMargin",
                "gross_margin",
                "毛利率",
                "營業毛利率",
            ],
            "op": [
                "OperatingMargin",
                "operating_margin",
                "營業利益率",
                "營益率",
                "營業利益率(%)",
                "營益率(%)",
            ],
            "net": [
                "NetMargin",
                "net_margin",
                "稅後淨利率",
                "淨利率",
                "稅後淨利率(%)",
                "淨利率(%)",
            ],
        }

        if gross_s.empty:
            gross_s = _series_by_metric(df, ratio_aliases["gross"])
        if op_s.empty:
            op_s = _series_by_metric(df, ratio_aliases["op"])
        if net_s.empty:
            net_s = _series_by_metric(df, ratio_aliases["net"])

        gross = _calc_current_qoq_yoy(gross_s, latest_statement_date)
        op = _calc_current_qoq_yoy(op_s, latest_statement_date)
        net = _calc_current_qoq_yoy(net_s, latest_statement_date)

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

    Project rule:
    - eps_Y column is kept for compatibility, but its displayed value is the
      latest quarter EPS.
    - eps_ttm is the latest four quarters total.
    - PER is calculated only when current_price is provided.
    """
    try:
        df = _standardize_financial_df(get_eps_raw(stock_id))
        if df.empty:
            return (None, None, None, None, False, False)

        eps_aliases = [
            "EPS",
            "BasicEPS",
            "basic_eps",
            "基本每股盈餘",
            "基本每股盈餘(元)",
            "基本每股盈餘（元）",
            "每股盈餘",
        ]
        eps_mask = _metric_name_mask(df, eps_aliases)
        eps_df = df.loc[eps_mask, ["date", "value"]].copy()

        # Avoid diluted EPS if both basic and diluted are present.
        for col in ("type", "name", "origin_name"):
            if col in df.columns and not eps_df.empty:
                labels = df.loc[eps_df.index, col].map(_normalize_metric_name)
                diluted_mask = labels.str.contains(
                    "稀釋", na=False) | labels.str.contains("diluted", na=False)
                if diluted_mask.any() and (~diluted_mask).any():
                    eps_df = eps_df.loc[~diluted_mask]

        if eps_df.empty:
            return (None, None, None, None, False, False)

        eps_df["value"] = pd.to_numeric(eps_df["value"], errors="coerce")
        eps_df = eps_df.dropna(subset=["date", "value"]).sort_values("date")
        if eps_df.empty:
            return (None, None, None, None, False, False)

        eps_df["year"] = eps_df["date"].dt.year.astype(int)
        eps_df["season"] = eps_df["date"].dt.quarter.astype(int)
        eps_df = (
            eps_df.drop_duplicates(["year", "season"], keep="last")
                  .sort_values("date")
                  .reset_index(drop=True)
        )
        if eps_df.empty:
            return (None, None, None, None, False, False)

        latest_eps_date = eps_df["date"].max()
        latest_eps_row = eps_df.iloc[-1]

        eps_latest_quarter = round(float(latest_eps_row["value"]), 2)

        latest4 = eps_df.tail(4)
        if len(latest4) >= 4:
            eps_ttm = round(float(latest4["value"].sum()), 2)
            eps_ttm_is_prev = bool(latest4["date"].max() < latest_eps_date)
        else:
            eps_ttm = eps_latest_quarter
            eps_ttm_is_prev = False

        def calc_per(price, eps):
            try:
                price = float(price)
                eps = float(eps)
            except (TypeError, ValueError):
                return None
            return round(price / eps, 2) if price > 0 and eps > 0 else None

        per_y = calc_per(current_price, eps_latest_quarter)
        per_ttm = calc_per(current_price, eps_ttm)

        return eps_latest_quarter, eps_ttm, per_y, per_ttm, False, eps_ttm_is_prev

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
