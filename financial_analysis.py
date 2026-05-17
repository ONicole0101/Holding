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


def get_profit_ratio(stock_id):
    """
    財務三率：用最新可計算資料；若最新公告期缺科目，回補最近一期有效值。
    回傳 current_is_prev 旗標，讓前端標註「(前期)」。
    """
    try:
        df = get_profit_ratio_raw(stock_id)
        if df is None or df.empty:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')

        pivot = df.pivot_table(
            index='date',
            columns='type',
            values='value',
            aggfunc='last',
        ).sort_index()

        cols = ['Revenue', 'GrossProfit',
                'OperatingIncome', 'IncomeAfterTaxes']
        for c in cols:
            if c not in pivot.columns:
                pivot[c] = pd.NA
        pivot = pivot[cols].apply(pd.to_numeric, errors='coerce')
        if pivot.empty:
            return None

        latest_report_date = pivot.index.max()

        def metric_series(numerator):
            tmp = pivot[[numerator, 'Revenue']].copy()
            tmp = tmp.dropna(subset=[numerator, 'Revenue'])
            tmp = tmp[tmp['Revenue'] > 0]
            if tmp.empty:
                return pd.Series(dtype='float64')
            return (tmp[numerator] / tmp['Revenue'] * 100).round(2)

        def pick_metric(numerator):
            s = metric_series(numerator)
            if s.empty:
                return {
                    'current': None,
                    'prev': None,
                    'yoy': None,
                    'qoq': None,
                    'yoy_diff': None,
                    'is_prev': False,
                }

            current_date = s.index[-1]
            current = float(s.iloc[-1])
            prev = float(s.iloc[-2]) if len(s) >= 2 else None
            yoy = float(s.iloc[-5]) if len(s) >= 5 else None

            return {
                'current': current,
                'prev': prev,
                'yoy': yoy,
                'qoq': calc_diff(current, prev),
                'yoy_diff': calc_diff(current, yoy),
                'is_prev': bool(current_date < latest_report_date),
            }

        gross = pick_metric('GrossProfit')
        op = pick_metric('OperatingIncome')
        net = pick_metric('IncomeAfterTaxes')

        if all(x['current'] is None for x in [gross, op, net]):
            return None

        return {
            'current': {
                'gross': gross['current'],
                'op': op['current'],
                'net': net['current'],
            },
            'prev': {
                'gross': gross['prev'],
                'op': op['prev'],
                'net': net['prev'],
            },
            'yoy': {
                'gross': gross['yoy'],
                'op': op['yoy'],
                'net': net['yoy'],
            },
            'qoq': {
                'gross': gross['qoq'],
                'op': op['qoq'],
                'net': net['qoq'],
            },
            'yoy_diff': {
                'gross': gross['yoy_diff'],
                'op': op['yoy_diff'],
                'net': net['yoy_diff'],
            },
            'current_is_prev': {
                'gross': gross['is_prev'],
                'op': op['is_prev'],
                'net': net['is_prev'],
            },
        }
    except Exception as e:
        print(f'❌ profit error {stock_id}: {e}')
        return None


def extract_metric(res, key):
    if not res:
        return None, None, None
    return (
        res['current'].get(key),
        res['qoq'].get(key),
        res['yoy_diff'].get(key),
    )


def get_eps_analysis(stock_id, current_price=None):
    """
    EPS / EPS TTM：不再綁死今年與去年季別。
    - eps_Y：最近一個四季完整年度 EPS；若不是最近可期待年度，標註前期。
    - eps_ttm：最近四個有效季 EPS 合計；若最新有效季早於最近已結束季，標註前期。
    回傳 6 欄：eps_Y, eps_ttm, per_Y, per_ttm, eps_Y_is_prev, eps_ttm_is_prev
    """
    try:
        data = get_eps_raw(stock_id)
        if not data:
            return (None, None, None, None, False, False)

        df = pd.DataFrame(data)
        if "type" not in df.columns:
            return (None, None, None, None, False, False)

        # FinMind 官方範例 type 為 EPS、origin_name 為「基本每股盈餘」。
        # 這裡同時支援 origin_name，避免部分資料列中文名稱存在但 type 命名差異造成誤判空白。
        type_text = df["type"].astype(str).str.strip().str.upper()
        eps_mask = type_text.eq("EPS")
        if "origin_name" in df.columns:
            origin_text = df["origin_name"].astype(str)
            eps_mask = eps_mask | origin_text.str.contains(
                "每股盈餘|基本每股盈餘|EPS", case=False, regex=True, na=False)

        df = df[eps_mask].copy()
        if df.empty:
            return (None, None, None, None, False, False)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        if df.empty:
            return (None, None, None, None, False, False)

        df["year"] = df["date"].dt.year
        df["season"] = df["date"].dt.quarter
        df = (
            df.sort_values("date")
              .drop_duplicates(["year", "season"], keep="last")
              .reset_index(drop=True)
        )

        today = pd.Timestamp(datetime.now().date())
        last_completed_q = ((today.month - 1) // 3)  # 0~3, 0 代表去年 Q4
        if last_completed_q == 0:
            expected_year = today.year - 1
            expected_season = 4
        else:
            expected_year = today.year
            expected_season = last_completed_q
        expected_q_end = pd.Timestamp(
            expected_year, expected_season * 3, 1) + pd.offsets.MonthEnd(0)

        # 最近完整年度 EPS：只要四季都有值即可，不再只看 this_year - 1。
        eps_last = None
        eps_y_is_prev = False
        full_years = []
        for year, g in df.groupby("year"):
            seasons = set(g["season"].astype(int).tolist())
            if {1, 2, 3, 4}.issubset(seasons):
                full_years.append(int(year))
        if full_years:
            annual_year = max(full_years)
            annual_df = df[df["year"] == annual_year].drop_duplicates(
                "season", keep="last")
            eps_last = round(float(annual_df["value"].sum()), 2)
            # 例如 2026 年應該可用的完整年度通常是 2025；若只回補到 2024，標為前期。
            eps_y_is_prev = bool(annual_year < today.year - 1)

        # 最近四季 TTM：取最近四個有效 EPS 季度。
        eps_ttm = None
        eps_ttm_is_prev = False
        if len(df) >= 4:
            latest4 = df.sort_values("date").tail(4)
            eps_ttm = round(float(latest4["value"].sum()), 2)
            latest_eps_date = latest4["date"].max()
            eps_ttm_is_prev = bool(latest_eps_date < expected_q_end)

        def calc_per(price, eps):
            try:
                price = float(price) if price is not None else None
            except Exception:
                price = None
            return round(price / eps, 2) if price and eps is not None and eps > 0 else None

        per_last = calc_per(current_price, eps_last)
        per_ttm = calc_per(current_price, eps_ttm)

        return eps_last, eps_ttm, per_last, per_ttm, eps_y_is_prev, eps_ttm_is_prev

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
