import pandas as pd


def add_indicators(df):
    try:
        low_min = df['min'].rolling(9).min()
        high_max = df['max'].rolling(9).max()
        denom = (high_max - low_min).replace(0, pd.NA)
        rsv = (df['close'] - low_min) / denom * 100
        rsv = rsv.ffill()
        df['K'] = rsv.ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()

        df['MA6'] = df['close'].rolling(6).mean()
        df['MA18'] = df['close'].rolling(18).mean()
        df['MA50'] = df['close'].rolling(50).mean()

        # MACD: 用於判斷主升段動能是否翻正、改善或降溫。
        # DIF = EMA12 - EMA26, DEA = DIF 的 9 日 EMA, HIST = DIF - DEA。
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_DIF'] = ema12 - ema26
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = df['MACD_DIF'] - df['MACD_DEA']

        std = df['close'].rolling(18).std()
        df['BB_upper'] = df['MA18'] + 2 * std
        df['BB_lower'] = df['MA18'] - 2 * std

        df['BIAS6'] = (df['close'] - df['MA6']) / df['MA6'] * 100
        df['BIAS18'] = (df['close'] - df['MA18']) / df['MA18'] * 100
        df['BIAS50'] = (df['close'] - df['MA50']) / df['MA50'] * 100

        df['BIAS6_60D_HIGH'] = df['BIAS6'].rolling(60, min_periods=30).max()
        df['BIAS6_60D_LOW'] = df['BIAS6'].rolling(60, min_periods=30).min()
        df['BIAS18_60D_HIGH'] = df['BIAS18'].rolling(60, min_periods=30).max()
        df['BIAS18_60D_LOW'] = df['BIAS18'].rolling(60, min_periods=30).min()
        df['BIAS50_60D_HIGH'] = df['BIAS50'].rolling(60, min_periods=30).max()
        df['BIAS50_60D_LOW'] = df['BIAS50'].rolling(60, min_periods=30).min()

        return df
    except Exception as e:
        print(f'❌ indicator error: {e}')
        return df


def get_kd_trend(df):
    if 'K' not in df.columns or 'D' not in df.columns:
        return {"kd_3d_up": None, "kd_trend": None}
    try:
        last3 = df.tail(3)

        # 資料不足
        if len(last3) < 3:
            return {
                "kd_3d_up": None,
                "kd_trend": None
            }

        k_vals = last3['K'].values
        d_vals = last3['D'].values

        # 避免 NaN
        if pd.isna(k_vals).any() or pd.isna(d_vals).any():
            return {
                "kd_3d_up": None,
                "kd_trend": None
            }

        # === K 三日趨勢 ===
        # Sample === k_vals[0]   # [0]三天前
        # Sample === k_vals[1]   # [1]前一天
        # Sample === k_vals[2]   # [2]最新一天
        k_up = k_vals[2] > k_vals[1] > k_vals[0]
        k_down = k_vals[2] < k_vals[1] < k_vals[0]
        k_up = k_vals[2] > k_vals[1] > k_vals[0]
        k_down = k_vals[2] < k_vals[1] < k_vals[0]

        # === KD 交叉（最重要）===
        cross_up = (k_vals[1] <= d_vals[1]) and (
            k_vals[2] > d_vals[2])     # 黃金交叉
        cross_down = (k_vals[1] >= d_vals[1]) and (
            k_vals[2] < d_vals[2])   # 死亡交叉

        # === 趨勢判斷 ===
        if cross_up:
            trend = "↑"       # 強烈買訊
        elif cross_down:
            trend = "↓"       # 強烈賣訊
        elif k_up:
            trend = "↗"
        elif k_down:
            trend = "↘"
        else:
            trend = "→"

        return {
            "kd_3d_up": k_up if k_up is not None else None,
            "kd_trend": trend,
        }

    except Exception as e:
        print(f"❌ KD trend error: {e}")
        return {
            "kd_3d_up": None,
            "kd_trend": None
        }


def get_MABias(df):
    if len(df) < 60:
        return {
            'ma6': None, 'ma18': None, 'ma50': None,
            'bias6': None, 'bias18': None, 'bias50': None,
            'bias6_min': None, 'bias6_max': None,
            'bias18_min': None, 'bias18_max': None,
            'bias50_min': None, 'bias50_max': None,
        }

    periods = [6, 18, 50]
    stats = {}

    for p in periods:
        ma_series = df['close'].rolling(p).mean()
        ma_value = ma_series.iloc[-1]
        stats[f'ma{p}'] = round(ma_value, 2) if pd.notna(ma_value) else None

        if ma_value == 0 or pd.isna(ma_value):
            stats[f'bias{p}'] = None
            stats[f'bias{p}_min'] = None
            stats[f'bias{p}_max'] = None
            continue

        bias_series = (df['close'] - ma_series) / ma_series * 100
        latest_bias = bias_series.iloc[-1]
        bias_60 = bias_series.iloc[-60:]

        stats[f'bias{p}'] = round(
            latest_bias, 2) if pd.notna(latest_bias) else None
        stats[f'bias{p}_min'] = round(
            bias_60.min(), 2) if bias_60.notna().any() else None
        stats[f'bias{p}_max'] = round(
            bias_60.max(), 2) if bias_60.notna().any() else None

    return stats


def get_bb_trend(df):
    if 'BB_upper' not in df.columns or 'BB_lower' not in df.columns:
        return {"bb_3d_up": None, "bb_trend": None, "bb_score": None}

    last3 = df.tail(3)

    if len(last3) < 3:
        return {"bb_3d_up": None, "bb_trend": None, "bb_score": None}

    def calc_pct(row):
        if pd.notna(row['BB_upper']) and pd.notna(row['BB_lower']) and row['BB_upper'] != row['BB_lower']:
            return (row['close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower']) * 100
        return None

    pcts = last3.apply(calc_pct, axis=1).values

    if pd.isna(pcts).any():
        return {"bb_3d_up": None, "bb_trend": None, "bb_score": None}

    up = pcts[2] > pcts[1] > pcts[0]
    down = pcts[2] < pcts[1] < pcts[0]

    if up:
        trend = "↗"
        score = 1
    elif down:
        trend = "↘"
        score = -1
    else:
        trend = "→"
        score = 0

    return {
        "bb_3d_up": up,
        "bb_trend": trend,
        "bb_score": score
    }


def safe_pos(value, low, high):
    if value is None or low is None or high is None or high == low:
        return None
    return (value - low) / (high - low)



def get_support_resistance_levels(df, lookback_days=120, pivot_window=5, tolerance_pct=1.2):
    """
    Use FinMind TaiwanStockPrice OHLCV data to estimate nearby resistance/support.

    Logic:
    - Use recent OHLCV rows only, default last 120 trading days.
    - Find swing highs as resistance candidates and swing lows as support candidates.
    - Merge nearby prices into clusters by tolerance_pct so repeated tests become one level.
    - Pick the nearest cluster above latest close as resistance, and nearest cluster below latest close as support.

    Returned prices are rounded to 2 decimals and safe for JSON/template rendering.
    """
    empty = {
        "resistance_price": None,
        "support_price": None,
        "resistance_distance_pct": None,
        "support_distance_pct": None,
        "resistance_touch_count": None,
        "support_touch_count": None,
    }

    def _safe_float(value):
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def _round_or_none(value, ndigits=2):
        value = _safe_float(value)
        if value is None:
            return None
        return round(value, ndigits)

    def _cluster_levels(levels, tolerance):
        """Cluster close prices and return weighted-average levels."""
        if not levels:
            return []

        levels = sorted(levels, key=lambda x: x["price"])
        clusters = []

        for item in levels:
            price = item["price"]
            weight = max(float(item.get("weight") or 1), 1.0)
            if not clusters:
                clusters.append({
                    "prices": [price],
                    "weights": [weight],
                    "dates": [item.get("date")],
                })
                continue

            cluster = clusters[-1]
            avg = sum(p * w for p, w in zip(cluster["prices"], cluster["weights"])) / sum(cluster["weights"])
            if avg and abs(price - avg) / avg <= tolerance:
                cluster["prices"].append(price)
                cluster["weights"].append(weight)
                cluster["dates"].append(item.get("date"))
            else:
                clusters.append({
                    "prices": [price],
                    "weights": [weight],
                    "dates": [item.get("date")],
                })

        result = []
        for cluster in clusters:
            weight_sum = sum(cluster["weights"])
            avg_price = sum(p * w for p, w in zip(cluster["prices"], cluster["weights"])) / weight_sum
            result.append({
                "price": avg_price,
                "touch_count": len(cluster["prices"]),
                "weight": weight_sum,
                "last_date": max([d for d in cluster["dates"] if d is not None], default=None),
            })
        return result

    try:
        if df is None or df.empty:
            return empty
        required = {"close", "max", "min"}
        if not required.issubset(df.columns):
            return empty

        data = df.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data.sort_values("date")

        for col in ["close", "max", "min", "volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=["close", "max", "min"])
        if data.empty:
            return empty

        window = data.tail(max(int(lookback_days or 120), 20)).copy()
        if window.empty:
            return empty

        latest_close = _safe_float(window["close"].iloc[-1])
        if latest_close is None or latest_close <= 0:
            return empty

        pivot_window = max(int(pivot_window or 5), 3)
        if pivot_window % 2 == 0:
            pivot_window += 1
        tolerance = max(float(tolerance_pct or 1.2), 0.1) / 100

        high_roll = window["max"].rolling(pivot_window, center=True, min_periods=3).max()
        low_roll = window["min"].rolling(pivot_window, center=True, min_periods=3).min()
        pivot_highs = window[window["max"].eq(high_roll)].copy()
        pivot_lows = window[window["min"].eq(low_roll)].copy()

        resistance_candidates = []
        support_candidates = []

        def _append_candidate(target, row, price_col):
            price = _safe_float(row.get(price_col))
            if price is None or price <= 0:
                return
            volume = _safe_float(row.get("volume")) if "volume" in row.index else None
            date_value = row.get("date") if "date" in row.index else None
            target.append({
                "price": price,
                "weight": volume if volume and volume > 0 else 1,
                "date": date_value,
            })

        for _, row in pivot_highs.iterrows():
            price = _safe_float(row.get("max"))
            if price is not None and price >= latest_close:
                _append_candidate(resistance_candidates, row, "max")

        for _, row in pivot_lows.iterrows():
            price = _safe_float(row.get("min"))
            if price is not None and price <= latest_close:
                _append_candidate(support_candidates, row, "min")

        # Add recent range extremes as fallbacks so the fields still work when few pivots exist.
        high_60 = _safe_float(window.tail(60)["max"].max())
        low_60 = _safe_float(window.tail(60)["min"].min())
        if high_60 is not None and high_60 >= latest_close:
            resistance_candidates.append({"price": high_60, "weight": 1, "date": None})
        if low_60 is not None and low_60 <= latest_close:
            support_candidates.append({"price": low_60, "weight": 1, "date": None})

        resistance_clusters = [c for c in _cluster_levels(resistance_candidates, tolerance) if c["price"] >= latest_close]
        support_clusters = [c for c in _cluster_levels(support_candidates, tolerance) if c["price"] <= latest_close]

        resistance = min(resistance_clusters, key=lambda c: (c["price"] - latest_close, -c["touch_count"]), default=None)
        support = max(support_clusters, key=lambda c: (c["price"], c["touch_count"]), default=None)

        result = empty.copy()
        if resistance:
            rp = resistance["price"]
            result.update({
                "resistance_price": _round_or_none(rp),
                "resistance_distance_pct": _round_or_none((rp - latest_close) / latest_close * 100),
                "resistance_touch_count": int(resistance.get("touch_count") or 0),
            })
        if support:
            sp = support["price"]
            result.update({
                "support_price": _round_or_none(sp),
                "support_distance_pct": _round_or_none((latest_close - sp) / latest_close * 100),
                "support_touch_count": int(support.get("touch_count") or 0),
            })
        return result

    except Exception as e:
        print(f"❌ support/resistance error: {e}")
        return empty
