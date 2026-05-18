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
        return {"kd_3d_up": None, "kd_trend": None, "kd_score": None}
    try:
        last3 = df.tail(3)

        # 資料不足
        if len(last3) < 3:
            return {
                "kd_3d_up": None,
                "kd_trend": None,
                "kd_score": None
            }

        k_vals = last3['K'].values
        d_vals = last3['D'].values

        # 避免 NaN
        if pd.isna(k_vals).any() or pd.isna(d_vals).any():
            return {
                "kd_3d_up": None,
                "kd_trend": None,
                "kd_score": None
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
            score = 2
        elif cross_down:
            trend = "↓"       # 強烈賣訊
            score = -2
        elif k_up:
            trend = "↗"
            score = 1
        elif k_down:
            trend = "↘"
            score = -1
        else:
            trend = "→"
            score = 0

        return {
            "kd_3d_up": k_up if k_up is not None else None,
            "kd_trend": trend,
            "kd_score": score,
        }

    except Exception as e:
        print(f"❌ KD trend error: {e}")
        return {
            "kd_3d_up": None,
            "kd_trend": None,
            "kd_score": None
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
