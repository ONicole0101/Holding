from technical_indicators import safe_pos


def _num(x):
    """Return x if it is a usable number, otherwise None."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _join_reasons(reasons):
    return ' / '.join([r for r in reasons if r]) if reasons else '等待觀察'


def _calc_price_volume_state(chgPct, amp, volume, prev_volume, prev2_volume, volume_ok=None):
    """
    價量關係：
    - 價漲量增：偏多，底部/上漲途中較有利
    - 價漲量縮：可能反彈或追價力道不足
    - 價跌量增：偏空，頂部/下跌途中較危險
    - 價跌量縮：跌勢趨緩，底部區可觀察是否止穩
    - 價平量增：多空換手，需看位階
    - 價平量縮：盤整
    """
    chgPct = _num(chgPct)
    amp = _num(amp)
    volume = _num(volume)
    prev_volume = _num(prev_volume)
    prev2_volume = _num(prev2_volume)

    price_up = chgPct is not None and chgPct > 0.5
    price_down = chgPct is not None and chgPct < -0.5
    price_flat = chgPct is not None and abs(chgPct) <= 0.5

    volume_2day_up = False
    volume_up = False
    volume_down = False
    volume_shrink = False
    volume_not_bad = False
    volume_spike = False

    if None not in (volume, prev_volume):
        volume_up = volume > prev_volume * 1.05
        volume_down = volume < prev_volume * 0.95
        volume_shrink = volume < prev_volume * 0.85
        volume_not_bad = volume >= prev_volume * 0.9
        volume_spike = volume >= prev_volume * 1.5

    if None not in (volume, prev_volume, prev2_volume):
        volume_2day_up = volume > prev_volume > prev2_volume
    elif volume_ok is not None:
        volume_2day_up = bool(volume_ok)
        volume_up = bool(volume_ok)
        volume_not_bad = bool(volume_ok)

    if price_up and volume_up:
        state = '價漲量增'
    elif price_up and volume_down:
        state = '價漲量縮'
    elif price_down and volume_up:
        state = '價跌量增'
    elif price_down and volume_down:
        state = '價跌量縮'
    elif price_flat and volume_up:
        state = '價平量增'
    elif price_flat and volume_down:
        state = '價平量縮'
    else:
        state = '價量中性'

    return {
        'state': state,
        'price_up': price_up,
        'price_down': price_down,
        'price_flat': price_flat,
        'volume_2day_up': volume_2day_up,
        'volume_up': volume_up,
        'volume_down': volume_down,
        'volume_shrink': volume_shrink,
        'volume_not_bad': volume_not_bad,
        'volume_spike': volume_spike,
    }


def _calc_position_zone(
    close,
    bb_pct,
    bias_low_zone,
    bias_high_zone,
    kd_low,
    kd_high,
    above_ma18,
    below_ma18,
    ma18_break,
    ma18_fall_break,
    kd_turn_strong,
    kd_turn_weak,
    k_trend_up,
    k_trend_down,
    ma18_up=False,
    ma50_up=False,
    above_ma50=False,
    below_ma50=False,
):
    """
    股價位階粗分：
    - 底部區域：布林低檔 / 乖離低檔 / KD低檔
    - 上漲途中：站上月線且動能偏強，或剛突破月線
    - 頂部區域：布林高檔 / 乖離高檔 / KD高檔
    - 下跌途中：跌破月線或在月線下且動能偏弱

    聯發科 2454 檢討修正：
    - 剛站回月線、KD轉強、價量轉強時，優先視為起漲/上漲途中。
    - KD高檔不等於頂部；必須搭配乖離/布林過熱或趨勢跌破才視為頂部。
    """
    bb_pct = _num(bb_pct)

    bb_low = bb_pct is not None and bb_pct < 20
    bb_mid_low = bb_pct is not None and 20 <= bb_pct <= 50
    bb_mid = bb_pct is not None and 35 <= bb_pct <= 80
    bb_high = bb_pct is not None and bb_pct > 80
    bb_overheat = bb_pct is not None and bb_pct > 95

    early_uptrend = ma18_break and (kd_turn_strong or k_trend_up)
    trend_supported = above_ma18 and (ma18_up or ma50_up or above_ma50)

    # 下跌途中優先於底部，避免「跌破後還沒止穩」被誤判為低接。
    if ma18_fall_break or (below_ma18 and (kd_turn_weak or k_trend_down)):
        zone = '下跌途中'
    # 起漲與上漲途中優先於 KD 高檔，避免聯發科 4 月剛啟動時被太早賣出。
    elif early_uptrend or (above_ma18 and (k_trend_up or kd_turn_strong) and not bb_overheat):
        zone = '上漲途中'
    elif trend_supported and not (bb_overheat and bias_high_zone):
        zone = '上漲途中'
    elif bb_overheat or (bias_high_zone and kd_high):
        zone = '頂部區域'
    elif bb_low or bias_low_zone or kd_low:
        zone = '底部區域'
    elif above_ma18:
        zone = '上漲途中'
    elif below_ma18 or below_ma50:
        zone = '下跌途中'
    else:
        zone = '盤整區域'

    return {
        'zone': zone,
        'bb_low': bb_low,
        'bb_mid_low': bb_mid_low,
        'bb_mid': bb_mid,
        'bb_high': bb_high,
        'bb_overheat': bb_overheat,
    }


def get_tech_signal(
    close,
    chgPct,
    amp,
    volume_ok=None,
    volume=None,
    prev_volume=None,
    prev2_volume=None,
    k=None,
    d=None,
    prev_k=None,
    prev_d=None,
    bb_pct=None,
    bias6=None,
    bias18=None,
    bias50=None,
    bias6_min=None,
    bias6_max=None,
    bias18_min=None,
    bias18_max=None,
    bias50_min=None,
    bias50_max=None,
    ma18=None,
    prev_ma18=None,
    prev_close=None,
    k_trend=None,
    d_trend=None,
    # 以下為向下相容的選填欄位；呼叫端沒有給也不影響原本功能。
    ma6=None,
    prev_ma6=None,
    ma50=None,
    prev_ma50=None,
    macd_hist=None,
    prev_macd_hist=None,
):
    """
    技術訊號主邏輯。

    2026 聯發科/廣達檢討改版重點：
    1. 先判斷「位階」：底部區域 / 上漲途中 / 頂部區域 / 下跌途中 / 盤整區域
    2. 再判斷「價量關係」：價漲量增 / 價漲量縮 / 價跌量增 / 價跌量縮 / 價平量增 / 價平量縮
    3. 最後才用 KD、月線、布林、乖離、MACD 輔助確認買賣訊號
    4. 避免「剛進入上漲途中」卻因 KD 高檔或短線拉回而過早賣出
    5. 下跌途中低接只給觀察，不在未止跌前直接買進或重壓
    """
    reasons = []

    close = _num(close)
    chgPct = _num(chgPct)
    amp = _num(amp)
    k = _num(k)
    d = _num(d)
    prev_k = _num(prev_k)
    prev_d = _num(prev_d)
    ma6 = _num(ma6)
    prev_ma6 = _num(prev_ma6)
    ma18 = _num(ma18)
    prev_ma18 = _num(prev_ma18)
    ma50 = _num(ma50)
    prev_ma50 = _num(prev_ma50)
    prev_close = _num(prev_close)
    macd_hist = _num(macd_hist)
    prev_macd_hist = _num(prev_macd_hist)

    if close is None:
        return {
            'signal': '等待觀察',
            'reason': '缺少收盤價資料',
            'signal_text': '資料不足',
        }

    # === KD 判斷 ===
    if None in (k, d, prev_k, prev_d):
        kd_gold_cross = False
        kd_dead_cross = False
    else:
        kd_gold_cross = prev_k <= prev_d and k > d
        kd_dead_cross = prev_k >= prev_d and k < d

    kd_low = (k is not None and d is not None and k < 30 and d < 30)
    kd_high = (k is not None and d is not None and k > 80 and d > 80)

    kd_turn_strong = False
    kd_turn_weak = False
    if prev_k is not None and k is not None:
        kd_turn_strong = k > prev_k
        kd_turn_weak = k < prev_k

    k_trend_up = k_trend in ('↑', '↗', 'up')
    k_trend_down = k_trend in ('↓', '↘', 'down')

    if kd_gold_cross:
        reasons.append('KD黃金交叉')
    if kd_dead_cross:
        reasons.append('KD死亡交叉')
    if kd_low:
        reasons.append('KD位於低檔區')
    if kd_high:
        reasons.append('KD位於高檔區')
    if k_trend_up and not kd_gold_cross:
        reasons.append('KD動能走強')
    if k_trend_down and not kd_dead_cross:
        reasons.append('KD動能轉弱')

    # === 股價 / 趨勢 ===
    price_up_raw = chgPct is not None and chgPct > 0
    price_down_raw = chgPct is not None and chgPct < 0
    price_flat_raw = chgPct is not None and abs(chgPct) < 0.5

    above_ma6 = ma6 is not None and close > ma6
    below_ma6 = ma6 is not None and close < ma6
    above_ma18 = ma18 is not None and close > ma18
    below_ma18 = ma18 is not None and close < ma18
    above_ma50 = ma50 is not None and close > ma50
    below_ma50 = ma50 is not None and close < ma50

    ma6_up = ma6 is not None and prev_ma6 is not None and ma6 > prev_ma6
    ma18_up = ma18 is not None and prev_ma18 is not None and ma18 >= prev_ma18
    ma50_up = ma50 is not None and prev_ma50 is not None and ma50 >= prev_ma50

    ma18_break = (
        ma18 is not None and prev_ma18 is not None and prev_close is not None
        and prev_close <= prev_ma18 and close > ma18
    )

    ma18_fall_break = (
        ma18 is not None and prev_ma18 is not None and prev_close is not None
        and prev_close >= prev_ma18 and close < ma18
    )

    ma6_fall_break = (
        ma6 is not None and prev_ma6 is not None and prev_close is not None
        and prev_close >= prev_ma6 and close < ma6
    )

    if price_up_raw:
        reasons.append('股價上漲')
    elif price_down_raw:
        reasons.append('股價下跌')
    if price_flat_raw:
        reasons.append('股價接近橫盤整理')

    if above_ma18:
        reasons.append('股價位於月線之上')
    elif below_ma18:
        reasons.append('股價位於月線之下')

    if ma18_break:
        reasons.append('股價突破月線')
    if ma18_fall_break:
        reasons.append('股價跌破月線')
    if ma18_up:
        reasons.append('月線走平向上')
    if above_ma50:
        reasons.append('股價位於季線之上')

    # === MACD 輔助 ===
    macd_turn_positive = False
    macd_turn_negative = False
    macd_improving = False
    macd_weakening = False
    if macd_hist is not None and prev_macd_hist is not None:
        macd_turn_positive = prev_macd_hist <= 0 < macd_hist
        macd_turn_negative = prev_macd_hist >= 0 > macd_hist
        macd_improving = macd_hist > prev_macd_hist
        macd_weakening = macd_hist < prev_macd_hist
        if macd_turn_positive:
            reasons.append('MACD柱狀體翻正')
        elif macd_turn_negative:
            reasons.append('MACD柱狀體翻黑')
        elif macd_improving:
            reasons.append('MACD動能改善')
        elif macd_weakening:
            reasons.append('MACD動能降溫')

    # === Bias 輔助 ===
    bias6_pos = safe_pos(bias6, bias6_min, bias6_max)
    bias18_pos = safe_pos(bias18, bias18_min, bias18_max)
    bias50_pos = safe_pos(bias50, bias50_min, bias50_max)

    low_count = 0
    high_count = 0
    for pos in (bias6_pos, bias18_pos, bias50_pos):
        if pos is None:
            continue
        if pos < 0.2:
            low_count += 1
        elif pos > 0.8:
            high_count += 1

    bias_low_zone = low_count >= 2
    bias_high_zone = high_count >= 2

    if bias_low_zone:
        reasons.append('乖離處於相對低檔')
    if bias_high_zone:
        reasons.append('乖離處於相對高檔')

    # === 價量關係 ===
    pv = _calc_price_volume_state(
        chgPct=chgPct,
        amp=amp,
        volume=volume,
        prev_volume=prev_volume,
        prev2_volume=prev2_volume,
        volume_ok=volume_ok,
    )

    volume_2day_up = pv['volume_2day_up']
    volume_up = pv['volume_up']
    volume_down = pv['volume_down']
    volume_not_bad = pv['volume_not_bad']
    volume_shrink = pv['volume_shrink']
    volume_spike = pv['volume_spike']

    price_up = pv['price_up']
    price_down = pv['price_down']
    price_volume_state = pv['state']
    reasons.append(price_volume_state)

    if volume_2day_up:
        reasons.append('成交量連續兩天放大')
    elif volume_up:
        reasons.append('成交量放大')
    elif volume_down:
        reasons.append('成交量縮小')
    elif volume_not_bad:
        reasons.append('成交量維持')
    if volume_spike:
        reasons.append('爆量換手')
    if volume_shrink:
        reasons.append('明顯量縮')

    # === 位階判斷 ===
    zone_info = _calc_position_zone(
        close=close,
        bb_pct=bb_pct,
        bias_low_zone=bias_low_zone,
        bias_high_zone=bias_high_zone,
        kd_low=kd_low,
        kd_high=kd_high,
        above_ma18=above_ma18,
        below_ma18=below_ma18,
        ma18_break=ma18_break,
        ma18_fall_break=ma18_fall_break,
        kd_turn_strong=kd_turn_strong,
        kd_turn_weak=kd_turn_weak,
        k_trend_up=k_trend_up,
        k_trend_down=k_trend_down,
        ma18_up=ma18_up,
        ma50_up=ma50_up,
        above_ma50=above_ma50,
        below_ma50=below_ma50,
    )

    position_zone = zone_info['zone']
    bb_low = zone_info['bb_low']
    bb_mid = zone_info['bb_mid']
    bb_high = zone_info['bb_high']
    bb_overheat = zone_info['bb_overheat']

    reasons.append(position_zone)

    if bb_low:
        reasons.append('接近布林下緣')
    elif bb_high:
        reasons.append('位於布林高檔區')
    elif bb_mid:
        reasons.append('布林位於中性偏強區')

    if bb_overheat:
        reasons.append('接近布林上緣過熱')

    # === 強弱輔助條件 ===
    kd_strong = kd_gold_cross or kd_turn_strong or k_trend_up
    kd_weak = kd_dead_cross or kd_turn_weak or k_trend_down
    trend_supported = above_ma18 and (ma18_up or ma50_up or above_ma50 or ma18_break)
    early_uptrend = ma18_break and kd_strong and price_volume_state in ('價漲量增', '價平量增', '價量中性')
    main_uptrend = position_zone == '上漲途中' and trend_supported and not ma18_fall_break
    overheat_confirmed = bb_overheat or (bb_high and bias_high_zone) or (bias_high_zone and kd_high)
    trend_break_confirmed = ma18_fall_break or (below_ma18 and kd_weak)

    # ============================================================
    # 規則判斷：位階 × 價量 × 技術確認
    # ============================================================

    # 1) 起漲保護：聯發科 2454 類型，剛站回月線 + 價量/KD轉強，不因短線高檔或剛獲利而賣。
    if early_uptrend:
        return {
            'signal': '觀察再買進',
            'reason': '剛突破月線且價量/KD轉強，屬起漲或轉強初期，持股不宜過早賣出',
            'signal_text': _join_reasons(reasons),
        }

    # 2) 明確賣出：高檔過熱後價跌量增，或已跌破月線，才直接賣出。
    if (
        price_volume_state == '價跌量增'
        and (overheat_confirmed or position_zone == '下跌途中' or trend_break_confirmed)
        and (kd_weak or trend_break_confirmed or macd_turn_negative)
    ):
        return {
            'signal': '賣出',
            'reason': '高檔過熱或下跌途中出現價跌量增，且動能/均線轉弱',
            'signal_text': _join_reasons(reasons),
        }

    # 3) 明確賣出：連續量增下跌且跌破月線。
    if (
        volume_2day_up
        and price_down
        and ma18_fall_break
        and (kd_weak or macd_turn_negative)
    ):
        return {
            'signal': '賣出',
            'reason': '連續放量下跌並跌破月線，轉弱訊號明確',
            'signal_text': _join_reasons(reasons),
        }

    # 4) 高檔轉弱：只要尚未跌破月線，不直接出清，以分批留意賣點為主。
    if (
        position_zone == '頂部區域'
        and overheat_confirmed
        and (kd_weak or macd_weakening or ma6_fall_break)
        and not ma18_fall_break
    ):
        return {
            'signal': '觀察再賣出',
            'reason': '高檔過熱且動能降溫，但尚未跌破月線，宜分批停利而非一次出清',
            'signal_text': _join_reasons(reasons),
        }

    # 5) 主升段保護：月線上方、趨勢仍受支撐，KD高檔或短線降溫不視為賣出。
    if (
        main_uptrend
        and price_volume_state in ('價漲量增', '價漲量縮', '價平量增', '價量中性')
        and not overheat_confirmed
    ):
        return {
            'signal': '觀察再買進' if price_volume_state in ('價漲量增', '價平量增') and kd_strong else '等待觀察',
            'reason': '股價仍在上漲途中且月線趨勢未破，持股以續抱觀察為主，不因KD高檔過早賣出',
            'signal_text': _join_reasons(reasons),
        }

    # 6) 上漲途中放量下跌：提高警戒，但未跌破月線前不直接賣出。
    if (
        main_uptrend
        and price_volume_state == '價跌量增'
        and (kd_weak or macd_weakening)
        and not ma18_fall_break
    ):
        return {
            'signal': '觀察再賣出',
            'reason': '上漲途中出現價跌量增與動能轉弱，若跌破月線應降低持股',
            'signal_text': _join_reasons(reasons),
        }

    # 7) 下跌途中反彈：不急著買，除非重新站回月線且價量/KD同步轉強。
    if (
        position_zone == '下跌途中'
        and not early_uptrend
        and price_volume_state in ('價漲量縮', '價跌量縮', '價平量縮', '價量中性')
    ):
        return {
            'signal': '等待觀察',
            'reason': '仍在下跌途中，反彈或量縮尚不足以確認轉強',
            'signal_text': _join_reasons(reasons),
        }

    # 8) 底部轉強：底部區 + 價漲量增 + KD/MACD改善；先觀察再買進，避免一次重壓。
    if (
        position_zone == '底部區域'
        and price_volume_state == '價漲量增'
        and (kd_strong or macd_improving)
        and not ma18_fall_break
    ):
        return {
            'signal': '觀察再買進',
            'reason': '底部區域出現價漲量增與動能改善，可觀察低檔轉強，但宜分批不宜重壓',
            'signal_text': _join_reasons(reasons),
        }

    # 9) 底部止跌：底部區 + 價跌量縮 / 價平量縮，只能當止跌觀察。
    if (
        position_zone == '底部區域'
        and price_volume_state in ('價跌量縮', '價平量縮')
        and (kd_turn_strong or k_trend_up or kd_low or volume_shrink)
    ):
        return {
            'signal': '等待觀察',
            'reason': '底部區域跌勢趨緩，但尚未出現明確價漲量增，先觀察止穩',
            'signal_text': _join_reasons(reasons),
        }

    # 10) 明確買進：突破或站上月線，價漲量增，KD/MACD轉強，且未明顯過熱。
    if (
        price_volume_state == '價漲量增'
        and (kd_strong or macd_turn_positive or macd_improving)
        and (above_ma18 or ma18_break)
        and not bb_overheat
        and not bias_high_zone
    ):
        return {
            'signal': '買進',
            'reason': '價漲量增，動能轉強，股價站上月線，技術面偏多',
            'signal_text': _join_reasons(reasons),
        }

    # 11) 上漲途中但價漲量縮：不追高，但也不急賣。
    if (
        position_zone == '上漲途中'
        and above_ma18
        and price_volume_state == '價漲量縮'
        and not ma18_fall_break
    ):
        return {
            'signal': '等待觀察',
            'reason': '上漲途中出現價漲量縮，持股可觀察但不宜追高',
            'signal_text': _join_reasons(reasons),
        }

    # 12) 上漲途中轉弱：月線上方先觀察，不因 KD 高檔或短線轉弱過早賣出。
    if (
        position_zone == '上漲途中'
        and above_ma18
        and kd_weak
        and price_volume_state in ('價漲量縮', '價平量增', '價量中性')
        and not ma18_fall_break
    ):
        return {
            'signal': '等待觀察',
            'reason': '上漲途中動能轉弱但尚未跌破月線，先觀察不急賣',
            'signal_text': _join_reasons(reasons),
        }

    # 13) 盤整區：價平量縮或訊號混雜。
    if (
        position_zone == '盤整區域'
        or price_volume_state in ('價平量縮', '價量中性')
    ):
        return {
            'signal': '等待觀察',
            'reason': '位階與價量尚未形成明確方向，等待突破或跌破確認',
            'signal_text': _join_reasons(reasons),
        }

    # 14) 保守預設。
    return {
        'signal': '等待觀察',
        'reason': '價格、量能、KD與布林尚未形成明確方向',
        'signal_text': _join_reasons(reasons),
    }
