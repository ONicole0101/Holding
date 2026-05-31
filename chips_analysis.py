import datetime
import pandas as pd
from FinMind.data import DataLoader

# 1. 初始化 FinMind 實體 (若有 Token 可以填入，沒有的話免費版有流量限制)
dl = DataLoader()
# dl.login(token="你的FinMindToken") # 選填

# 2. 設定觀察設定
stock_id = "2330"  # 你想觀察的股票代碼
today = datetime.date.today()
start_date = (today - datetime.timedelta(days=15)
              ).strftime("%Y-%m-%d")  # 抓取過去15天確保有3個交易日

print(f"正在分析 {stock_id} 近期的籌碼變化...")

try:
    # 3. 抓取「券商分點買賣超」數據 (TaiwanStockBrokerBS)
    # 註：此 API 免費版可能每日有調用限制，或需要傳入精確日期
    df = dl.taiwan_stock_broker_bs(
        stock_id=stock_id,
        start_date=start_date
    )

    if df.empty:
        print("未抓取到數據，可能非交易日或超出免費版權限。")
    else:
        # 4. 資料前處理：計算每日的「主力買賣超」與「買賣家數差」
        # FinMind 欄位說明：date(日期), broker(券商), buy(買進張數), sell(賣出張數)
        df['net_buy'] = df['buy'] - df['sell']  # 計算每家券商的淨買超

        daily_report = []

        # 依日期分組計算
        for date, group in df.groupby('date'):
            # 計算買賣家數差：有買的券商家數 - 有賣的券商家數
            active_buyers = group[group['buy'] > 0]['broker'].nunique()
            active_sellers = group[group['sell'] > 0]['broker'].nunique()
            broker_diff = active_buyers - active_sellers

            # 計算主力買賣超（取淨買超前 15 大與淨賣超前 15 大）
            sorted_group = group.sort_values(by='net_buy', ascending=False)
            top_15_buy = sorted_group.head(15)['net_buy'].sum()
            top_15_sell = sorted_group.tail(15)['net_buy'].sum()
            main_force_net = top_15_buy + top_15_sell  # 賣超為負數，相加即為差額

            daily_report.append({
                'Date': date,
                '主力買賣超': int(main_force_net),
                '買賣家數差': int(broker_diff)
            })

        report_df = pd.DataFrame(daily_report).sort_values(
            by='Date', ascending=False)

        # 5. 取出最近的 3 個交易日
        recent_3_days = report_df.head(3)
        print("\n【最近 3 個交易日籌碼數據】")
        print(recent_3_days.to_string(index=False))
        print("-" * 40)

        # 6. 核心邏輯判斷（連續3天主力買、家數差為負）
        is_main_buying = all(recent_3_days['主力買賣超'] > 0)
        is_house_decreasing = all(recent_3_days['買賣家數差'] < 0)

        if is_main_buying and is_house_decreasing:
            print(f"🔥 訊號觸發！股票 {stock_id} 出現『連續3天籌碼高度集中』！")
            print("→ 原因：連續3天主力大買，且券商家數變少（少數大戶收走散戶股票）。")
        elif is_main_buying:
            print(f"⚠️ 提示：連續3天主力買進，但家數差未完全收斂，籌碼可能稍嫌分散。")
        elif all(recent_3_days['主力買賣超'] < 0) and all(recent_3_days['買賣家數差'] > 0):
            print(f"🚨 警報！股票 {stock_id} 出現『連續3天大戶出貨』！")
            print("→ 原因：連續3天主力大賣，且券商家數變多（股票散落到散戶手中）。")
        else:
            print(" 籌碼處於震盪洗盤階段，方向未定，建議繼續觀察。")

except Exception as e:
    print(f"程式執行出錯: {e}")
    print("提示：FinMind 的分點數據 (TaiwanStockBrokerBS) 資料量較大，若是免費版帳號，建議縮短 start_date 的範圍，或確認 API Token 是否設定。")
