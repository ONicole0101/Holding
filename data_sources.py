import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from FinMind.data import DataLoader
from loguru import logger

API_TOKEN = os.getenv('FINMIND_TOKEN')
API_URL = 'https://api.finmindtrade.com/api/v4/data'
api = DataLoader()

# 停用所有來自 FinMind 的 Log 訊息
logger.remove()
logging.getLogger('FinMind').setLevel(logging.WARNING)


def get_stock_data(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': str(stock_id),
            'start_date': '2023-01-01',
            'token': API_TOKEN,
        }
        res = requests.get(API_URL, params=params, timeout=300)
        data = res.json()

        if res.status_code == 402:
            raise RuntimeError(
                f"FinMind quota exceeded for {stock_id}: {data.get('msg')}")

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
            'token': API_TOKEN,
        }

        res = requests.get(API_URL, params=params, timeout=300)

        if res.status_code != 200:
            return []

        data = res.json().get('data', [])
        return data

    except Exception as e:
        print(f'❌ revenue source error {stock_id}: {e}')
        return []


def get_profit_ratio(stock_id):
    try:
        df = api.taiwan_stock_financial_statement(
            stock_id=stock_id,
            start_date='2022-01-01',
        )
        return df
    except Exception as e:
        print(f'❌ profit source error {stock_id}: {e}')
        return pd.DataFrame()


def get_eps_raw(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockFinancialStatements',
            'data_id': stock_id,
            'start_date': '2020-01-01',
            'token': API_TOKEN,
        }
        return requests.get(API_URL, params=params, timeout=300).json().get('data', [])
    except Exception as e:
        print(f'❌ EPS source error {stock_id}: {e}')
        return []


def get_dividend_raw(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockDividend',
            'data_id': stock_id,
            'start_date': '2020-01-01',
            'token': API_TOKEN,
        }
        res = requests.get(API_URL, params=params, timeout=300)
        if res.status_code != 200:
            return []
        return res.json().get('data', [])
    except Exception as e:
        print(f'❌ dividend source error {stock_id}: {e}')
        return []


def get_per_raw(stock_id):
    try:
        params = {
            'dataset': 'TaiwanStockPER',
            'data_id': stock_id,
            'start_date': '2023-01-01',
            'token': API_TOKEN,
        }
        res = requests.get(API_URL, params=params, timeout=300)
        return res.json().get('data', [])
    except Exception:
        return []


def get_per_pbr_90d_stats(stock_id, days=90):
    """
    回傳：
    {
        "per": 最新PER,
        "per_90d_high": 90天PER最高,
        "per_90d_low": 90天PER最低,
        "pbr": 最新PBR,
        "pbr_90d_high": 90天PBR最高,
        "pbr_90d_low": 90天PBR最低,
    }
    """
    try:
        start_date = (datetime.now() - timedelta(days=days * 2)
                      ).strftime("%Y-%m-%d")
        # 抓寬一點，避免遇到非交易日不夠 90 筆

        params = {
            "dataset": "TaiwanStockPER",
            "data_id": str(stock_id),
            "start_date": start_date,
            "token": API_TOKEN,
        }

        res = requests.get(API_URL, params=params, timeout=300)
        if res.status_code != 200:
            return {
                "per": None,
                "per_90d_high": None,
                "per_90d_low": None,
                "pbr": None,
                "pbr_90d_high": None,
                "pbr_90d_low": None,
            }

        data = res.json().get("data", [])
        if not data:
            return {
                "per": None,
                "per_90d_high": None,
                "per_90d_low": None,
                "pbr": None,
                "pbr_90d_high": None,
                "pbr_90d_low": None,
            }

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(days).copy()

        # 用候選欄位名提高相容性
        per_col = next(
            (c for c in ["price_to_earning_ratio",
             "PER", "per"] if c in df.columns),
            None
        )
        pbr_col = next(
            (c for c in ["price_book_ratio", "PBR", "pbr"] if c in df.columns),
            None
        )

        if per_col:
            df[per_col] = pd.to_numeric(df[per_col], errors="coerce")
            latest_per = df[per_col].iloc[-1] if not df.empty else None
            per_high = df[per_col].max()
            per_low = df[per_col].min()
        else:
            latest_per = per_high = per_low = None

        if pbr_col:
            df[pbr_col] = pd.to_numeric(df[pbr_col], errors="coerce")
            latest_pbr = df[pbr_col].iloc[-1] if not df.empty else None
            pbr_high = df[pbr_col].max()
            pbr_low = df[pbr_col].min()
        else:
            latest_pbr = pbr_high = pbr_low = None

        def safe_round(v):
            return round(float(v), 2) if pd.notna(v) else None

        return {
            "per": safe_round(latest_per),
            "per_90d_high": safe_round(per_high),
            "per_90d_low": safe_round(per_low),
            "pbr": safe_round(latest_pbr),
            "pbr_90d_high": safe_round(pbr_high),
            "pbr_90d_low": safe_round(pbr_low),
        }

    except Exception as e:
        print(f"❌ PER/PBR 90D error {stock_id}: {e}")
        return {
            "per": None,
            "per_90d_high": None,
            "per_90d_low": None,
            "pbr": None,
            "pbr_90d_high": None,
            "pbr_90d_low": None,
        }
