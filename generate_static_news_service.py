#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日產製 AllStatic_news.csv。

設計目標：
1. GitHub Actions 每天台灣 15:30 執行。
2. 參考 stocks.csv，產出固定欄位：stock_id, name, 產業, 新聞。
3. 「產業」偏半靜態：若 AllStatic_news.csv 已有內容，預設沿用，避免每日漂移。
4. 「新聞」每日更新近十天 Google News RSS 標題後交由 OpenAI API 摘要。
5. 若 OPENAI_API_KEY 未設定或 API 失敗，仍產出 fallback，避免排程中斷。
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import config
except Exception:
    config = None

TAIWAN_TZ = dt.timezone(dt.timedelta(hours=8))
OUTPUT_COLUMNS = ["stock_id", "name", "產業", "新聞"]


@dataclass
class Stock:
    stock_id: str
    name: str


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default


def _detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    if "\t" in first_line:
        return "\t"
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except Exception:
        return ","


def read_stocks(path: str | Path) -> List[Stock]:
    """讀取 stocks.csv；支援現有專案的 Ticker/Name tab 分隔格式。"""
    path = Path(path)
    delimiter = _detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{path} 沒有表頭")

        rows: List[Stock] = []
        for row in reader:
            stock_id = (
                row.get("stock_id")
                or row.get("Ticker")
                or row.get("代碼")
                or row.get("股票代碼")
                or row.get("證券代號")
                or ""
            ).strip()
            name = (
                row.get("name")
                or row.get("Name")
                or row.get("名稱")
                or row.get("股票名稱")
                or row.get("證券名稱")
                or ""
            ).strip()
            if stock_id and name:
                rows.append(Stock(stock_id=stock_id, name=name))
    return rows


def load_existing_news(path: str | Path) -> dict[str, dict]:
    """讀取既有 AllStatic_news.csv，用於保留半靜態的產業摘要。"""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            out: dict[str, dict] = {}
            for row in reader:
                sid = str(row.get("stock_id") or "").strip()
                if sid:
                    out[sid] = row
            return out
    except Exception as exc:
        print(f"⚠️ 讀取既有 AllStatic_news.csv 失敗，將重新產製: {exc}", flush=True)
        return {}


def fetch_google_news_rss(stock: Stock, days: int = 10, max_items: int = 12) -> List[dict]:
    """抓 Google News RSS 標題；只用於新聞摘要，不作投資建議。"""
    query = f'"{stock.name}" OR "{stock.stock_id}" 台股 股票'
    params = urllib.parse.urlencode({
        "q": f"{query} when:{days}d",
        "hl": "zh-TW",
        "gl": "TW",
        "ceid": "TW:zh-Hant",
    })
    url = f"https://news.google.com/rss/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    items: List[dict] = []
    for item in root.findall("./channel/item"):
        title = html.unescape((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        if title:
            items.append({"title": title, "source": source, "link": link, "pubDate": pub_date})
    return items[:max_items]


def _compact_zh(text: str, max_chars: int = 58) -> str:
    text = re.sub(r"\s+", "", str(text or ""))
    text = text.replace("，。", "。")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _fallback_industry(stock: Stock, existing_industry: str = "") -> str:
    if existing_industry:
        return _compact_zh(existing_industry, 58)
    return _compact_zh(
        f"{stock.name}產業與產品定位待AI摘要補齊；請搭配年報、法說與產業資料更新護城河。",
        58,
    )


def _fallback_news(news_items: List[dict]) -> str:
    titles = [str(x.get("title", "")).strip() for x in news_items[:4] if x.get("title")]
    if not titles:
        return "近十天未擷取到明確公司相關新聞。"
    return _compact_zh("；".join(titles), 58)


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def summarize_with_openai(stock: Stock, news_items: List[dict], existing_industry: str = "") -> Tuple[str, str]:
    """用 OpenAI API 整理產業與新聞。

    預設若 existing_industry 有值，會保留該值，只請模型更新新聞；
    若要每天重寫產業，可設定 REFRESH_INDUSTRY=1。
    """
    refresh_industry = _env("REFRESH_INDUSTRY", "0").lower() in {"1", "true", "yes", "y", "on"}
    keep_industry = bool(existing_industry and not refresh_industry)

    api_key = _env("OPENAI_API_KEY", "")
    if not api_key:
        return _fallback_industry(stock, existing_industry), _fallback_news(news_items)

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return _fallback_industry(stock, existing_industry), _fallback_news(news_items)

    titles = [
        f"- {x.get('title', '')}｜{x.get('source', '')}｜{x.get('pubDate', '')}"
        for x in news_items
    ]
    news_text = "\n".join(titles) if titles else "無明確新聞"

    industry_instruction = (
        f"沿用既有 industry，不要改寫：{existing_industry}"
        if keep_industry else
        "產生 industry：該公司在該產業的產品趨勢、產品護城河、產品定位，如獨佔、寡佔性、市占排名；繁體中文約50字。"
    )

    prompt = f"""
你是台股產業與新聞摘要助理。請只輸出 JSON，不要 markdown。
公司：{stock.stock_id} {stock.name}
近十天新聞標題：
{news_text}

任務：
1. {industry_instruction}
2. 產生 news：彙總該公司十天內新聞與關鍵字，繁體中文約50字；若新聞不足，寫「近十天新聞量較少」。

限制：
- 不要給投資建議。
- 不確定市占時，不要編精確數字；改用「具規模」「主要供應鏈」「利基型」等保守描述。
- news 只根據上方新聞標題整理，不要補不存在的事件。
- JSON 格式：{{"industry":"...","news":"..."}}
""".strip()

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你只輸出有效 JSON。"},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = _safe_json_loads(content)
        industry = existing_industry if keep_industry else str(data.get("industry") or "")
        news = str(data.get("news") or "")
        industry = _compact_zh(industry, 58)
        news = _compact_zh(news, 58)
        if not industry or not news:
            raise ValueError("empty summary")
        return industry, news
    except Exception as exc:
        print(f"⚠️ AI 摘要失敗 {stock.stock_id}: {exc}", flush=True)
        return _fallback_industry(stock, existing_industry), _fallback_news(news_items)


def atomic_write_csv(rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(out_path)


def write_allstatic_news(
    stocks: Iterable[Stock],
    out_path: str | Path,
    days: int = 10,
    sleep_sec: float = 0.6,
    max_news_items: int = 12,
) -> None:
    existing = load_existing_news(out_path)
    rows = []
    stock_list = list(stocks)
    for i, stock in enumerate(stock_list, start=1):
        existing_row = existing.get(stock.stock_id, {})
        existing_industry = str(existing_row.get("產業") or existing_row.get("industry_summary") or "").strip()
        try:
            news_items = fetch_google_news_rss(stock, days=days, max_items=max_news_items)
            industry, news = summarize_with_openai(stock, news_items, existing_industry=existing_industry)
            status = "ok"
        except Exception as exc:
            industry = _fallback_industry(stock, existing_industry)
            news = _fallback_news([])
            status = f"fallback: {exc}"

        rows.append({"stock_id": stock.stock_id, "name": stock.name, "產業": industry, "新聞": news})
        print(f"[{i}/{len(stock_list)}] {stock.stock_id} {stock.name} {status}", flush=True)
        if sleep_sec and sleep_sec > 0:
            time.sleep(sleep_sec)

    atomic_write_csv(rows, out_path)


def get_default_stocks_csv() -> str:
    return _env("STOCKS_CSV", getattr(config, "CSV_FILE", "stocks.csv") if config else "stocks.csv")


def get_default_output_csv() -> str:
    config_value = getattr(config, "ALLSTATIC_NEWS_OUTPUT_FILE", "AllStatic_news.csv") if config else "AllStatic_news.csv"
    return _env("ALLSTATIC_NEWS_OUTPUT_FILE", _env("ALLSTATIC_NEWS_CSV", _env("ALLSTATIC_NEWS_FILE", config_value)))


def main() -> None:
    stocks_csv = get_default_stocks_csv()
    out_csv = get_default_output_csv()
    days = _env_int("NEWS_DAYS", 10)
    max_stocks = _env_int("MAX_STOCKS", 0)
    max_news_items = _env_int("NEWS_MAX_ITEMS", 12)
    sleep_sec = _env_float("NEWS_SLEEP_SEC", 0.6)

    stocks = read_stocks(stocks_csv)
    if max_stocks > 0:
        stocks = stocks[:max_stocks]

    print(f"Start AllStatic_news at {dt.datetime.now(TAIWAN_TZ).strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"stocks={len(stocks)}, days={days}, out={out_csv}", flush=True)
    print(f"refresh_industry={_env('REFRESH_INDUSTRY', '0')}, openai_key_present={bool(_env('OPENAI_API_KEY', ''))}", flush=True)
    write_allstatic_news(stocks, out_csv, days=days, sleep_sec=sleep_sec, max_news_items=max_news_items)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
