"""Anonymous historical US equity/ETF daily bars for the full Part 3 pipeline."""

import math
import re
from datetime import date as CalendarDate, datetime, timezone
from urllib.parse import urlencode, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"


def get_us_daily_bar(ticker: str, date: str) -> dict | None:
    """Return Yahoo's USD quote for exactly one completed US trading date, else None.

    The OHLC values come directly from indicators.quote. Yahoo's historical Close
    accounts for stock splits; the dividend-adjusted indicators.adjclose series
    is a different field. Volume is the provider's daily share count.
    """
    if not isinstance(ticker, str) or not isinstance(date, str):
        return None
    symbol = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:[-.][A-Z0-9]{1,3})?", symbol):
        return None
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date):
        return None
    try:
        requested = CalendarDate.fromisoformat(date)
        if requested < CalendarDate(1970, 1, 1) or requested >= datetime.now(timezone.utc).date():
            return None
        # 美国股票的常规交易时段位于同一 UTC 日。返回后再按交易所本地日期
        # 核验，避免休市日查不到报价时误把邻近交易日的价格当成答案。
        start = (requested - CalendarDate(1970, 1, 1)).days * 86400
        params = {"period1": start, "period2": start + 86400, "interval": "1d"}
        url = f"{CHART_BASE}/{symbol}"
        with requests.Session() as client:
            # 使用系统常规网络路径。禁用 .netrc 自动认证、账户 cookie 和重定向，
            # 只继承匿名代理地址。完整来源 URL 没有密钥，可随结果一起核对。
            client.trust_env = False
            proxies = requests.utils.get_environ_proxies(url)
            for protocol in ("https", "all"):
                proxy = proxies.get(protocol)
                if proxy and urlsplit(proxy).username is None:
                    client.proxies[protocol] = proxy
            response = client.get(
                url, params=params, headers={"User-Agent": "CS329A-learning-project/1.0"},
                timeout=(4, 10), allow_redirects=False,
            )
            if response.status_code != 200:
                return None
            chart = response.json()["chart"]
        if chart.get("error") or len(chart["result"]) != 1:
            return None
        dataset = chart["result"][0]
        meta = dataset["meta"]
        if (meta.get("symbol") != symbol or meta.get("currency") != "USD"
                or meta.get("instrumentType") not in ("EQUITY", "ETF")
                or meta.get("exchangeTimezoneName") != "America/New_York"
                or meta.get("dataGranularity") != "1d"):
            return None
        zone = ZoneInfo("America/New_York")
        quotes = dataset["indicators"]["quote"][0]
        for index, timestamp in enumerate(dataset["timestamp"]):
            if datetime.fromtimestamp(timestamp, zone).date() != requested:
                continue
            prices = {field: quotes[field][index] for field in ("open", "high", "low", "close")}
            if any(type(p) not in (float, int) or not math.isfinite(p) or p <= 0 for p in prices.values()):
                return None
            if min(prices.values()) != prices["low"] or max(prices.values()) != prices["high"]:
                return None
            volume = quotes["volume"][index]
            if type(volume) is not int or volume < 0:
                return None
            return {
                "ticker": symbol, "date": date, "currency": "USD", **prices, "volume": volume,
                "timezone": zone.key, "source": "Yahoo Finance Chart",
                "source_url": f"{url}?{urlencode(params)}",
                "price_basis": "Daily Yahoo quote OHLC; split-adjusted close, excluding dividend adjustments.",
            }
    except (requests.RequestException, ValueError, TypeError, KeyError, IndexError, OverflowError,
            OSError, ZoneInfoNotFoundError):
        # 没有可验证的当日数据时，由调用方记录证据缺口并决定是否采用其他工具。
        return None
    return None
