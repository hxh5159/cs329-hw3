import ipaddress
import json
import os
import re
import socket
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Literal, TypeVar
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
import urllib3
from bs4 import BeautifulSoup
from geopy.exc import GeocoderServiceError
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location
from pydantic import BaseModel, Field

from cs329a_hw3.utils import generate_together, get_model, get_structured_output_mode


class GoogleSearchParams(BaseModel):
    search_query: str = Field(min_length=1, description="Standalone web search terms.")
    num_results: int = Field(default=2, ge=1, le=10)


class StockDataParams(BaseModel):
    ticker: str = Field(description="US stock ticker symbol, such as AAPL.")
    date: str = Field(description="One trading date in YYYY-MM-DD format.")


class ComputeParams(BaseModel):
    wolfram_query: str = Field(description="The actual mathematical query for Wolfram Alpha.")


class GetWeatherParams(BaseModel):
    location: str = Field(description="City with state or country where needed. Omit street addresses.")
    date: str = Field(description="One local calendar date in YYYY-MM-DD format.")
    hour: str = Field(default="12", description="Local hour, HH in 24-hour notation.")


class APIResponse(BaseModel):
    api_name: Literal["google_search", "get_stock_data", "compute", "get_weather"]


Schema = TypeVar("Schema", bound=BaseModel)


@lru_cache(maxsize=4)
def _city_geocoder(domain: str) -> Callable[[str], Location | None]:
    # 公开 Nominatim 按实例串行调用，每秒最多一次，并复用同一城市的经纬度。
    geocoder = Nominatim(domain=domain, user_agent="cs329a-hw3-learning-project", timeout=8)
    limited = RateLimiter(
        geocoder.geocode, min_delay_seconds=1, max_retries=0, swallow_exceptions=False
    )
    cached = lru_cache(maxsize=64)(limited)
    lock = Lock()

    def geocode(city: str) -> Location | None:
        with lock:
            return cached(city)

    return geocode


class APIManager:
    """用语言模型把自然语言问题转换为有明确参数的工具调用。"""

    def __init__(
        self,
        google_api_key: str | None = None,
        google_cx_id: str | None = None,
        polygon_api_key: str | None = None,
        wolfram_app_id: str | None = None,
        router_model: str | None = None,
    ):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.google_cx_id = google_cx_id or os.getenv("GOOGLE_CX_ID")
        self.polygon_api_key = polygon_api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        self.wolfram_app_id = wolfram_app_id or os.getenv("WOLFRAM_APP_ID")
        self.router_model = router_model or get_model("router")

        # 以模块所在目录为基准，Notebook 和已安装的包都能找到天气代码表。
        codes_path = Path(__file__).with_name("weather_codes.json")
        self._weather_codes = json.loads(codes_path.read_text(encoding="utf-8"))

        domain = os.getenv("NOMINATIM_DOMAIN") or "nominatim.openstreetmap.org"
        self._geocode = _city_geocoder(domain)

        self.api_functions = {
            "google_search": (self.google_search, GoogleSearchParams),
            "get_stock_data": (self.get_stock_data, StockDataParams),
            "compute": (self.compute, ComputeParams),
            "get_weather": (self.get_weather, GetWeatherParams),
        }

    @staticmethod
    def _structured_response(
        query: str, instruction: str, schema: type[Schema], model: str
    ) -> Schema | None:
        # 服务商支持的格式各有不同。每种模式都在本地校验字段和类型，再调用实际工具。
        schema_definition = schema.model_json_schema()
        schema_text = json.dumps(schema_definition, ensure_ascii=False)
        mode = get_structured_output_mode()
        response_format = None
        if mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema_definition},
            }
        elif mode == "json_object":
            response_format = {"type": "json_object"}

        try:
            message = generate_together(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{instruction} Return only JSON matching this schema.\n{schema_text}",
                    },
                    {"role": "user", "content": query},
                ],
                response_format=response_format,
                temperature=0,
            )
            if message is None or not message.content:
                return None
            content = message.content.strip()
            if mode == "prompt":
                # 个别模型会给纯 JSON 加一层 Markdown 代码围栏。
                fenced = re.fullmatch(
                    r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content, re.IGNORECASE | re.DOTALL
                )
                if fenced is not None:
                    content = fenced.group(1)
            return schema.model_validate_json(content)
        except ValueError:
            return None

    def _parse_query_params(
        self, query: str, function_name: str, model: str | None = None
    ) -> dict[str, Any] | None:
        tool = self.api_functions.get(function_name)
        if tool is None:
            return None
        function, schema = tool
        today = datetime.now(timezone.utc).date().isoformat()
        parsed = self._structured_response(
            query,
            (
                f"Extract the arguments for {function_name}. {function.__doc__} "
                f"The current UTC date is {today}. Preserve explicitly supplied dates and times. "
                "Select parameters that retrieve the evidence needed to answer the question."
            ),
            schema,
            model or self.router_model,
        )
        return parsed.model_dump() if parsed is not None else None

    def route_query(self, query: str, model: str | None = None) -> dict[str, Any]:
        """选择工具、抽取并校验参数，返回工具名称、参数和原始结果。"""
        output: dict[str, Any] = {"api_used": None, "params": {}, "results": None}
        if not query.strip():
            return {**output, "error": "Query is empty."}

        # 讲义第 4 讲把工具调用写成「动作 → 观察」。动作只能选固定映射中的工具，
        # 两步校验后才执行。工具原始结果作为观察交给后续回答模型。
        chosen = self._structured_response(
            query,
            (
                "Choose the single tool that can obtain the most relevant evidence. "
                "Use google_search for web facts, recent events, and literature; "
                "get_stock_data for daily stock market data; compute for mathematical operations; "
                "get_weather for weather at a location, date, and local hour."
            ),
            APIResponse,
            model or self.router_model,
        )
        if chosen is None:
            return {**output, "error": "The model did not select a valid tool."}

        output["api_used"] = chosen.api_name
        params = self._parse_query_params(query, chosen.api_name, model)
        if params is None:
            return {**output, "error": "The model did not provide valid tool parameters."}

        output["params"] = params
        function, _ = self.api_functions[chosen.api_name]
        results = function(**params)
        output["results"] = results
        if isinstance(results, dict) and "error" in results:
            output["error"] = results["error"]
        if isinstance(results, list) and len(results) == 1 and "error" in results[0]:
            output["error"] = results[0]["error"]
        return output

    @staticmethod
    def _get_json(
        url: str, *, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        # 外部 API 的错误信息可能包含 URL 中的密钥。只返回状态或失败类型。
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=(5, 15), allow_redirects=False
            )
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
            result = response.json()
        except requests.RequestException as error:
            raise ValueError(type(error).__name__) from None
        if not isinstance(result, dict):
            raise ValueError("The service did not return a JSON object.")
        return result

    @staticmethod
    def _request_webpage(url: str) -> tuple[str, bytes]:
        """对一个已解析到公网 IP 的网页发请求；返回重定向目标或可解析的正文。"""
        if any(ord(character) < 32 for character in url):
            return "", b""
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username is not None:
            return "", b""
        if not parsed.hostname or "%" in parsed.hostname:
            return "", b""

        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii")
        port = 443 if parsed.scheme == "https" else 80
        if parsed.port not in {None, port}:
            return "", b""

        # 每个跳转都重新验证。固定到同一次 DNS 查询的公网 IP，避免连接时再次解析域名。
        addresses = [
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ]
        if not addresses or any(not ip.is_global or ip.is_multicast for ip in addresses):
            return "", b""
        ip = str(min(addresses, key=lambda address: address.version))
        if parsed.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                ip, port=port, server_hostname=host, assert_hostname=host
            )
        else:
            pool = urllib3.HTTPConnectionPool(ip, port=port)

        # 直连校验过的地址，不使用环境代理。TLS 和 Host 仍按原域名校验与路由。
        target = requests.utils.requote_uri(urlunsplit(("", "", parsed.path or "/", parsed.query, "")))
        host_header = f"[{host}]" if ":" in host else host
        with pool:
            response = pool.urlopen(
                "GET",
                target,
                headers={
                    "Host": host_header,
                    "User-Agent": "cs329a-hw3-learning-project",
                    "Accept-Encoding": "identity",
                },
                redirect=False,
                retries=False,
                preload_content=False,
                timeout=urllib3.Timeout(connect=3, read=5, total=8),
            )
            try:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location", "")
                    return (urljoin(url, location) if location else ""), b""
                if response.status != 200:
                    return "", b""
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                    return "", b""
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    return "", b""
                # 网页是外部输入，限制单次读入量。搜索摘要可在正文不可读时继续使用。
                return "", response.read(1_000_000, decode_content=False)
            finally:
                response.close()

    def _extract_webpage_content(self, url: str) -> str:
        """读取公开网页的可见文字，最多保留 2000 字符。"""
        try:
            for _ in range(4):
                redirect, body = self._request_webpage(url)
                if redirect:
                    url = redirect
                    continue
                if body:
                    soup = BeautifulSoup(body, "html.parser")
                    for element in soup(["script", "style", "noscript", "template", "nav", "footer"]):
                        element.extract()
                    scope = soup.find("main") or soup.find("article") or soup.body or soup
                    return scope.get_text(" ", strip=True)[:2000]
                break
        except (urllib3.exceptions.HTTPError, OSError, ValueError):
            pass
        return ""

    def google_search(self, search_query: str, num_results: int = 2) -> list[dict]:
        """查询近期事实、网页和公开文献并返回标题、链接、摘要及可读正文。"""
        if not search_query.strip() or num_results < 1:
            return [{"error": "Provide a search query and request at least one result."}]
        num_results = min(num_results, 10)
        matches = []
        source = "Google Custom Search"
        if self.google_api_key and self.google_cx_id:
            try:
                result = self._get_json(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": self.google_api_key,
                        "cx": self.google_cx_id,
                        "q": search_query,
                        "num": num_results,
                    },
                )
                matches = result.get("items") or []
            except ValueError:
                pass

        if not matches:
            # Google 已停止接受新用户。备用搜索结果映射为相同结构，调用方可以共用。
            source = "DDGS"
            try:
                from ddgs import DDGS
                from ddgs.exceptions import DDGSException
            except ImportError:
                pass
            else:
                try:
                    matches = DDGS(timeout=8).text(search_query, max_results=num_results) or []
                except (DDGSException, OSError):
                    pass

        if not matches:
            # 公共搜索偶尔限流。Wikipedia 的官方 API 无需密钥，但仅返回英文百科条目。
            source = "English Wikipedia"
            try:
                result = self._get_json(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "generator": "search",
                        "gsrsearch": search_query,
                        "gsrlimit": num_results,
                        "prop": "info|extracts",
                        "inprop": "url",
                        "exintro": 1,
                        "explaintext": 1,
                        "exlimit": num_results,
                        "exchars": 1200,
                        "format": "json",
                        "formatversion": 2,
                    },
                    headers={"User-Agent": "cs329a-hw3-learning-project/0.1 (Python)"},
                )
            except ValueError as error:
                return [{"error": f"Web search failed and the Wikipedia fallback is unavailable: {error}."}]
            if "error" in result:
                return [{"error": "Wikipedia could not process the search."}]
            pages = result.get("query", {}).get("pages", [])
            for page in sorted(pages, key=lambda item: item.get("index", 0)):
                content = (page.get("extract") or "").strip()[:2000]
                matches.append({
                    "title": page.get("title"),
                    "link": page.get("fullurl"),
                    "snippet": " ".join(content.split())[:300],
                    "content": content,
                })

        # 讲义第 7 讲强调从搜索结果提炼与问题相关的证据。保留链接、摘要和短正文，
        # 后续模型可以引用给定出处，并在正文无法加载时继续使用搜索摘要。
        results = []
        seen_links = set()
        for match in matches:
            link = match.get("link") or match.get("href")
            if not link or not link.startswith(("https://", "http://")) or link in seen_links:
                continue
            seen_links.add(link)
            results.append(
                {
                    "title": match.get("title") or "",
                    "link": link,
                    "snippet": match.get("snippet") or match.get("body") or "",
                    "content": (
                        match["content"] if source == "English Wikipedia" else self._extract_webpage_content(link)
                    ),
                    "source": source,
                }
            )
        return results

    def get_stock_data(self, ticker: str, date: str) -> dict[str, Any]:
        """查询一只美国股票在指定交易日的开盘、最高、最低、收盘和成交量。"""
        if not self.polygon_api_key:
            return {"error": "Set MASSIVE_API_KEY or POLYGON_API_KEY to query stock data."}
        ticker = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,31}", ticker):
            return {"error": "Invalid stock ticker."}
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            return {"error": "Use YYYY-MM-DD for the trading date."}
        try:
            datetime.strptime(date, "%Y-%m-%d")
            result = self._get_json(
                f"https://api.massive.com/v1/open-close/{ticker}/{date}",
                params={"adjusted": "true"},
                headers={"Authorization": f"Bearer {self.polygon_api_key}"},
            )
        except ValueError as error:
            return {"error": f"Stock data unavailable: {error}."}
        if any(result.get(field) is None for field in ("open", "high", "low", "close", "volume")):
            return {"error": f"No daily bar found for {ticker} on {date}."}
        # 开盘和收盘是不同时间的价格。保留两者，回答模型才能比较同日开收盘。
        return {
            "date": result.get("from", date),
            "ticker": ticker,
            "open": result["open"],
            "high": result["high"],
            "low": result["low"],
            "close": result["close"],
            "volume": result["volume"],
            "currency": "USD",
        }

    def get_weather(self, location: str, date: str, hour: str = "12") -> dict[str, Any]:
        """按所在地日期和当地小时查询温度、天气现象、湿度和 10 米高度风速。"""
        if not location.strip():
            return {"error": "Provide a location for the weather query."}
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or not re.fullmatch(r"\d{1,2}", hour):
            return {"error": "Use YYYY-MM-DD for the date and an hour from 00 to 23."}
        try:
            requested_day = datetime.strptime(date, "%Y-%m-%d").date()
            local_hour = int(hour)
        except ValueError:
            return {"error": "Invalid weather date or hour."}
        if not 0 <= local_hour <= 23:
            return {"error": "Use an hour from 00 to 23."}

        try:
            point = self._geocode(location.strip())
        except GeocoderServiceError:
            return {"error": "Nominatim could not look up the location."}
        if point is None:
            return {"error": f"Location not found: {location}."}

        # 历史再分析数据有发布延迟。最近几天和预报使用 Forecast API。
        historical = (datetime.now(timezone.utc).date() - requested_day).days > 7
        endpoint = (
            "https://archive-api.open-meteo.com/v1/archive"
            if historical
            else "https://api.open-meteo.com/v1/forecast"
        )
        fields = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "weather_code"]
        try:
            result = self._get_json(
                endpoint,
                params={
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "timezone": "auto",
                    "start_date": date,
                    "end_date": date,
                    "hourly": ",".join(fields),
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                },
            )
        except ValueError as error:
            return {"error": f"Open-Meteo data unavailable: {error}."}

        # 按本地时间戳定位这一小时；模型可直接使用结果中的摄氏或华氏温度。
        timestamp = f"{date}T{local_hour:02d}:00"
        try:
            hourly = result["hourly"]
            index = hourly["time"].index(timestamp)
            temperature, humidity, wind_speed, code = (hourly[field][index] for field in fields)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            return {"error": f"Open-Meteo has no record for {timestamp} local time."}

        if all(value is None for value in (temperature, humidity, wind_speed, code)):
            return {"error": f"Open-Meteo has no measurements for {timestamp} local time."}
        return {
            "location": location,
            "local_time": timestamp,
            "timezone": result.get("timezone"),
            "temperature": temperature,
            "temperature_unit": "°C",
            "temperature_f": round(temperature * 9 / 5 + 32, 2) if temperature is not None else None,
            "humidity": humidity,
            "humidity_unit": "%",
            "wind_speed": wind_speed,
            "wind_speed_unit": "km/h",
            "weather_code": code,
            "weather_description": self._weather_codes.get(str(code)),
            "data_source": "Open-Meteo Historical Weather" if historical else "Open-Meteo Weather Forecast",
            "geocoding_attribution": "© OpenStreetMap contributors (Nominatim)",
        }

    def compute(self, wolfram_query: str) -> dict[str, str]:
        """通过 Wolfram Alpha 执行数学计算，提取主要结果的纯文本。"""
        if not self.wolfram_app_id:
            return {"error": "Set WOLFRAM_APP_ID to use Wolfram Alpha."}
        if not wolfram_query.strip():
            return {"error": "Provide a mathematical query."}
        try:
            result = self._get_json(
                "https://api.wolframalpha.com/v2/query",
                params={
                    "appid": self.wolfram_app_id,
                    "input": wolfram_query,
                    "format": "plaintext",
                    "output": "json",
                },
            ).get("queryresult", {})
        except ValueError as error:
            return {"error": f"Wolfram Alpha could not complete the computation: {error}."}
        if not result.get("success"):
            return {"error": "Wolfram Alpha did not find a result for this query."}

        # Full Results API 把输入解释和计算结果分到多个 pod。优先保留主要答案。
        pods = result.get("pods") or []
        primary = [pod for pod in pods if pod.get("primary") or pod.get("id") == "Result"]
        answers = []
        for pod in primary or pods:
            if pod.get("id") in {"Input", "InputInterpretation"}:
                continue
            for subpod in pod.get("subpods") or []:
                plaintext = (subpod.get("plaintext") or "").strip()
                if plaintext and plaintext not in answers:
                    answers.append(plaintext)
        if not answers:
            return {"error": "Wolfram Alpha returned no textual result."}
        # 后续模型以工具算出的数值为依据，负责表达和按题意取整。
        return {"result": "\n".join(answers)[:2000]}
