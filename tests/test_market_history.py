import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import requests

from cs329a_hw3.market_history import CHART_BASE, get_us_daily_bar


BAR = {
    "meta": {"symbol": "MSFT", "currency": "USD", "exchangeTimezoneName": "America/New_York",
             "instrumentType": "EQUITY", "dataGranularity": "1d"},
    "timestamp": [1718199000],  # 2024-06-12 09:30 ET
    "indicators": {
        "quote": [{"open": [100.0], "high": [104.0], "low": [99.0], "close": [101.0], "volume": [400]}],
        "adjclose": [{"adjclose": [90.0]}],
    },
}


class MarketHistoryTests(unittest.TestCase):
    def setUp(self):
        self.dataset = deepcopy(BAR)
        response = Mock(status_code=200)
        response.json.return_value = {"chart": {"result": [self.dataset], "error": None}}
        patched = patch.object(requests.Session, "get", return_value=response)
        self.get = patched.start()
        self.addCleanup(patched.stop)

    def test_dated_quote_and_source(self):
        result = get_us_daily_bar("msft", "2024-06-12")
        self.assertEqual((result["close"], result["currency"], result["date"]), (101.0, "USD", "2024-06-12"))
        self.assertTrue(result["source_url"].startswith(f"{CHART_BASE}/MSFT?period1="))
        args, kwargs = self.get.call_args
        self.assertEqual(args, (f"{CHART_BASE}/MSFT",))
        self.assertEqual(kwargs["params"]["period2"] - kwargs["params"]["period1"], 86400)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(set(kwargs["headers"]), {"User-Agent"})

    def test_invalid_arguments_make_no_request(self):
        for ticker, day in (("MSFT/path", "2024-06-12"), ("MSFT", "20240612"),
                            ("MSFT", "2024-02-30"), ("MSFT", "2999-06-01")):
            with self.subTest(ticker=ticker, day=day):
                self.assertIsNone(get_us_daily_bar(ticker, day))
        self.get.assert_not_called()

    def test_no_adjacent_day_or_adjusted_close(self):
        self.assertIsNone(get_us_daily_bar("MSFT", "2024-06-13"))
        self.dataset["indicators"]["quote"][0]["close"][0] = None
        self.assertIsNone(get_us_daily_bar("MSFT", "2024-06-12"))

    def test_skip_mismatched_metadata_and_bad_ohlc(self):
        for key, value in (("currency", "EUR"), ("symbol", "OTHER"), ("dataGranularity", "1h"),
                           ("exchangeTimezoneName", "Europe/London")):
            with self.subTest(key=key):
                with patch.dict(self.dataset["meta"], {key: value}):
                    self.assertIsNone(get_us_daily_bar("MSFT", "2024-06-12"))
        self.dataset["indicators"]["quote"][0]["open"] = [1000.0]
        self.assertIsNone(get_us_daily_bar("MSFT", "2024-06-12"))

    def test_redirect_service_error_and_empty_chart(self):
        for status in (302, 429, 500):
            self.get.return_value.status_code = status
            self.assertIsNone(get_us_daily_bar("MSFT", "2024-06-12"))
        self.get.return_value.status_code = 200
        self.get.return_value.json.return_value = {"chart": {"result": None, "error": {"code": "Not Found"}}}
        self.assertIsNone(get_us_daily_bar("MSFT", "2024-06-12"))
        self.get.side_effect = requests.Timeout()
        self.assertIsNone(get_us_daily_bar("MSFT", "2024-06-12"))
