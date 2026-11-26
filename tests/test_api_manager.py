"""覆盖 HW3 的工具契约和失败路径。测试使用离线的模型与服务响应。"""

import json
import os
import socket
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from ddgs.exceptions import DDGSException

from cs329a_hw3.api_manager import APIManager


MODULE = "cs329a_hw3.api_manager"


def message(payload):
    return SimpleNamespace(content=json.dumps(payload))


class APIManagerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.manager = APIManager(router_model="test-router")

    def test_routes_and_validates_parameters_before_using_tool(self):
        result = {"local_time": "2025-08-01T12:00", "temperature": 22.0}
        tool, schema = self.manager.api_functions["get_weather"]
        with patch.object(self.manager, tool.__name__, return_value=result) as weather:
            self.manager.api_functions["get_weather"] = (weather, schema)
            with patch(
                f"{MODULE}.generate_together",
                side_effect=[
                    message({"api_name": "get_weather"}),
                    message({"location": "Boston, MA", "date": "2025-08-01"}),
                ],
            ) as model:
                output = self.manager.route_query("What was the weather in Boston on August 1, 2025?")
        weather.assert_called_once_with(location="Boston, MA", date="2025-08-01", hour="12")
        self.assertEqual(output["api_used"], "get_weather")
        self.assertEqual(output["params"]["hour"], "12")
        self.assertEqual(output["results"], result)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.call_args.kwargs["model"], "test-router")
        kwargs = model.call_args.kwargs
        response_schema = kwargs["response_format"]["json_schema"]["schema"]
        self.assertIn(json.dumps(response_schema, ensure_ascii=False), kwargs["messages"][0]["content"])

    def test_route_rejects_unknown_tools_and_malformed_parameters(self):
        with patch(f"{MODULE}.generate_together", return_value=message({"api_name": "unknown"})):
            output = self.manager.route_query("Use a tool")
        self.assertIsNone(output["api_used"])
        self.assertIn("error", output)

        with patch(
            f"{MODULE}.generate_together",
            side_effect=[
                message({"api_name": "get_weather"}),
                message({"location": "Seattle"}),  # 缺少必填日期，工具不能收到残缺参数。
            ],
        ), patch.object(self.manager, "_get_json") as request:
            output = self.manager.route_query("Weather in Seattle")
        self.assertEqual(output["api_used"], "get_weather")
        self.assertIn("error", output)
        request.assert_not_called()

    def test_route_attaches_requested_daily_summary_flag_only_to_weather(self):
        cases = [
            (
                "get_weather", {"location": "New York", "date": "2025-08-01"},
                {"location": "New York", "date": "2025-08-01", "hour": "12", "include_daily_summary": True},
            ),
            ("compute", {"wolfram_query": "2+2"}, {"wolfram_query": "2+2"}),
        ]
        for tool, parsed, passed in cases:
            with self.subTest(tool=tool):
                _, schema = self.manager.api_functions[tool]
                result = {"actual_result": "available"}
                execute = Mock(return_value=result)
                with patch.dict(self.manager.api_functions, {tool: (execute, schema)}), patch(
                    f"{MODULE}.generate_together", side_effect=[message({"api_name": tool}), message(parsed)]
                ):
                    output = self.manager.route_query("Query", include_daily_weather=True)
                execute.assert_called_once_with(**passed)
                self.assertEqual(output["results"], result)

    def test_routes_with_json_object_and_prompt_only_providers(self):
        examples = [
            ("json_object", {"type": "json_object"}, message({"api_name": "compute"})),
            ("prompt", None, SimpleNamespace(content='```json\n{"api_name":"compute"}\n```')),
        ]
        for mode, response_format, selection in examples:
            with self.subTest(mode=mode):
                with patch(f"{MODULE}.get_structured_output_mode", return_value=mode), patch(
                    f"{MODULE}.generate_together",
                    side_effect=[selection, message({"wolfram_query": "2+2"})],
                ) as model:
                    output = self.manager.route_query("What is 2+2?")

                self.assertEqual(output["api_used"], "compute")
                self.assertEqual(output["params"], {"wolfram_query": "2+2"})
                self.assertEqual(model.call_count, 2)
                for call in model.call_args_list:
                    self.assertEqual(call.kwargs["response_format"], response_format)
                    self.assertIn(
                        "Return only JSON matching this schema", call.kwargs["messages"][0]["content"]
                    )

    def test_prompt_only_still_validates_tool_parameters(self):
        with patch(f"{MODULE}.get_structured_output_mode", return_value="prompt"), patch(
            f"{MODULE}.generate_together",
            side_effect=[message({"api_name": "compute"}), message({"wolfram_query": []})],
        ), patch.object(self.manager, "_get_json") as request:
            output = self.manager.route_query("What is 2+2?")
        self.assertEqual(output["api_used"], "compute")
        self.assertEqual(output["params"], {})
        self.assertIn("error", output)
        request.assert_not_called()

    def test_route_preserves_tool_error_in_response(self):
        with patch(
            f"{MODULE}.generate_together",
            side_effect=[
                message({"api_name": "compute"}),
                message({"wolfram_query": "2+2"}),
            ],
        ):
            output = self.manager.route_query("What is 2+2?")
        self.assertEqual(output["params"], {"wolfram_query": "2+2"})
        self.assertIn("WOLFRAM_APP_ID", output["error"])
        self.assertEqual(output["results"]["error"], output["error"])

    def test_google_preferred_and_ddgs_fallback_normalizes_results(self):
        self.manager.google_api_key = "fake-key"
        self.manager.google_cx_id = "fake-search-engine"
        link = "https://example.org/article"
        with patch.object(
            self.manager, "_get_json", return_value={"items": [{"title": "Google", "link": link}]}
        ), patch.object(self.manager, "_extract_webpage_content", return_value="Page text"):
            output = self.manager.google_search("query")
        self.assertEqual(output[0]["source"], "Google Custom Search")
        self.assertEqual(output[0]["content"], "Page text")

        with patch.object(self.manager, "_get_json", side_effect=ValueError("HTTP 403")), patch(
            "ddgs.DDGS"
        ) as ddgs, patch.object(self.manager, "_extract_webpage_content", return_value=""):
            hit = {"title": "DDGS", "href": link, "body": "Search snippet"}
            ddgs.return_value.text.return_value = [hit, hit]
            output = self.manager.google_search("query", num_results=2)
        ddgs.return_value.text.assert_called_once_with("query", max_results=2)
        self.assertEqual(output, [
            {
                "title": "DDGS", "link": link, "snippet": "Search snippet",
                "content": "", "source": "DDGS",
            }
        ])

    def test_wikipedia_is_labeled_when_general_search_is_unavailable(self):
        payload = {"query": {"pages": [{
            "title": "Cowboy Carter", "fullurl": "https://en.wikipedia.org/wiki/Cowboy_Carter",
            "extract": "Country album", "index": 1,
        }]}}
        with patch("ddgs.DDGS") as ddgs, patch.object(
            self.manager, "_get_json", return_value=payload
        ) as request, patch.object(self.manager, "_extract_webpage_content") as fetch:
            ddgs.return_value.text.side_effect = DDGSException("No hits")
            output = self.manager.google_search("Country album")
        self.assertEqual(request.call_args.args[0], "https://en.wikipedia.org/w/api.php")
        fetch.assert_not_called()
        self.assertEqual(output, [{
            "title": "Cowboy Carter", "link": "https://en.wikipedia.org/wiki/Cowboy_Carter",
            "snippet": "Country album", "content": "Country album", "source": "English Wikipedia",
        }])

    def test_stock_and_wolfram_preserve_key_fields(self):
        self.manager.polygon_api_key = "fake-stock-key"
        bar = {"open": 10, "high": 13, "low": 9, "close": 12, "volume": 1000}
        with patch.object(self.manager, "_get_json", return_value=bar) as request:
            output = self.manager.get_stock_data("aapl", "2025-03-05")
        self.assertEqual(output, {**bar, "date": "2025-03-05", "ticker": "AAPL", "currency": "USD"})
        self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], "Bearer fake-stock-key")

        self.manager.wolfram_app_id = "fake-wolfram-id"
        payload = {"queryresult": {"success": True, "pods": [
            {"id": "Input", "subpods": [{"plaintext": "2+2"}]},
            {"id": "Result", "primary": True, "subpods": [{"plaintext": "4"}]},
        ]}}
        with patch.object(self.manager, "_get_json", return_value=payload) as request:
            output = self.manager.compute("2+2")
        self.assertEqual(output, {"result": "4"})
        self.assertEqual(request.call_args.kwargs["params"]["output"], "json")

    def test_weather_uses_local_hour_and_explicit_units_in_both_endpoints(self):
        cases = [
            ("2020-08-01", "https://archive-api.open-meteo.com/v1/archive"),
            (datetime.now(timezone.utc).date().isoformat(), "https://api.open-meteo.com/v1/forecast"),
        ]
        for day, endpoint in cases:
            payload = {"timezone": "America/New_York", "hourly": {
                "time": [f"{day}T00:00", f"{day}T09:00"],
                "temperature_2m": [10, 25],
                "relative_humidity_2m": [70, 40],
                "wind_speed_10m": [8, 16],
                "weather_code": [0, 95],
            }}
            with self.subTest(day=day), patch.object(
                self.manager, "_geocode", return_value=SimpleNamespace(latitude=42.3, longitude=-71.1)
            ), patch.object(self.manager, "_get_json", return_value=payload) as request:
                output = self.manager.get_weather("Boston, MA", day, "9")
                self.assertEqual(output["local_time"], f"{day}T09:00")
                self.assertEqual(output["timezone"], "America/New_York")
                self.assertEqual(output["temperature"], 25)
                self.assertEqual(output["temperature_unit"], "°C")
                self.assertEqual(output["temperature_f"], 77)
                self.assertEqual(output["humidity"], 40)
                self.assertEqual(output["humidity_unit"], "%")
                self.assertEqual(output["wind_speed_unit"], "km/h")
                self.assertEqual(output["weather_description"], "Thunderstorm")
                self.assertIn("OpenStreetMap contributors", output["geocoding_attribution"])
                self.assertNotIn("daily_wind_summary", output)
                self.assertEqual(request.call_args.args[0], endpoint)
                self.assertEqual(request.call_args.kwargs["params"]["timezone"], "auto")

    def test_daily_wind_summary_uses_available_samples_and_flags_missing_coverage(self):
        day = "2025-08-01"
        hours = list(range(24))
        winds = [float(hour) for hour in hours]
        winds[6] = None
        payload = {"timezone": "America/New_York", "hourly": {
            "time": [f"{day}T{hour:02d}:00" for hour in hours] + ["2025-08-02T00:00"],
            "temperature_2m": [20] * 24,
            "relative_humidity_2m": [50] * 24,
            "wind_speed_10m": winds + [180],
            "weather_code": [0] * 24,
        }}
        with patch.object(self.manager, "_geocode", return_value=SimpleNamespace(
            latitude=40.7, longitude=-74.0
        )), patch.object(self.manager, "_get_json", return_value=payload) as request:
            weather = self.manager.get_weather("New York", day, include_daily_summary=True)
        request.assert_called_once()
        summary = weather["daily_wind_summary"]
        self.assertEqual(summary["date"], day)
        self.assertEqual(summary["sample_count"], 23)
        self.assertEqual(summary["max_wind_speed"], 23)
        self.assertEqual(summary["wind_speed_unit"], "km/h")
        self.assertAlmostEqual(summary["mean_wind_speed"], (sum(range(24)) - 6) / 23)
        if summary["expected_sample_count"] is None:
            self.assertEqual(summary["coverage"], "unknown")
        else:
            self.assertEqual(summary["expected_sample_count"], 24)
            self.assertEqual(summary["coverage"], "partial")

    def test_daily_wind_handles_spring_gap_and_autumn_fold(self):
        try:
            ZoneInfo("America/New_York")
        except ZoneInfoNotFoundError:
            self.skipTest("IANA timezone database is unavailable")

        cases = [
            ("2025-03-09", [0, 1, *range(3, 24)], "02", 23, "complete"),
            ("2025-11-02", [0, 1, 1, *range(2, 24)], "12", 25, "complete"),
            ("2025-11-02", list(range(24)), "12", 25, "partial"),
        ]
        for day, hours, requested_hour, expected, coverage in cases:
            with self.subTest(day=day, coverage=coverage):
                payload = {"timezone": "America/New_York", "hourly": {
                    "time": [f"{day}T{hour:02d}:00" for hour in hours],
                    "temperature_2m": [20] * len(hours),
                    "relative_humidity_2m": [50] * len(hours),
                    "wind_speed_10m": [10] * len(hours),
                    "weather_code": [0] * len(hours),
                }}
                with patch.object(self.manager, "_geocode", return_value=SimpleNamespace(
                    latitude=40.7, longitude=-74.0
                )), patch.object(self.manager, "_get_json", return_value=payload):
                    weather = self.manager.get_weather("New York", day, requested_hour, include_daily_summary=True)
                summary = weather["daily_wind_summary"]
                self.assertEqual(summary["sample_count"], len(hours))
                self.assertEqual(summary["expected_sample_count"], expected)
                self.assertEqual(summary["coverage"], coverage)
                self.assertEqual(summary["mean_wind_speed"], 10)
                if requested_hour == "02":
                    self.assertIn("hourly_error", weather)
                    self.assertNotIn("error", weather)

    def test_web_fetch_blocks_private_addresses_and_checks_redirects(self):
        public = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]
        private = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))]
        with patch(f"{MODULE}.socket.getaddrinfo", side_effect=[public, private]), patch(
            f"{MODULE}.urllib3.HTTPSConnectionPool"
        ) as pool:
            response = pool.return_value.urlopen.return_value
            response.status = 302
            response.headers = {"Location": "https://private.example/"}
            self.assertEqual(self.manager._extract_webpage_content("https://example.org/"), "")
        pool.assert_called_once_with(
            "93.184.216.34", port=443, server_hostname="example.org", assert_hostname="example.org"
        )
        self.assertFalse(pool.return_value.urlopen.call_args.kwargs["redirect"])
        self.assertEqual(pool.return_value.urlopen.call_args.kwargs["headers"]["Host"], "example.org")
        response.close.assert_called_once()

    def test_web_fetch_strips_noncontent_and_truncates_body(self):
        body = b"<nav>Menu</nav><main><script>hidden</script><p>" + b"x" * 2500 + b"</p></main>"
        with patch.object(self.manager, "_request_webpage", return_value=("", body)):
            self.assertEqual(self.manager._extract_webpage_content("https://example.org/"), "x" * 2000)

    def test_external_api_error_does_not_return_the_credentials(self):
        with patch(f"{MODULE}.requests.get", side_effect=requests.ConnectionError("secret-marker")):
            self.manager.wolfram_app_id = "secret-marker"
            output = self.manager.compute("2+2")
        self.assertIn("ConnectionError", output["error"])
        self.assertNotIn("secret-marker", output["error"])


if __name__ == "__main__":
    unittest.main()
