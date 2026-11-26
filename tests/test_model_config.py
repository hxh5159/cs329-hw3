import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from cs329a_hw3 import utils


class ModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        utils._client_for.cache_clear()
        self.addCleanup(utils._client_for.cache_clear)

    def test_together_defaults_and_gateway_credentials_are_separate(self):
        os.environ["TOGETHER_API_KEY"] = "test-together"
        self.assertEqual(utils.get_structured_output_mode(), "json_schema")
        with patch("cs329a_hw3.utils.openai.OpenAI") as sdk:
            utils.get_together_client()
            self.assertEqual(sdk.call_args.kwargs["api_key"], "test-together")
            timeout = sdk.call_args.kwargs["timeout"]
            self.assertEqual(timeout.connect, 15.0)
            self.assertEqual(timeout.read, 240.0)
            self.assertEqual(sdk.call_args.kwargs["max_retries"], 2)
            os.environ["CS329_BASE_URL"] = "https://llmtoken.io/v1/"
            with self.assertRaisesRegex(ValueError, "CS329_API_KEY"):
                utils.get_together_client()
            os.environ["CS329_API_KEY"] = "test-gateway"
            utils.get_together_client()
            self.assertEqual(sdk.call_args.kwargs["api_key"], "test-gateway")
            self.assertEqual(sdk.call_args.kwargs["base_url"], "https://llmtoken.io/v1")

    def test_gateway_uses_explicit_models_and_per_role_override(self):
        os.environ.update(CS329_BASE_URL="https://llmtoken.io/v1", CS329_MODEL="small",
                          CS329_JUDGE_MODEL="reviewer", CS329_CANDIDATE_MODELS="a,b,c")
        self.assertEqual(utils.get_model("generation"), "small")
        self.assertEqual(utils.get_model("judge"), "reviewer")
        self.assertEqual(utils.get_candidate_models(), ["a", "b", "c"])
        self.assertEqual(utils.get_structured_output_mode(), "prompt")
        os.environ["CS329_CANDIDATE_MODELS"] = "a,a,b"
        with self.assertRaises(ValueError):
            utils.get_candidate_models()

    def test_api_failure_logs_http_metadata_without_provider_content(self):
        request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
        error = openai.BadRequestError(
            "secret-provider-detail", response=httpx.Response(400, request=request), body=None
        )
        with patch.object(utils, "get_together_client") as get_client, self.assertLogs(
            utils.__name__, level="WARNING"
        ) as logs:
            get_client.return_value.chat.completions.create.side_effect = error
            result = utils.generate_together("test-model", [{"role": "user", "content": "secret-prompt"}])
        self.assertIsNone(result)
        recorded = "\n".join(logs.output)
        self.assertIn("BadRequestError, HTTP 400", recorded)
        self.assertNotIn("secret", recorded)
        self.assertNotIn("relay.example", recorded)

    def test_empty_chat_text_logs_only_a_known_finish_reason(self):
        message = SimpleNamespace(content=" ", reasoning_content="secret-reasoning")
        for reason, expected in [("length", "length"), ("secret-gateway-detail", "unknown")]:
            with self.subTest(reason=reason), patch.object(utils, "get_together_client") as get_client, (
                self.assertLogs(utils.__name__, level="WARNING")
            ) as logs:
                get_client.return_value.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason=reason)]
                )
                result = utils.generate_together("test-model", [{"role": "user", "content": "question"}])
            self.assertIs(result, message)
            recorded = "\n".join(logs.output)
            self.assertIn(f"finish_reason={expected}", recorded)
            self.assertNotIn("secret", recorded)
