"""Offline regression checks. Stubbed tool and model responses do not measure QA accuracy."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from cs329a_hw3.api_manager import APIManager
from cs329a_hw3.DeepResearchAgent import DeepResearchAgent
from cs329a_hw3.evaluation import evaluate_qa
from cs329a_hw3.multi_lm_agent import MultiLMAgent


def observation(result, tool="get_stock_data"):
    return {"api_used": tool, "params": {}, "results": result, "error": None}


def failed_observation(tool="get_stock_data"):
    return {**observation(None, tool), "error": "Tool unavailable"}


class AnswerWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tools = Mock(spec=APIManager)
        self.agent = MultiLMAgent(self.tools, "planner", "refiner", "fuser")

    def test_single_call_supplies_tool_observation_to_model(self):
        query = "What was the closing price?"
        self.tools.route_query.return_value = observation({"close": 388.76})
        with patch.object(self.agent, "generate", return_value="$388.76") as generate:
            self.assertEqual(self.agent.single_LM_with_single_API_call(query), "$388.76")

        self.tools.route_query.assert_called_once_with(query)
        self.assertIn(query, generate.call_args.args[0])
        self.assertIn("388.76", generate.call_args.args[0])

    def test_decompose_caps_deduplicates_and_keeps_a_failed_observation(self):
        tagged_plan = (
            "<sub-query>Chicago wind</sub-query><sub-query>CHICAGO WIND</sub-query>"
            "<sub-query>Boston wind</sub-query><sub-query>Ignore beyond cap</sub-query>"
        )
        results = {
            "Chicago wind": observation({"wind": 6}, "get_weather"),
            "Boston wind": {"api_used": "get_weather", "params": {}, "results": None, "error": "Unavailable"},
        }
        self.tools.route_query.side_effect = results.__getitem__
        with patch.object(self.agent, "generate", return_value=tagged_plan):
            observations = self.agent.decompose_query("Compare wind speeds", max_sub_queries=2)

        self.assertEqual([entry["sub_query"] for entry in observations], list(results))
        self.assertEqual(observations[0]["results"], {"wind": 6})
        self.assertEqual(observations[1]["error"], "Unavailable")
        self.tools.route_query.assert_has_calls([call("Chicago wind"), call("Boston wind")], any_order=True)
        self.assertEqual(self.tools.route_query.call_count, 2)

    def test_fusion_still_sees_evidence_when_one_candidate_fails(self):
        evidence = [{**observation({"close": 388.76}), "sub_query": "Look up close"}]
        candidates = {"model-a": "Answer from A", "model-c": "Answer from C"}

        def generate(prompt, *, model, temperature=None):
            if model == "model-b":
                raise RuntimeError("Endpoint unavailable")
            if model == "fuser":
                self.assertIn("388.76", prompt)
                for response in candidates.values():
                    self.assertIn(response, prompt)
                return "Fused answer"
            return candidates[model]

        with (
            patch.object(self.agent, "decompose_query", return_value=evidence),
            patch.object(self.agent, "generate", side_effect=generate) as generation,
            patch("cs329a_hw3.multi_lm_agent.get_candidate_models", return_value=["model-a", "model-b", "model-c"]),
            self.assertLogs("cs329a_hw3.multi_lm_agent", level="WARNING"),
        ):
            self.assertEqual(self.agent.decompose_and_fuse("What was the closing price?"), "Fused answer")

        self.assertCountEqual(
            [request.kwargs["model"] for request in generation.call_args_list],
            ["model-a", "model-b", "model-c", "fuser"],
        )

    def test_refinement_carries_dependencies_and_stops_at_budget(self):
        self.tools.route_query.side_effect = [
            observation({"close": 388.76}),
            observation({"result": 432.30}, "compute"),
        ]
        price_query = "Look up the closing price"
        compute_query = "Calculate 388.76 times 1.112"
        with patch.object(self.agent, "generate", side_effect=[
            f"<sub-query>{price_query}</sub-query>",
            f"<sub-query>{compute_query}</sub-query>",
            "$432.30",
        ]) as generate:
            self.assertEqual(self.agent.iterative_refine("Increase the close by 11.2%", max_iterations=2), "$432.30")

        self.assertEqual(self.tools.route_query.call_args_list, [call(price_query), call(compute_query)])
        self.assertIn("388.76", generate.call_args_list[1].args[0])
        self.assertIn("432.3", generate.call_args_list[2].args[0])
        self.assertEqual(generate.call_count, 3)

    def test_pipeline_single_preserves_a_successful_tool_result(self):
        self.tools.route_query.return_value = observation({"close": 42})
        with patch.object(self.agent, "generate", side_effect=["single", "42 dollars"]) as generate:
            answer = self.agent.run_pipeline("What was the closing price?")
        self.assertEqual(answer, "42 dollars")
        self.tools.route_query.assert_called_once_with("What was the closing price?", include_daily_weather=True)
        self.assertIn('"close": 42', generate.call_args.args[0])

    def test_pipeline_retries_a_failed_single_lookup_once(self):
        self.tools.route_query.side_effect = [
            failed_observation("get_weather"),
            observation({"wind_speed": 8}, "get_weather"),
        ]
        query = "What was the wind speed?"
        retry = "Look up wind speed for the specified location and hour"
        with patch.object(self.agent, "generate", side_effect=[
            "single", f"<sub-query>{retry}</sub-query>", "8 km/h"
        ]) as generate:
            answer = self.agent.run_pipeline(query)
        self.assertEqual(answer, "8 km/h")
        self.assertEqual(self.tools.route_query.call_args_list, [
            call(query, include_daily_weather=True), call(retry, include_daily_weather=True)
        ])
        self.assertIn("Tool unavailable", generate.call_args_list[1].args[0])
        self.assertIn('"wind_speed": 8', generate.call_args_list[2].args[0])

    def test_pipeline_marks_model_reasoning_after_both_single_lookups_fail(self):
        self.tools.route_query.return_value = failed_observation("compute")
        query = "What is 17 plus 25?"
        retry = "Calculate the sum of 17 and 25"
        with patch.object(self.agent, "generate", side_effect=[
            "single", f"<sub-query>{retry}</sub-query>", "42"
        ]) as generate:
            answer = self.agent.run_pipeline(query)
        self.assertIn("42\n\n[Model-derived response.", answer)
        self.assertEqual(self.tools.route_query.call_args_list, [
            call(query, include_daily_weather=True), call(retry, include_daily_weather=True)
        ])
        self.assertIn("unavailable external evidence", generate.call_args.args[0])

    def test_pipeline_parallel_preserves_partial_tool_evidence(self):
        evidence = [observation({"close": 42}), failed_observation("compute")]
        with (
            patch.object(self.agent, "generate", return_value="parallel") as generate,
            patch.object(self.agent, "decompose_query", return_value=evidence) as decompose,
            patch.object(self.agent, "_fuse_observations", return_value="Evidence-based") as fuse,
        ):
            self.assertEqual(self.agent.run_pipeline("Original question"), "Evidence-based")
        decompose.assert_called_once_with("Original question", for_pipeline=True)
        fuse.assert_called_once_with("Original question", evidence)
        self.assertEqual(generate.call_count, 1)

    def test_pipeline_sequential_marks_model_answer_after_tool_failure(self):
        self.tools.route_query.return_value = failed_observation("get_stock_data")
        retry = "Look up closing price"
        with patch.object(self.agent, "generate", side_effect=[
            "Choose another workflow", f"<sub-query>{retry}</sub-query>",
            "<final_answer>Unverified quote</final_answer>", "The closing price is unavailable."
        ]):
            answer = self.agent.run_pipeline("What was the closing price?")
        self.tools.route_query.assert_called_once_with(retry, include_daily_weather=True)
        self.assertNotIn("Unverified quote", answer)
        self.assertIn("The closing price is unavailable.", answer)
        self.assertIn("Tool verification unavailable", answer)

    def test_part3_price_recovery_keeps_the_first_error_and_the_actual_price_source(self):
        query = "What was the daily closing price?"
        self.tools.route_query.return_value = {
            **failed_observation(), "params": {"ticker": "MSFT", "date": "2024-05-01"}
        }
        bar = {"ticker": "MSFT", "close": 400, "currency": "USD", "source": "Yahoo Finance Chart"}
        with patch("cs329a_hw3.multi_lm_agent.get_us_daily_bar", return_value=bar) as fetch:
            result = self.agent._pipeline_observation(query)
        fetch.assert_called_once_with("MSFT", "2024-05-01")
        self.assertIsNone(result["error"])
        self.assertEqual(result["results"]["initial_lookup_error"], "Tool unavailable")
        self.assertEqual(result["results"]["source"], "Yahoo Finance Chart")

    def test_single_tool_benchmark_retains_the_original_failure_behavior(self):
        self.tools.route_query.return_value = failed_observation("compute")
        query = "What is 17 plus 25?"
        with patch.object(self.agent, "generate", return_value="No computation result.") as generate:
            answer = self.agent.single_LM_with_single_API_call(query)
        self.assertEqual(answer, "No computation result.")
        self.tools.route_query.assert_called_once_with(query)
        self.assertEqual(generate.call_count, 1)
        self.assertIn("an error is not evidence", generate.call_args.args[0])

    @patch("cs329a_hw3.evaluation.generate_together")
    def test_unavailable_judge_does_not_count_as_wrong(self, generate):
        generate.return_value = None
        with self.assertRaises(RuntimeError):
            evaluate_qa("Question", "Response", "Reference", model="judge")


class ResearchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tools = Mock(spec=APIManager)
        self.agent = DeepResearchAgent(self.tools)

    @patch("cs329a_hw3.DeepResearchAgent.generate_together")
    def test_corrects_citations_without_repeating_searches(self, generate):
        first = "https://first.example/report"
        second = "https://second.example/update"
        self.tools.google_search.side_effect = [
            [{"link": first, "title": "First", "snippet": "March 2024 milestone"}],
            [{"link": second, "title": "Second", "content": "October 2024 update"}],
        ]
        replies = [
            {"search_queries": ["March 2024 milestone", "October 2024 update"]},
            {"paragraphs": ["Overview [[S1]]", "March [[S1]]", "October [[S99]]", "Limits"]},
            {"paragraphs": ["Overview [[S1]]", "March [[S1]]", "October [[S2]]", "Limits [[S2]]"]},
        ]
        generate.side_effect = [SimpleNamespace(content=json.dumps(reply)) for reply in replies]

        result = self.agent.research("What changed in 2024?")

        self.assertEqual(self.tools.google_search.call_args_list, [
            call(search_query="March 2024 milestone", num_results=3),
            call(search_query="October 2024 update", num_results=3),
        ])
        self.assertEqual(generate.call_count, 3)
        first_generation, correction = generate.call_args_list[-2:]
        first_evidence = first_generation.kwargs["messages"][1]["content"]
        corrected_evidence = correction.kwargs["messages"][1]["content"]
        self.assertEqual(first_evidence, corrected_evidence)
        self.assertEqual(result["sources"], [first, second])
        self.assertEqual(len(result["report"].split("\n\n")), 4)
        self.assertIn(f"[S1]({first})", result["report"])
        self.assertIn(f"[S2]({second})", result["report"])

    @patch("cs329a_hw3.DeepResearchAgent.generate_together")
    def test_failed_synthesis_returns_only_real_source_links(self, generate):
        trusted = "https://study.example/paper"
        self.tools.google_search.return_value = [{
            "link": trusted, "title": "Paper [untrusted](https://outside.example)",
            "snippet": "Reported results. <img src=https://outside.example/logo.png>",
        }]
        generate.side_effect = [
            SimpleNamespace(content='{"search_queries":["study"]}'),
            SimpleNamespace(content='{"paragraphs": ["incomplete"]}'),
            SimpleNamespace(content='{"paragraphs": ["still incomplete"]}'),
        ]

        result = self.agent.research("What did the study report?")

        self.assertEqual(result["sources"], [trusted])
        self.assertIn("Reported results.", result["report"])
        self.assertNotIn("https://outside.example", result["report"])
        self.tools.google_search.assert_called_once_with(search_query="study", num_results=3)


if __name__ == "__main__":
    unittest.main()
