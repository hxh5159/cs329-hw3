"""Offline regression checks. Stubbed tool and model responses do not measure QA accuracy."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock, call, patch

from cs329a_hw3.api_manager import APIManager
from cs329a_hw3.DeepResearchAgent import DeepResearchAgent
from cs329a_hw3.evaluation import evaluate_qa
from cs329a_hw3.multi_lm_agent import MultiLMAgent


def observation(result, tool="get_stock_data"):
    return {"api_used": tool, "params": {}, "results": result, "error": None}


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

    def test_planner_dispatches_only_one_supported_workflow(self):
        cases = [
            ("single", "single_LM_with_single_API_call", {}),
            ("parallel", "decompose_and_fuse", {}),
            ("sequential", "iterative_refine", {"max_iterations": 4}),
            ("I recommend single or parallel.", "iterative_refine", {"max_iterations": 4}),
        ]
        methods = {name: DEFAULT for _, name, _ in cases}
        for response, chosen, kwargs in cases:
            with (
                self.subTest(response=response),
                patch.object(self.agent, "generate", return_value=response),
                patch.multiple(self.agent, **methods) as flows,
            ):
                flows[chosen].return_value = "Selected"
                self.assertEqual(self.agent.run_pipeline("Original question"), "Selected")
                flows[chosen].assert_called_once_with("Original question", **kwargs)
                for name in methods.keys() - {chosen}:
                    flows[name].assert_not_called()

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
