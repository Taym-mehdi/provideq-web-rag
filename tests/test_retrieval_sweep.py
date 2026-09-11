from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.run_retrieval_sweep import CONFIGS, main


EXPECTED_NAMES = (
    "01_paperclip_bm25_raw",
    "02_paperclip_bm25_anchored_hyde",
    "03_paperclip_bm25_anchored_llm_expansion",
    "04_paperclip_vector_raw",
    "05_paperclip_vector_anchored_hyde",
    "06_paperclip_vector_anchored_llm_expansion",
    "07_paperclip_hybrid_raw",
    "08_paperclip_hybrid_anchored_hyde",
    "09_paperclip_hybrid_anchored_llm_expansion",
    "10_europepmc_synonyms_raw",
    "11_europepmc_synonyms_anchored_hyde",
    "12_europepmc_synonyms_anchored_llm_expansion",
    "13_europepmc_no_synonyms_raw",
    "14_europepmc_no_synonyms_anchored_hyde",
    "15_europepmc_no_synonyms_anchored_llm_expansion",
)


def _value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


class RetrievalSweepTests(unittest.TestCase):
    def test_sweep_has_the_exact_fifteen_named_configurations(self) -> None:
        self.assertEqual(tuple(config.name for config in CONFIGS), EXPECTED_NAMES)
        self.assertEqual(len({config.name for config in CONFIGS}), 15)

    def test_each_retrieval_setting_has_all_three_query_strategies(self) -> None:
        grouped: dict[tuple[str, str], set[str]] = {}
        for config in CONFIGS:
            arguments = config.arguments
            retriever = _value(arguments, "--retriever")
            if retriever == "paperclip":
                setting = _value(arguments, "--paperclip-ranking")
            else:
                setting = (
                    "synonyms"
                    if "--europepmc-synonym" in arguments
                    else "no_synonyms"
                )
            grouped.setdefault((retriever, setting), set()).add(
                _value(arguments, "--query-strategy")
            )

        self.assertEqual(
            grouped,
            {
                ("paperclip", "bm25"): {"raw", "hyde", "llmexpand"},
                ("paperclip", "vector"): {"raw", "hyde", "llmexpand"},
                ("paperclip", "hybrid"): {"raw", "hyde", "llmexpand"},
                ("europepmc", "synonyms"): {"raw", "hyde", "llmexpand"},
                ("europepmc", "no_synonyms"): {"raw", "hyde", "llmexpand"},
            },
        )

    def test_non_raw_queries_are_anchored_for_each_retriever(self) -> None:
        for config in CONFIGS:
            arguments = config.arguments
            strategy = _value(arguments, "--query-strategy")
            retriever = _value(arguments, "--retriever")
            with self.subTest(config=config.name):
                if retriever == "paperclip":
                    self.assertEqual(
                        "--paperclip-query-fusion" in arguments,
                        strategy != "raw",
                    )
                else:
                    self.assertIn("--europepmc-mode", arguments)
                    self.assertEqual(_value(arguments, "--europepmc-mode"), "multi")
                    self.assertEqual(
                        "--europepmc-use-reformulated-query" in arguments,
                        strategy != "raw",
                    )

    @patch("evaluation.run_retrieval_sweep.subprocess.run")
    @patch("builtins.print")
    def test_runner_writes_numbered_runs_and_one_summary(
        self,
        _print,
        run_subprocess,
    ) -> None:
        run_subprocess.return_value = SimpleNamespace(returncode=0)
        output_dir = "outputs/retrieval_test"

        with patch(
            "sys.argv",
            [
                "run_retrieval_sweep",
                "--num-questions", "1",
                "--output-dir", output_dir,
            ],
        ):
            self.assertEqual(main(), 0)

        self.assertEqual(run_subprocess.call_count, 16)
        retrieval_commands = [
            call.args[0] for call in run_subprocess.call_args_list[:15]
        ]
        self.assertEqual(
            [command[command.index("--run-name") + 1] for command in retrieval_commands],
            list(EXPECTED_NAMES),
        )
        summary_command = run_subprocess.call_args_list[-1].args[0]
        self.assertEqual(
            summary_command[-4:],
            [
                "--input-dir", output_dir,
                "--output", f"{output_dir}/summary.csv",
            ],
        )


if __name__ == "__main__":
    unittest.main()
