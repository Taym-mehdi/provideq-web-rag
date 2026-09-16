from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.run_fusion_sweep import (
    CONFIGS,
    DEFAULT_HYDE_CACHE,
    DEFAULT_LLM_EXPANSION_CACHE,
    main,
)


EXPECTED_NAMES = (
    "01_fusion_paperclip_vector_raw_europepmc_raw_no_synonyms",
    "02_fusion_paperclip_vector_anchored_hyde_europepmc_raw_no_synonyms",
    "03_fusion_paperclip_vector_anchored_llm_expansion_europepmc_raw_no_synonyms",
)


def _value(arguments: tuple[str, ...] | list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


class FusionSweepTests(unittest.TestCase):
    def test_sweep_has_the_three_focused_configurations(self) -> None:
        self.assertEqual(tuple(config.name for config in CONFIGS), EXPECTED_NAMES)
        self.assertEqual(
            {config.query_strategy for config in CONFIGS},
            {"raw", "hyde", "llmexpand"},
        )

    def test_every_configuration_uses_the_fixed_source_settings(self) -> None:
        for config in CONFIGS:
            arguments = config.arguments
            with self.subTest(config=config.name):
                self.assertEqual(_value(arguments, "--retriever"), "fusion")
                self.assertEqual(_value(arguments, "--paperclip-ranking"), "vector")
                self.assertEqual(_value(arguments, "--europepmc-mode"), "multi")
                self.assertIn("--no-europepmc-synonym", arguments)
                self.assertIn(
                    "--no-europepmc-use-reformulated-query",
                    arguments,
                )

    def test_generated_queries_are_anchored_only_inside_paperclip(self) -> None:
        for config in CONFIGS:
            arguments = config.arguments
            with self.subTest(config=config.name):
                if config.query_strategy == "raw":
                    self.assertNotIn("--paperclip-query-fusion", arguments)
                else:
                    self.assertIn("--paperclip-query-fusion", arguments)
                    self.assertEqual(_value(arguments, "--query-fusion-rrf-k"), "10")
                    self.assertEqual(
                        _value(arguments, "--reformulated-query-weight"),
                        "0.5",
                    )

    @patch("evaluation.run_fusion_sweep.subprocess.run")
    @patch("builtins.print")
    def test_missing_approved_cache_stops_before_retrieval(
        self,
        _print,
        run_subprocess,
    ) -> None:
        with patch(
            "sys.argv",
            [
                "run_fusion_sweep",
                "--hyde-cache", "missing-hyde-cache.json",
            ],
        ), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main()

        run_subprocess.assert_not_called()

    @patch("evaluation.run_fusion_sweep.subprocess.run")
    @patch("builtins.print")
    def test_runner_writes_three_runs_and_one_summary(
        self,
        _print,
        run_subprocess,
    ) -> None:
        run_subprocess.return_value = SimpleNamespace(returncode=0)
        output_dir = "outputs/fusion_test"

        with patch(
            "sys.argv",
            [
                "run_fusion_sweep",
                "--num-questions", "1",
                "--output-dir", output_dir,
                "--retry-errors",
                "--retry-warnings",
            ],
        ):
            self.assertEqual(main(), 0)

        self.assertEqual(run_subprocess.call_count, 4)
        retrieval_commands = [
            call.args[0] for call in run_subprocess.call_args_list[:3]
        ]
        self.assertEqual(
            [command[command.index("--run-name") + 1] for command in retrieval_commands],
            list(EXPECTED_NAMES),
        )
        self.assertNotIn("--llm-cache", retrieval_commands[0])
        self.assertEqual(
            _value(retrieval_commands[1], "--llm-cache"),
            str(DEFAULT_HYDE_CACHE),
        )
        self.assertEqual(
            _value(retrieval_commands[2], "--llm-cache"),
            str(DEFAULT_LLM_EXPANSION_CACHE),
        )
        for command in retrieval_commands:
            self.assertIn("--retry-errors", command)
            self.assertIn("--retry-warnings", command)

        summary_command = run_subprocess.call_args_list[-1].args[0]
        self.assertEqual(
            summary_command[-4:],
            [
                "--input-dir", str(Path(output_dir)),
                "--output", str(Path(output_dir) / "summary.csv"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
