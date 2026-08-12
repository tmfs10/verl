from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_opsd_throughput.py")
SPEC = importlib.util.spec_from_file_location("summarize_opsd_throughput", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SummarizeOpsdThroughputTest(unittest.TestCase):
    def test_parse_step_metrics_keeps_latest_record(self) -> None:
        lines = [
            "(Task pid=1) step:1 - timing_s/step:11.0 - timing_s/gen:4.0 - actor/opsd_distill_weight:1.0",
            "(Task pid=1) step:2 - timing_s/step:9.0 - timing_s/gen:3.0 - timing_s/save_checkpoint:0.0",
        ]
        parsed = MODULE.parse_step_metrics(lines)
        self.assertEqual(parsed[1]["timing_s/step"], 11.0)
        self.assertEqual(parsed[1]["actor/opsd_distill_weight"], 1.0)
        self.assertEqual(parsed[2]["timing_s/gen"], 3.0)

    def test_iteration_capacity_reserves_fixed_checkpoint_safety_and_cold_premium(self) -> None:
        capacity = MODULE.iteration_capacity(
            fixed_overhead=100.0,
            checkpoint=50.0,
            cold_step=130.0,
            steady_step=100.0,
        )
        self.assertEqual(capacity, (14_400 - 100 - 50 - 900 - 30) // 100)

    def test_warmup_extrapolation_reports_post_warmup_count(self) -> None:
        source = MODULE.VariantSummary(
            variant="v3_separate_sft",
            source="measured",
            job_id="123",
            elapsed_seconds=1000.0,
            fixed_overhead_seconds=100.0,
            cold_step_seconds=120.0,
            steady_step_seconds_median=100.0,
            steady_step_seconds_max=110.0,
            checkpoint_seconds=40.0,
            expected_iterations_4h=120,
            conservative_iterations_4h=105,
            expected_post_warmup_iterations_4h=None,
            conservative_post_warmup_iterations_4h=None,
            step_metrics={},
        )
        result = MODULE.extrapolate_warmup(source)
        self.assertEqual(result.expected_iterations_4h, 120)
        self.assertEqual(result.expected_post_warmup_iterations_4h, 90)
        self.assertEqual(result.conservative_post_warmup_iterations_4h, 75)


if __name__ == "__main__":
    unittest.main()
