from __future__ import annotations

import unittest

import numpy as np

from app.core.paper_predicate_engine import compute_paper_predicate_sequence
from app.core.rpi_engine import (
    _build_interval_axes,
    _interval_mask,
    compute_recursive_predicates,
)


class RecursivePredicateInductionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        signal = np.arange(120, dtype=float)
        self.x = np.column_stack([signal, signal, rng.normal(size=120)])
        self.selected = np.array([(signal >= 30) & (signal <= 59)])

    def test_returns_multiple_equivalent_high_f1_candidates(self) -> None:
        predicates, qualities, candidates, diagnostics = compute_recursive_predicates(
            self.x,
            self.selected,
            ["signal_a", "signal_b", "noise"],
            max_depth=3,
            max_solutions=20,
            max_states=10_000,
            max_intervals_per_factor=5,
            beam_width=100,
        )

        perfect_candidates = [candidate for candidate in candidates[0] if candidate["quality"]["f1"] == 1.0]
        perfect_attributes = {
            candidate["predicate"][0]["attribute"]
            for candidate in perfect_candidates
            if len(candidate["predicate"]) == 1
        }

        self.assertEqual(qualities[0]["f1"], 1.0)
        self.assertEqual(predicates[0], candidates[0][0]["predicate"])
        self.assertTrue({"signal_a", "signal_b"}.issubset(perfect_attributes))
        self.assertFalse(diagnostics["brushes"][0]["truncated"])

    def test_observed_intervals_use_midpoints_and_preserve_repeated_values(self) -> None:
        values = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3], dtype=float)[:, None]
        axes, unique_values, interval_counts = _build_interval_axes(values, ["value"])
        interval_mask = _interval_mask(axes[0], 1, 2)

        np.testing.assert_array_equal(interval_mask, np.isin(values[:, 0], [1, 2]))
        self.assertEqual(unique_values["value"], 4)
        self.assertEqual(interval_counts["value"], 9)
        self.assertEqual(axes[0].boundaries[1], 0.5)
        self.assertEqual(axes[0].boundaries[3], 2.5)

    def test_exact_interval_recovers_selection_spanning_multiple_old_bins(self) -> None:
        predicates, qualities, _, diagnostics = compute_recursive_predicates(
            self.x[:, [0, 2]],
            self.selected,
            ["signal", "noise"],
            max_depth=2,
            max_states=1_000,
            max_intervals_per_factor=5,
            beam_width=20,
        )

        self.assertEqual(qualities[0]["f1"], 1.0)
        self.assertEqual(qualities[0]["recall"], 1.0)
        self.assertEqual(predicates[0][0]["attribute"], "signal")
        self.assertEqual(diagnostics["interval_generation"], "all contiguous observed-value intervals")

    def test_reports_when_search_budget_truncates_the_tree(self) -> None:
        _, _, _, diagnostics = compute_recursive_predicates(
            self.x,
            self.selected,
            ["signal_a", "signal_b", "noise"],
            max_states=3,
            max_intervals_per_factor=5,
            beam_width=100,
        )

        self.assertTrue(diagnostics["brushes"][0]["truncated"])
        self.assertEqual(diagnostics["brushes"][0]["evaluated_state_count"], 3)

    def test_reports_when_solution_return_limit_is_applied(self) -> None:
        _, _, candidates, diagnostics = compute_recursive_predicates(
            self.x,
            self.selected,
            ["signal_a", "signal_b", "noise"],
            max_solutions=1,
            max_states=10_000,
            max_intervals_per_factor=5,
            beam_width=100,
        )

        self.assertEqual(len(candidates[0]), 1)
        self.assertTrue(diagnostics["brushes"][0]["solutions_limited"])

    def test_reports_exactly_when_recursive_interval_branches_are_limited(self) -> None:
        _, _, _, limited_diagnostics = compute_recursive_predicates(
            self.x,
            self.selected,
            ["signal_a", "signal_b", "noise"],
            max_states=None,
            max_intervals_per_factor=1,
            beam_width=None,
        )
        _, _, _, complete_diagnostics = compute_recursive_predicates(
            self.x[:8, :2],
            np.array([[False, False, True, True, True, False, False, False]]),
            ["signal_a", "signal_b"],
            max_states=None,
            max_intervals_per_factor=None,
            beam_width=None,
        )

        self.assertTrue(limited_diagnostics["brushes"][0]["interval_branch_limited"])
        self.assertFalse(limited_diagnostics["brushes"][0]["search_complete"])
        self.assertFalse(complete_diagnostics["brushes"][0]["interval_branch_limited"])
        self.assertTrue(complete_diagnostics["brushes"][0]["search_complete"])

    def test_returned_multifactor_rules_have_no_f1_redundant_clause(self) -> None:
        _, _, candidates, _ = compute_recursive_predicates(
            self.x,
            self.selected,
            ["signal_a", "signal_b", "noise"],
            max_depth=3,
            max_solutions=50,
            max_states=10_000,
            max_intervals_per_factor=5,
            beam_width=100,
        )

        for candidate in candidates[0]:
            if len(candidate["predicate"]) > 1:
                self.assertTrue(all(clause["importance"] > 0 for clause in candidate["predicate"]))


class PaperPredicateRegressionTests(unittest.TestCase):
    def test_optimizes_paper_objective_and_returns_finite_predicate(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(160, 4))
        selected = np.array(
            [(x[:, 0] > -0.4) & (x[:, 0] < 0.5) & (x[:, 1] > -0.8) & (x[:, 1] < 0.9)]
        )

        predicates, qualities, diagnostics = compute_paper_predicate_sequence(
            x,
            selected,
            ["signal_a", "signal_b", "noise_a", "noise_b"],
            n_iter=200,
            force_cpu=True,
        )

        self.assertLess(diagnostics["final_loss"]["total"], diagnostics["initial_loss"])
        self.assertGreater(qualities[0]["f1"], 0.7)
        self.assertGreater(len(predicates[0]), 0)
        for clause in predicates[0]:
            self.assertTrue(np.isfinite(clause["interval"]).all())
            self.assertLess(clause["interval"][0], clause["interval"][1])

    def test_sequence_loss_contains_both_paper_smoothness_terms(self) -> None:
        rng = np.random.default_rng(19)
        x = rng.normal(size=(100, 3))
        selected = np.array([x[:, 0] < -0.3, x[:, 0] > 0.3])

        _, _, diagnostics = compute_paper_predicate_sequence(
            x,
            selected,
            ["signal", "noise_a", "noise_b"],
            n_iter=30,
            gamma_l1=0.001,
            gamma_a=0.2,
            gamma_mu=0.2,
            force_cpu=True,
        )

        self.assertGreaterEqual(diagnostics["final_loss"]["smooth_a"], 0.0)
        self.assertGreaterEqual(diagnostics["final_loss"]["smooth_mu"], 0.0)
        self.assertEqual(diagnostics["paper_equations"], [5, 6, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
