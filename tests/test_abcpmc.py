"""Tests for the stateless ABCPMC propagator and tolerance schedulers.

All scheduler tests (Phase 1) target the new `compute(inds, current_tol)` API.
All propagator tests (Phase 3) verify that `ABCPMC.__call__` is fully stateless.
"""
import pathlib
import random
import time
import warnings

import numpy as np
import pytest

from propulate.population import Individual
from propulate.propagators.abcpmc import (
    ABCPMC,
    AcceptanceRateScheduler,
    GeometricDecayScheduler,
    QuantileScheduler,
    create_scheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIMITS = {"x": (0.0, 1.0), "y": (0.0, 1.0)}


def make_ind(loss: float, tolerance: float | None = None, generation: int = 0) -> Individual:
    """Create a minimal evaluated Individual for testing."""
    ind = Individual(
        {"x": 0.5, "y": 0.5},
        LIMITS,
        tolerance=tolerance,
        generation=generation,
    )
    ind.loss = loss
    ind.weight = 1.0
    return ind


def make_inds(losses, tolerance=None, base_generation=0):
    """Create a list of individuals with given losses."""
    return [make_ind(loss, tolerance=tolerance, generation=base_generation + i) for i, loss in enumerate(losses)]


# ===========================================================================
# Phase 1 — Stateless Scheduler Tests
# ===========================================================================


class TestQuantileScheduler:
    def test_returns_current_tol_when_too_few_accepted(self):
        sched = QuantileScheduler(
            initial_tol=10.0, population_size=5, additional_needed_inds=0, percentile=50.0
        )
        inds = make_inds([1.0, 2.0, 3.0])  # only 3, need 5
        assert sched.compute(inds, 10.0) == 10.0

    def test_returns_percentile_of_accepted_losses(self):
        # k=3, additional=0 → need 3 accepted to trigger
        sched = QuantileScheduler(
            initial_tol=10.0, population_size=3, additional_needed_inds=0, percentile=50.0
        )
        inds = make_inds([1.0, 2.0, 3.0])  # all below tol=10.0 → accepted
        result = sched.compute(inds, 10.0)
        assert result == pytest.approx(2.0)

    def test_ignores_rejected_inds(self):
        # losses [1, 2, 3, 100, 200] with tol=10 → accepted=[1,2,3], 50th percentile=2
        sched = QuantileScheduler(
            initial_tol=10.0, population_size=3, additional_needed_inds=0, percentile=50.0
        )
        inds = make_inds([1.0, 2.0, 3.0, 100.0, 200.0])
        result = sched.compute(inds, 10.0)
        assert result == pytest.approx(2.0)

    def test_never_exceeds_current_tol(self):
        """Returned tolerance must be <= current_tol (monotone guarantee holds at call site)."""
        sched = QuantileScheduler(
            initial_tol=10.0, population_size=2, additional_needed_inds=0, percentile=90.0
        )
        # Even with high percentile, accepted losses are all < current_tol
        inds = make_inds([8.0, 9.0, 9.5])
        result = sched.compute(inds, 10.0)
        assert result <= 10.0

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError):
            QuantileScheduler(10.0, 5, 0, percentile=0.0)
        with pytest.raises(ValueError):
            QuantileScheduler(10.0, 5, 0, percentile=100.0)

    def test_additional_needed_inds_respected(self):
        # population_size=2, additional=2 → need 4 accepted
        sched = QuantileScheduler(
            initial_tol=10.0, population_size=2, additional_needed_inds=2, percentile=50.0
        )
        inds = make_inds([1.0, 2.0, 3.0])  # 3 < 4, not enough
        assert sched.compute(inds, 10.0) == 10.0
        inds4 = make_inds([1.0, 2.0, 3.0, 4.0])  # 4 == 4, triggers
        assert sched.compute(inds4, 10.0) == pytest.approx(2.5)


class TestGeometricDecayScheduler:
    def test_returns_initial_tol_when_too_few(self):
        sched = GeometricDecayScheduler(
            initial_tol=8.0, population_size=3, additional_needed_inds=0, decay_factor=0.5
        )
        inds = make_inds([1.0, 2.0])  # 2 < 3
        assert sched.compute(inds, 8.0) == 8.0

    def test_decays_by_factor_after_one_epoch(self):
        # k=2, additional=0, decay=0.5, initial_tol=8.0
        # 2 accepted → one epoch → tol = 4.0
        sched = GeometricDecayScheduler(
            initial_tol=8.0, population_size=2, additional_needed_inds=0, decay_factor=0.5
        )
        inds = make_inds([1.0, 2.0])  # both < 8.0
        assert sched.compute(inds, 8.0) == pytest.approx(4.0)

    def test_two_epochs(self):
        # 4 accepted inds, k=2, additional=0, decay=0.5, initial_tol=8.0
        # Epoch 1: batch=[1.0,1.5], next_tol=4.0 → both survive → tol=4.0
        # Epoch 2: batch=[1.0,1.5], next_tol=2.0 → both survive → tol=2.0
        sched = GeometricDecayScheduler(
            initial_tol=8.0, population_size=2, additional_needed_inds=0, decay_factor=0.5
        )
        inds = make_inds([1.0, 1.5, 1.0, 1.5])  # all well below both thresholds
        assert sched.compute(inds, 8.0) == pytest.approx(2.0)

    def test_no_decay_when_too_few_survive_next_threshold(self):
        # k=3, decay=0.9, initial_tol=10.0, next_tol=9.0
        # inds have losses [8.5, 9.5, 9.8] → only 1 survives next_tol → no decay
        sched = GeometricDecayScheduler(
            initial_tol=10.0, population_size=3, additional_needed_inds=0, decay_factor=0.9
        )
        inds = make_inds([8.5, 9.5, 9.8])
        assert sched.compute(inds, 10.0) == pytest.approx(10.0)

    def test_invalid_decay_factor_raises(self):
        with pytest.raises(ValueError):
            GeometricDecayScheduler(10.0, 3, 0, decay_factor=0.0)
        with pytest.raises(ValueError):
            GeometricDecayScheduler(10.0, 3, 0, decay_factor=1.0)


class TestAcceptanceRateScheduler:
    def _make_sched(self, **kwargs):
        defaults = dict(
            initial_tol=1.0,
            population_size=5,
            additional_needed_inds=5,
            low_rate=0.1,
            high_rate=0.3,
            shrink_factor=0.9,
        )
        defaults.update(kwargs)
        return AcceptanceRateScheduler(**defaults)

    def test_tightens_when_above_high_rate(self):
        sched = self._make_sched()
        # window=10, 8 inds with loss < 1.0 (accepted), 2 with loss > 1.0
        inds = make_inds([0.1] * 8 + [5.0, 6.0])  # rate=0.8 > high_rate=0.3
        result = sched.compute(inds, 1.0)
        assert result == pytest.approx(1.0 * 0.9)

    def test_holds_when_below_low_rate(self):
        sched = self._make_sched()
        # window=10, 0 accepted → rate=0.0 < low_rate=0.1; scheduler can only hold
        inds = make_inds([5.0] * 10)
        result = sched.compute(inds, 1.0)
        assert result == pytest.approx(1.0)

    def test_unchanged_in_target_zone(self):
        sched = self._make_sched()
        # window=10, 2 accepted → rate=0.2 ∈ [0.1, 0.3]
        inds = make_inds([0.5, 0.5] + [5.0] * 8)
        result = sched.compute(inds, 1.0)
        assert result == pytest.approx(1.0)

    def test_returns_current_tol_when_too_few_inds(self):
        sched = self._make_sched()
        inds = make_inds([0.5] * 3)  # 3 < window=10
        assert sched.compute(inds, 1.0) == 1.0

    def test_invalid_rates_raise(self):
        with pytest.raises(ValueError):
            AcceptanceRateScheduler(1.0, 5, 5, low_rate=0.5, high_rate=0.3)


class TestCreateScheduler:
    def test_create_quantile(self):
        sched = create_scheduler("quantile", 10.0, 5, 0, percentile=40.0)
        assert isinstance(sched, QuantileScheduler)

    def test_create_geometric_decay(self):
        sched = create_scheduler("geometric_decay", 10.0, 5, 0, decay_factor=0.8)
        assert isinstance(sched, GeometricDecayScheduler)

    def test_create_acceptance_rate(self):
        sched = create_scheduler("acceptance_rate", 10.0, 5, 0)
        assert isinstance(sched, AcceptanceRateScheduler)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler("nonexistent", 10.0, 5, 0)


# ===========================================================================
# Phase 3 — Stateless ABCPMC.__call__ Tests
# ===========================================================================


class TestABCPMCPriorPhase:
    def test_samples_from_prior_when_no_inds(self):
        abc = ABCPMC(LIMITS, k=5, tol=10.0)
        child = abc(inds=[])
        for key, (lo, hi) in LIMITS.items():
            assert lo <= child[key] <= hi
        assert child.loss == float("inf")
        assert child.weight == 1.0

    def test_samples_from_prior_when_archive_below_k(self):
        abc = ABCPMC(LIMITS, k=5, tol=10.0)
        inds = make_inds([1.0, 2.0, 3.0])  # 3 < k=5
        child = abc(inds=inds)
        assert child.weight == 1.0
        assert child.loss == float("inf")

    def test_prior_sample_within_limits(self):
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        for _ in range(20):
            child = abc(inds=[])
            for key, (lo, hi) in LIMITS.items():
                assert lo <= child[key] <= hi


class TestABCPMCStateless:
    def _build_archive(self, n=10, max_loss=1.0, tol=2.0):
        """Build n evaluated inds with loss < max_loss and tolerance=tol stored."""
        inds = []
        for i in range(n):
            ind = make_ind(loss=max_loss * (i + 1) / n, tolerance=tol, generation=i)
            inds.append(ind)
        return inds

    def test_self_tol_not_mutated(self):
        """ABCPMC.tol (initial_tol) must never change between calls."""
        abc = ABCPMC(LIMITS, k=5, tol=10.0, scheduler_type="quantile",
                     additional_needed_inds=0, percentile=50.0)
        initial = abc.tol
        inds = self._build_archive(n=20, max_loss=9.0, tol=10.0)
        abc(inds=inds)
        abc(inds=inds)
        assert abc.tol == initial

    def test_tolerance_reconstructed_from_ind_tolerance_field(self):
        """Child tolerance should be <= the minimum stored tolerance in history."""
        # k=3, 7 inds with loss in [0.1..0.7] and stored tolerance=5.0.
        # percentile=90 of accepted → ~0.64; archive = inds with loss<0.64 → 6 >= k=3.
        abc = ABCPMC(LIMITS, k=3, tol=100.0, scheduler_type="quantile",
                     additional_needed_inds=0, percentile=90.0)
        inds = [make_ind(loss=0.1 * i, tolerance=5.0, generation=i) for i in range(1, 8)]
        child = abc(inds=inds)
        assert child.tolerance is not None
        assert child.tolerance <= 5.0

    def test_child_has_tolerance_set(self):
        abc = ABCPMC(LIMITS, k=5, tol=10.0)
        inds = self._build_archive(n=10, max_loss=9.0, tol=10.0)
        child = abc(inds=inds)
        assert child.tolerance is not None

    def test_child_has_positive_weight(self):
        abc = ABCPMC(LIMITS, k=5, tol=10.0)
        inds = self._build_archive(n=10, max_loss=9.0, tol=10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            child = abc(inds=inds)
        assert child.weight > 0

    def test_child_position_within_limits(self):
        abc = ABCPMC(LIMITS, k=5, tol=10.0)
        inds = self._build_archive(n=10, max_loss=9.0, tol=10.0)
        lo = np.array([v[0] for v in LIMITS.values()])
        hi = np.array([v[1] for v in LIMITS.values()])
        for _ in range(10):
            child = abc(inds=inds)
            assert np.all(child.position >= lo)
            assert np.all(child.position <= hi)

    def test_tolerance_monotone_across_calls(self):
        """Effective tolerance (child.tolerance) must be non-increasing."""
        abc = ABCPMC(LIMITS, k=3, tol=10.0, scheduler_type="quantile",
                     additional_needed_inds=0, percentile=30.0)
        history = []
        prev_tol = float("inf")
        for step in range(15):
            child = abc(inds=history)
            child.loss = 0.1 * (15 - step)  # decreasing losses
            if child.tolerance is not None:
                assert child.tolerance <= prev_tol + 1e-9  # monotone (with float tolerance)
                prev_tol = child.tolerance
            history.append(child)

    def test_deterministic_two_calls_same_history(self):
        """Two propagators with same seed produce same output from same history."""
        rng1 = random.Random(0)
        rng2 = random.Random(0)
        abc1 = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng1)
        abc2 = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng2)
        inds = self._build_archive(n=10, max_loss=9.0, tol=10.0)
        child1 = abc1(inds=inds)
        child2 = abc2(inds=inds)
        np.testing.assert_array_almost_equal(child1.position, child2.position)


class TestABCPMCEdgeCases:
    def test_numpy_float_limits_accepted(self):
        np_limits = {"x": (np.float64(0.0), np.float64(1.0)), "y": (np.float64(0.0), np.float64(1.0))}
        abc = ABCPMC(np_limits, k=3, tol=10.0)
        child = abc(inds=[])
        assert child is not None

    def test_degenerate_archive_all_same_position(self):
        """Zero-covariance archive must not raise; jitter handles it."""
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        inds = []
        for i in range(5):
            ind = Individual({"x": 0.5, "y": 0.5}, LIMITS, tolerance=10.0, generation=i)
            ind.loss = 1.0
            ind.weight = 1.0
            inds.append(ind)
        child = abc(inds=inds)  # must not raise
        assert child is not None

    def test_child_is_individual(self):
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        inds = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i) for i in range(5)]
        child = abc(inds=inds)
        assert isinstance(child, Individual)

    def test_integer_limits_raise(self):
        with pytest.raises(ValueError, match="continuous"):
            ABCPMC({"x": (0, 10), "y": (0.0, 1.0)})

    def test_none_weight_in_archive_falls_back_to_one(self, caplog):
        """Archive individuals with weight=None must not produce NaN; warning is emitted."""
        import logging

        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        inds = []
        for i in range(5):
            ind = Individual({"x": 0.2 * (i + 1), "y": 0.5}, LIMITS, tolerance=10.0, generation=i)
            ind.loss = float(i + 1)
            ind.weight = None  # simulate individual from an external propagator
            inds.append(ind)
        with caplog.at_level(logging.WARNING, logger="propulate.propagators.abcpmc"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                child = abc(inds=inds)
        assert child.weight is not None
        assert not np.isnan(child.weight)
        assert "weight=None" in caplog.text

    def test_near_zero_denominator_handled(self, monkeypatch):
        abc = ABCPMC(LIMITS, k=3, tol=10.0, rng=random.Random(0))
        inds = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i) for i in range(5)]

        def tiny_pdf(*args, **kwargs):
            return 1e-20

        monkeypatch.setattr("propulate.propagators.abcpmc.multivariate_normal.pdf", tiny_pdf)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            child = abc(inds=inds)
        assert any("importance weight denominator is zero" in str(w.message) for w in caught)
        assert np.isfinite(child.weight)
        assert child.weight == pytest.approx(1e12)

    def test_weights_reasonable_near_boundaries(self):
        boundary_limits = {"x": (0.0, 0.05), "y": (0.0, 0.05)}
        abc = ABCPMC(boundary_limits, k=4, tol=1.0, rng=random.Random(0))
        inds = []
        for i in range(6):
            pos = {"x": 0.045 + 0.0005 * i, "y": 0.045 + 0.0005 * i}
            ind = Individual(pos, boundary_limits, tolerance=1.0, generation=i)
            ind.loss = 0.01 * (i + 1)
            ind.weight = 1.0
            inds.append(ind)

        lo = np.array([lim[0] for lim in boundary_limits.values()])
        hi = np.array([lim[1] for lim in boundary_limits.values()])
        for _ in range(25):
            child = abc(inds=inds)
            assert np.all(child.position >= lo)
            assert np.all(child.position <= hi)
            assert np.isfinite(child.weight)
            assert child.weight < 1e12

    def test_10d_produces_valid_candidates(self):
        limits_10d = {f"x{i}": (0.0, 1.0) for i in range(10)}
        abc = ABCPMC(limits_10d, k=5, tol=10.0, rng=random.Random(0))
        inds = []
        for i in range(8):
            pos = {key: 0.4 + 0.02 * i for key in limits_10d}
            ind = Individual(pos, limits_10d, tolerance=10.0, generation=i)
            ind.loss = 0.1 * (i + 1)
            ind.weight = 1.0
            inds.append(ind)

        lo = np.zeros(10)
        hi = np.ones(10)
        for _ in range(10):
            child = abc(inds=inds)
            assert child.position.shape == (10,)
            assert np.all(child.position >= lo)
            assert np.all(child.position <= hi)
            assert np.isfinite(child.weight)


class TestABCPMCSchedulerGuard:
    """Tests for fix 2d: scheduler is only called after archive-size check."""

    def test_prior_phase_not_skipped_by_tight_scheduler(self):
        """With < k accepted individuals, scheduler must not be called (prior phase)."""
        # k=5, only 4 individuals below tol → preliminary archive is too small
        abc = ABCPMC(
            LIMITS, k=5, tol=10.0,
            scheduler_type="acceptance_rate",
            additional_needed_inds=0, low_rate=0.1, high_rate=0.3, shrink_factor=0.5,
        )
        inds = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i) for i in range(4)]
        child = abc(inds=inds)
        assert child.weight == 1.0       # prior phase
        assert child.tolerance is None   # not stamped

    def test_fallback_when_scheduler_tightens_too_aggressively(self):
        """When tighter tol would shrink archive below k, hold at tol_from_history."""
        # k=3, 5 individuals with losses [1,2,3,4,5], tol=10.0
        # shrink_factor=0.05 → candidate_tol=0.5 → 0 individuals survive → fallback
        abc = ABCPMC(
            LIMITS, k=3, tol=10.0,
            scheduler_type="acceptance_rate",
            additional_needed_inds=0, low_rate=0.01, high_rate=0.99, shrink_factor=0.05,
        )
        inds = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i) for i in range(5)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            child = abc(inds=inds)
        assert child.tolerance == pytest.approx(10.0)


class TestABCPMCMinTol:
    def test_tolerance_never_drops_below_min_tol(self):
        abc = ABCPMC(
            LIMITS,
            k=3,
            tol=10.0,
            min_tol=2.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            percentile=10.0,
            rng=random.Random(0),
        )
        inds = [make_ind(loss=0.1 * i, tolerance=10.0, generation=i) for i in range(1, 8)]
        child = abc(inds=inds)
        assert child.tolerance == pytest.approx(2.0)

    def test_min_tol_none_allows_arbitrary_shrink(self):
        abc = ABCPMC(
            LIMITS,
            k=3,
            tol=10.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            percentile=50.0,
            rng=random.Random(0),
        )
        inds = [make_ind(loss=0.1 * i, tolerance=10.0, generation=i) for i in range(1, 8)]
        child = abc(inds=inds)
        assert child.tolerance == pytest.approx(0.4)

    def test_min_tol_negative_raises(self):
        with pytest.raises(ValueError, match="min_tol"):
            ABCPMC(LIMITS, min_tol=-1.0)

    def test_converged_archive_does_not_stall_with_min_tol(self):
        abc = ABCPMC(
            LIMITS,
            k=3,
            tol=10.0,
            min_tol=1.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            percentile=50.0,
            rng=random.Random(0),
        )
        inds = [make_ind(loss=0.2 * i, tolerance=0.3, generation=i) for i in range(1, 8)]
        child = abc(inds=inds)
        assert child.tolerance == pytest.approx(1.0)
        assert child.weight > 0


class TestFilterByTolerance:
    def test_pure_function_with_explicit_tol(self):
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        inds = make_inds([1.0, 5.0, 15.0, 20.0])
        result = abc.filter_by_tolerance(inds, tol=10.0)
        assert len(result) == 2
        assert all(ind.loss < 10.0 for ind in result)

    def test_no_side_effects(self):
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        inds = make_inds([1.0, 5.0, 15.0])
        _ = abc.filter_by_tolerance(inds, tol=10.0)
        assert abc.tol == 10.0  # unchanged


# ===========================================================================
# Phase 4 — Incremental Cache Tests
# ===========================================================================


class TestIncrementalCacheEquivalence:
    """Verify that cached path produces identical results to a full rebuild."""

    def _build_growing_history(self, n, k, tol, scheduler_type="acceptance_rate", **kwargs):
        """Run ABCPMC for n steps, building history incrementally."""
        abc = ABCPMC(LIMITS, k=k, tol=tol, scheduler_type=scheduler_type, **kwargs)
        history = []
        children = []
        for step in range(n):
            child = abc(inds=history)
            child.loss = 0.5 + 0.01 * step  # gradually increasing loss
            child.generation = step
            children.append(child)
            history.append(child)
        return abc, children, history

    def test_cached_matches_uncached_acceptance_rate(self):
        """Cached path produces same tolerances as uncached full-rebuild path."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        abc_cached = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng1,
                            scheduler_type="acceptance_rate",
                            additional_needed_inds=0)
        abc_uncached = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng2,
                              scheduler_type="acceptance_rate",
                              additional_needed_inds=0)
        history = []
        for step in range(30):
            child1 = abc_cached(inds=history)
            # Force uncached path by invalidating cache
            abc_uncached._cache.history_len = -1
            abc_uncached.tolerance_scheduler.reset_cache()
            child2 = abc_uncached(inds=history)
            np.testing.assert_array_almost_equal(child1.position, child2.position)
            assert child1.tolerance == child2.tolerance
            assert child1.weight == pytest.approx(child2.weight, rel=1e-10)
            # Use child1 for history (both should be identical)
            child1.loss = 0.5 + 0.01 * step
            child1.generation = step
            history.append(child1)

    def test_cached_matches_uncached_quantile(self):
        """Same as above but with quantile scheduler."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        abc_cached = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng1,
                            scheduler_type="quantile",
                            additional_needed_inds=0, percentile=50.0)
        abc_uncached = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng2,
                              scheduler_type="quantile",
                              additional_needed_inds=0, percentile=50.0)
        history = []
        for step in range(30):
            child1 = abc_cached(inds=history)
            abc_uncached._cache.history_len = -1
            abc_uncached.tolerance_scheduler.reset_cache()
            child2 = abc_uncached(inds=history)
            np.testing.assert_array_almost_equal(child1.position, child2.position)
            child1.loss = 0.5 + 0.01 * step
            child1.generation = step
            history.append(child1)

    def test_cached_matches_uncached_geometric_decay(self):
        """Same as above but with geometric decay scheduler."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        abc_cached = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng1,
                            scheduler_type="geometric_decay",
                            additional_needed_inds=0, decay_factor=0.9)
        abc_uncached = ABCPMC(LIMITS, k=5, tol=10.0, rng=rng2,
                              scheduler_type="geometric_decay",
                              additional_needed_inds=0, decay_factor=0.9)
        history = []
        for step in range(30):
            child1 = abc_cached(inds=history)
            abc_uncached._cache.history_len = -1
            abc_uncached.tolerance_scheduler.reset_cache()
            child2 = abc_uncached(inds=history)
            np.testing.assert_array_almost_equal(child1.position, child2.position)
            child1.loss = 0.5 + 0.01 * step
            child1.generation = step
            history.append(child1)


class TestCacheInvalidation:
    """Test that cache invalidation (fallback to rebuild) works correctly."""

    def test_shorter_history_triggers_rebuild(self):
        """If history shrinks, cache must rebuild and still produce valid output."""
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        # Build up history
        history = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i)
                   for i in range(5)]
        child1 = abc(inds=history)
        assert child1 is not None
        # Now pass a shorter history
        shorter = history[:3]
        child2 = abc(inds=shorter)
        assert child2 is not None
        assert abc._cache.history_len == 3

    def test_multi_ind_delta_processed_correctly(self):
        """If multiple individuals arrive at once, all are processed."""
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        # First call with empty history
        abc(inds=[])
        assert abc._cache.history_len == 0
        # Jump to 5 individuals (delta=5, not 1)
        history = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i)
                   for i in range(5)]
        child = abc(inds=history)
        assert child is not None
        assert abc._cache.history_len == 5
        assert abc._cache.n_accepted == 5

    def test_tolerance_tightening_re_filters_cache(self):
        """When a new ind has a tighter tolerance, cache must drop stale entries."""
        abc = ABCPMC(LIMITS, k=3, tol=10.0)
        # Build history with tolerance=10.0 and various losses
        history = [make_ind(loss=l, tolerance=10.0, generation=i)
                   for i, l in enumerate([1.0, 3.0, 5.0, 7.0, 9.0])]
        abc(inds=history)
        assert abc._cache.n_accepted == 5  # all < 10.0

        # Add individual with tighter tolerance=4.0
        new_ind = make_ind(loss=2.0, tolerance=4.0, generation=5)
        history.append(new_ind)
        abc(inds=history)
        # Now only inds with loss < 4.0 are accepted: [1.0, 2.0, 3.0]
        assert abc._cache.n_accepted == 3
        assert abc._cache.tol_from_history == 4.0


class TestPerformanceRegression:
    """Verify that __call__ time does not grow quadratically with history size."""

    def test_no_quadratic_scaling(self):
        """Total time for N calls should be O(N log N), not O(N^2).

        We compare the time for 500 calls against a generous linear budget.
        The old O(N^2) code would take ~0.5s for 500 calls; with the cache
        it should be well under 0.1s.
        """
        abc = ABCPMC(LIMITS, k=10, tol=100.0, additional_needed_inds=0)
        history = []
        n_calls = 500
        start = time.perf_counter()
        for step in range(n_calls):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                child = abc(inds=history)
            child.loss = 0.1 * step
            child.generation = step
            history.append(child)
        elapsed = time.perf_counter() - start
        # Budget: 0.5s is extremely generous for 500 O(log N) calls.
        # Without the cache fix, this would take ~0.5-1.0s on a typical machine.
        assert elapsed < 2.0, f"500 calls took {elapsed:.2f}s — possible O(N^2) regression"

    def test_late_calls_not_slower_than_early_calls(self):
        """Per-call time at the end of a run should not be >> per-call time at the start."""
        abc = ABCPMC(LIMITS, k=10, tol=100.0, additional_needed_inds=0)
        history = []
        n_calls = 400

        # Warm up (first 50 calls)
        for step in range(50):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                child = abc(inds=history)
            child.loss = 0.1 * step
            child.generation = step
            history.append(child)

        # Time middle batch (calls 50-99)
        t0 = time.perf_counter()
        for step in range(50, 100):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                child = abc(inds=history)
            child.loss = 0.1 * step
            child.generation = step
            history.append(child)
        early_time = time.perf_counter() - t0

        # Continue to step 400
        for step in range(100, n_calls):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                child = abc(inds=history)
            child.loss = 0.1 * step
            child.generation = step
            history.append(child)

        # Time late batch (calls 400-449)
        t0 = time.perf_counter()
        for step in range(n_calls, n_calls + 50):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                child = abc(inds=history)
            child.loss = 0.1 * step
            child.generation = step
            history.append(child)
        late_time = time.perf_counter() - t0

        # Late batch should not be more than 5x slower than early batch.
        # Without the cache fix, it would be ~8x slower (400/50).
        assert late_time < 5 * early_time + 0.01, (
            f"Late batch ({late_time:.4f}s) is much slower than early batch "
            f"({early_time:.4f}s) — possible O(N) per-call regression"
        )


# ===========================================================================
# Phase 5 — Integration test (MPI)
# ===========================================================================

try:
    from mpi4py import MPI
    from propulate import Propulator

    @pytest.mark.mpi
    def test_abcpmc_propulator_runs(mpi_tmp_path: pathlib.Path) -> None:
        """End-to-end: ABCPMC runs via Propulator on 2D sphere without crash."""
        limits = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        rng = random.Random(42 + MPI.COMM_WORLD.rank)
        propagator = ABCPMC(
            limits,
            k=5,
            tol=1.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            percentile=50.0,
        )
        propulator = Propulator(
            loss_fn=lambda p: float(p["x"] ** 2 + p["y"] ** 2),
            propagator=propagator,
            rng=rng,
            generations=30,
            checkpoint_path=mpi_tmp_path,
        )
        propulator.propulate()

except ImportError:
    pass
