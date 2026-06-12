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
        # Lower-rank 50th percentile: index = int(0.5 * 4) = 2 → losses_sorted[2] = 3.0.
        assert sched.compute(inds4, 10.0) == pytest.approx(3.0)


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
        """Persistent denominator underflow yields weight=0 (no 1e12 outliers)."""
        abc = ABCPMC(LIMITS, k=3, tol=10.0, rng=random.Random(0))
        inds = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i) for i in range(5)]

        def force_underflow(*args, **kwargs):
            # Return huge Mahalanobis residuals so every retry's log-PDF
            # underflows. After _MAX_WEIGHT_RETRIES the candidate must
            # receive weight=0 rather than a 1e12 floor weight.
            b = args[1]
            return np.full_like(b, 1e8, dtype=float)

        monkeypatch.setattr(
            "propulate.propagators.abcpmc.solve_triangular", force_underflow
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            child = abc(inds=inds)
        assert any("weight=0" in str(w.message) for w in caught)
        assert np.isfinite(child.weight)
        assert child.weight == 0.0

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


class TestW1LogSpaceAMIS:
    """W1.1 — log-space AMIS assembly stability at higher d."""

    def test_d15_gaussian_kernel_finite_weights_no_warning(self):
        """200 archive-phase calls in d=15 Gaussian-kernel mode: all weights finite, no RuntimeWarning."""
        limits = {f"x{i}": (0.0, 1.0) for i in range(15)}
        abc = ABCPMC(
            limits,
            k=10,
            tol=1.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            percentile=50.0,
            kernel="gaussian",
            amis_snapshots=5,
            amis_interval=4,
            rng=random.Random(0),
        )
        history = []
        rng_np = np.random.default_rng(0)
        # bootstrap
        for i in range(10):
            child = abc(history)
            child.loss = float(rng_np.uniform(0.0, 2.0))
            child.generation = i
            history.append(child)
        # 200 archive-phase calls, capturing warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            for i in range(10, 210):
                child = abc(history)
                child.loss = float(rng_np.uniform(0.0, 2.0))
                child.generation = i
                history.append(child)
                assert np.isfinite(child.weight)
                assert child.weight >= 0.0
        amis_warns = [w for w in caught if "importance weight denominator" in str(w.message)]
        assert amis_warns == [], (
            f"Unexpected AMIS underflow warnings in log-space path: "
            f"{[str(w.message) for w in amis_warns]}"
        )


class TestW1BoundaryFallback:
    """W1.2 — uniform-prior-draw fallback when kernel cov exceeds the box."""

    def test_wide_kernel_falls_back_to_uniform_with_weight_one(self):
        """Force kernel cov ≫ box → fallback to uniform-prior draw, weight=1.0."""
        # Tiny box + huge perturbation_scale + widely-scattered archive →
        # kernel cov dominates the box on every retry → fallback path fires.
        tiny_limits = {"x": (0.0, 1e-6), "y": (0.0, 1e-6)}
        abc = ABCPMC(
            tiny_limits,
            k=3,
            tol=10.0,
            perturbation_scale=1e6,
            rng=random.Random(0),
        )
        inds = []
        positions = [(0.0, 0.0), (1e-6, 1e-6), (0.0, 1e-6)]
        for i, (x, y) in enumerate(positions * 2):
            ind = Individual({"x": x, "y": y}, tiny_limits, tolerance=10.0, generation=i)
            ind.loss = 0.1 * (i + 1)
            ind.weight = 1.0
            inds.append(ind)

        lo = np.array([0.0, 0.0])
        hi = np.array([1e-6, 1e-6])
        for _ in range(20):
            child = abc(inds=inds)
            # Position must remain in the box (uniform-prior draw, not clipped).
            assert np.all(child.position >= lo)
            assert np.all(child.position <= hi)
            # Fallback semantics: pi/pi = 1.0.
            assert child.weight == 1.0
            # Tolerance still stamped per paper §3.4.
            assert child.tolerance is not None


class TestW1RetryExhaustion:
    """W1.3 — reject-and-resample-parent retry replaces the 1e12 weight floor."""

    def test_partial_retry_then_success_yields_normal_weight(self, monkeypatch):
        """If the first call underflows but a subsequent retry succeeds, weight is normal (not 0, not 1e12)."""
        from propulate.propagators import abcpmc as mod
        abc = ABCPMC(LIMITS, k=3, tol=10.0, rng=random.Random(0))
        inds = [make_ind(loss=float(i + 1), tolerance=10.0, generation=i) for i in range(5)]

        real_solve = mod.solve_triangular
        call_count = {"n": 0}

        def underflow_then_real(L, b, lower=True):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return np.full_like(b, 1e8, dtype=float)  # underflow once
            return real_solve(L, b, lower=lower)

        monkeypatch.setattr(mod, "solve_triangular", underflow_then_real)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            child = abc(inds=inds)
        assert np.isfinite(child.weight)
        assert 0.0 < child.weight < 1e12  # not 0 (retry succeeded), not floor (no clamp)
        # weight=0 warning must NOT fire on partial-retry success.
        assert not any("weight=0" in str(w.message) for w in caught)


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

    def test_equal_length_content_change_triggers_rebuild(self):
        """Island model (R6): an active set that changes content without
        changing length (one emigrant deactivated + one immigrant appended)
        must not be served from a stale cache. A length-only check would miss
        this; the tail-identity check catches it and rebuilds."""
        abc = ABCPMC(LIMITS, k=3, tol=20.0)
        a, b, c, d = [make_ind(loss=float(i + 1), tolerance=5.0, generation=i)
                      for i in range(4)]
        e = make_ind(loss=2.5, tolerance=2.0, generation=4)  # holds the running-min tol
        history = [a, b, c, d, e]
        abc(inds=history)
        assert abc._cache.history_len == 5
        assert abc._cache.tol_from_history == 2.0  # from e

        # Migration: e emigrates (removed), immigrant f arrives (appended).
        # Length is unchanged (5), but the last element is now f, not e.
        f = make_ind(loss=2.5, tolerance=5.0, generation=5)
        swapped = [a, b, c, d, f]
        abc(inds=swapped)
        assert abc._cache._last_ind is f
        # A stale (no-op) cache would keep tol_from_history == 2.0 (e's value);
        # the running min only ever decreases on the incremental path. A rebuild
        # recomputes it from the new content → 5.0, proving the rebuild fired.
        assert abc._cache.tol_from_history == 5.0

    def test_append_only_does_not_rebuild(self):
        """R6 perf guard: pure-append growth must stay on the incremental path
        (no rebuild), so the island-model fix doesn't regress steady-state cost."""
        abc = ABCPMC(LIMITS, k=3, tol=20.0)
        history = [make_ind(loss=float(i + 1), tolerance=5.0, generation=i)
                   for i in range(4)]
        abc(inds=history)  # first call rebuilds (cached_len < 0)

        sched = abc.tolerance_scheduler
        rebuilds = {"n": 0}
        orig_reset = sched.reset_cache  # reset_cache is called only on the rebuild branch

        def counting_reset():
            rebuilds["n"] += 1
            orig_reset()

        sched.reset_cache = counting_reset

        # Append-only growth: no rebuild expected.
        for i in range(4, 9):
            history.append(make_ind(loss=float(i + 1), tolerance=5.0, generation=i))
            abc(inds=history)
        assert rebuilds["n"] == 0

        # Replace the last element (same length, different object) → one rebuild.
        history[-1] = make_ind(loss=2.5, tolerance=5.0, generation=99)
        abc(inds=history)
        assert rebuilds["n"] == 1


class TestPerformanceRegression:
    """Verify that __call__ time does not grow quadratically with history size."""

    def test_no_quadratic_scaling(self):
        """Total time for N calls should be O(N log N), not O(N^2).

        We compare the time for 500 calls against a generous linear budget.
        The old O(N^2) code would take ~0.5s for 500 calls; with the cache
        it should be well under 0.1s. AMIS is disabled (snapshots=0) so this
        test measures algorithmic-scaling behaviour, not the per-call AMIS
        snapshot-evaluation cost.
        """
        abc = ABCPMC(LIMITS, k=10, tol=100.0, additional_needed_inds=0, amis_snapshots=0)
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
        # Budget: 5.0s catches the old O(N^2) regression (which took 50+ s
        # at this scale). Log-space arithmetic (W1.1) carries a constant
        # overhead of ~3-5x over the old linear-space np.dot path, which is
        # absorbed in the budget. The relative-scaling companion test below
        # is the more meaningful guard against O(N) per-call regressions.
        assert elapsed < 5.0, f"500 calls took {elapsed:.2f}s — possible O(N^2) regression"

    def test_late_calls_not_slower_than_early_calls(self):
        """Per-call time at the end of a run should not be >> per-call time at the start.

        Disables AMIS to isolate per-call algorithmic scaling from snapshot
        evaluation cost (which is O(S·k·d²) per call but does not grow with
        history size).
        """
        abc = ABCPMC(LIMITS, k=10, tol=100.0, additional_needed_inds=0, amis_snapshots=0)
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
# Phase 6 — Smooth-kernel and AMIS reweighting tests
# ===========================================================================


from propulate.propagators.abcpmc import (
    _EpanechnikovKernel,
    _GaussianKernel,
    _HardKernel,
    _make_kernel,
)


class TestKernels:
    """Direct tests of the K_eps(rho) kernel implementations."""

    def test_hard_kernel_indicator(self):
        k = _HardKernel()
        rho = np.array([0.0, 0.5, 1.0, 1.5])
        w = k.weight(rho, 1.0)
        assert np.allclose(w, [1.0, 1.0, 0.0, 0.0])

    def test_gaussian_kernel_values(self):
        k = _GaussianKernel()
        rho = np.array([0.0, 1.0])
        w = k.weight(rho, 1.0)
        # exp(0) = 1, exp(-1/2) ~ 0.6065
        assert np.isclose(w[0], 1.0)
        assert np.isclose(w[1], np.exp(-0.5))

    def test_gaussian_kernel_never_zero_except_eps_zero(self):
        k = _GaussianKernel()
        rho = np.array([100.0])
        # Very large rho but eps > 0: kernel is exponentially small but >= 0.
        w = k.weight(rho, 1.0)
        assert np.all(w >= 0.0)

    def test_epanechnikov_compact_support(self):
        k = _EpanechnikovKernel()
        rho = np.array([0.0, 0.5, 1.0, 1.5])
        w = k.weight(rho, 1.0)
        # 1 - rho^2/eps^2 for rho < eps, else 0.
        assert np.isclose(w[0], 1.0)
        assert np.isclose(w[1], 0.75)
        assert np.isclose(w[2], 0.0)
        assert np.isclose(w[3], 0.0)

    def test_log_weight_consistent_with_weight(self):
        for name in ("hard", "gaussian", "epanechnikov"):
            k = _make_kernel(name)
            rho = np.array([0.1, 0.5, 0.9, 1.2])
            w = k.weight(rho, 1.0)
            lw = k.log_weight(rho, 1.0)
            # For nonzero w, exp(log_weight) == weight.
            nz = w > 0.0
            assert np.allclose(np.exp(lw[nz]), w[nz])
            # Zero w corresponds to -inf log weight.
            assert np.all(np.isneginf(lw[~nz])) if (~nz).any() else True

    def test_make_kernel_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown ABCPMC kernel"):
            _make_kernel("triangular")


class TestABCPMCKernelModes:
    """ABCPMC end-to-end behaviour under each kernel mode."""

    @pytest.mark.parametrize("kernel", ["hard", "gaussian", "epanechnikov"])
    def test_run_produces_finite_weights(self, kernel):
        limits = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        rng = random.Random(7)
        abc = ABCPMC(
            limits=limits,
            k=10,
            tol=1.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            percentile=50.0,
            kernel=kernel,
            rng=rng,
        )
        inds = []
        rng_np = np.random.default_rng(123)
        for i in range(60):
            child = abc(inds)
            child.loss = float(rng_np.uniform(0.0, 2.0))
            child.generation = i
            inds.append(child)
        # After bootstrap we should be in archive phase: every child has a
        # finite, positive weight and (for archive-phase) a stamped tolerance.
        for child in inds[-5:]:
            assert np.isfinite(child.weight)
            assert child.weight > 0.0

    def test_smooth_kernel_bootstrap_rule(self):
        """Smooth kernels exit prior phase once history >= k, regardless of loss."""
        abc = ABCPMC(
            limits=LIMITS,
            k=5,
            tol=0.01,                       # very tight initial bandwidth
            scheduler_type="quantile",
            additional_needed_inds=0,
            kernel="gaussian",
            rng=random.Random(11),
        )
        # Feed 4 individuals with large losses (would be rejected under hard kernel).
        inds = make_inds([10.0, 10.0, 10.0, 10.0])
        # With smooth kernel we still need 5 to exit prior phase.
        child = abc(inds)
        assert child.tolerance is None  # prior-phase child has no stamped tolerance
        # Adding a 5th flips us into archive phase.
        inds.append(make_ind(10.0, generation=4))
        child2 = abc(inds)
        assert child2.tolerance is not None
        assert np.isfinite(child2.weight)

    def test_hard_kernel_preserves_classical_prior_phase(self):
        """Hard kernel keeps the count_below(eps) >= k rule."""
        abc = ABCPMC(
            limits=LIMITS,
            k=5,
            tol=1.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            kernel="hard",
            rng=random.Random(13),
        )
        # 10 inds all above tol => still prior phase (under hard kernel).
        inds = make_inds([5.0] * 10, tolerance=None)
        child = abc(inds)
        # Hard prior-phase child: weight=1.0, no stamped tolerance.
        assert child.weight == 1.0
        assert child.tolerance is None

    def test_epanechnikov_all_outside_support_fallback(self):
        """When every archive particle has K_eps == 0 the algorithm falls back to uniform weights."""
        # Tight bandwidth + archive populated with high-loss particles =>
        # all kernel weights zero. Algorithm should still produce a finite weight.
        abc = ABCPMC(
            limits=LIMITS,
            k=3,
            tol=0.01,
            scheduler_type="quantile",
            additional_needed_inds=0,
            kernel="epanechnikov",
            rng=random.Random(17),
        )
        inds = [make_ind(5.0, tolerance=0.01, generation=i) for i in range(6)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            child = abc(inds)
        assert np.isfinite(child.weight)


class TestAMISBuffer:
    """Streaming AMIS snapshot buffer behaviour."""

    def test_enabled_by_default(self):
        """Default amis_snapshots=20 (paper §3.5 / W2.2): cumulative-mixture
        denominator on for any new ABCPMC construction without an explicit
        opt-out."""
        abc = ABCPMC(limits=LIMITS, k=5, tol=1.0, scheduler_type="quantile", additional_needed_inds=0)
        assert abc._amis_snapshots == 20

    def test_can_opt_out_to_legacy_single_proposal(self):
        abc = ABCPMC(
            limits=LIMITS, k=5, tol=1.0, scheduler_type="quantile",
            additional_needed_inds=0, amis_snapshots=0,
        )
        assert abc._amis_snapshots == 0

    def test_buffer_populates_at_interval(self):
        limits = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        abc = ABCPMC(
            limits=limits,
            k=5,
            tol=1.0,
            scheduler_type="quantile",
            additional_needed_inds=0,
            kernel="gaussian",
            amis_snapshots=4,
            amis_interval=3,
            rng=random.Random(19),
        )
        inds = []
        rng_np = np.random.default_rng(31)
        for i in range(40):
            child = abc(inds)
            child.loss = float(rng_np.uniform(0.0, 2.0))
            child.generation = i
            inds.append(child)
        # We've made many archive-phase calls; buffer should be at its max
        # configured size (newer snapshots evict the oldest).
        assert len(abc._snapshots) == 4

    def test_amis_denominator_differs_from_naive(self):
        """With a non-empty snapshot buffer, the denom uses past proposals too."""
        abc_naive = ABCPMC(
            limits=LIMITS, k=5, tol=1.0, scheduler_type="quantile",
            additional_needed_inds=0, kernel="gaussian",
            amis_snapshots=0, rng=random.Random(23),
        )
        abc_amis = ABCPMC(
            limits=LIMITS, k=5, tol=1.0, scheduler_type="quantile",
            additional_needed_inds=0, kernel="gaussian",
            amis_snapshots=4, amis_interval=2, rng=random.Random(23),
        )
        # Same evaluated history (controlled losses), same seeds — the weight
        # streams diverge once the AMIS buffer kicks in.
        inds_naive, inds_amis = [], []
        rng_np = np.random.default_rng(29)
        common_losses = list(rng_np.uniform(0.0, 2.0, size=40))
        for i, loss in enumerate(common_losses):
            c1 = abc_naive(inds_naive)
            c1.loss = loss
            c1.generation = i
            inds_naive.append(c1)
            c2 = abc_amis(inds_amis)
            c2.loss = loss
            c2.generation = i
            inds_amis.append(c2)
        # By construction the AMIS denom averages over multiple snapshots,
        # so at least one archive-phase weight should differ.
        archive_phase_naive = [i.weight for i in inds_naive if i.tolerance is not None]
        archive_phase_amis = [i.weight for i in inds_amis if i.tolerance is not None]
        assert len(archive_phase_amis) == len(archive_phase_naive)
        assert not np.allclose(archive_phase_naive, archive_phase_amis)


class TestSchedulerKernelAwareFlag:
    """Schedulers accept the kernel_aware flag."""

    def test_quantile_accepts_kernel_aware(self):
        # Without kernel_fn the kernel_aware flag has no effect and the
        # scheduler falls back to the loss-quantile rule.
        sched = QuantileScheduler(10.0, 5, 0, percentile=50.0, kernel_aware=True)
        assert sched.kernel_aware is True
        inds = make_inds([0.1, 0.2, 0.3, 0.4, 0.5])
        eps = sched.compute(inds, 10.0)
        assert isinstance(eps, float)

    def test_factory_passes_kernel_aware(self):
        sched = create_scheduler(
            "geometric_decay", 10.0, 5, 0, kernel_aware=True, decay_factor=0.9
        )
        assert sched.kernel_aware is True


class TestKernelAwareBisection:
    """Target-ESS bisection for smooth-kernel schedulers (W3.1)."""

    def _gaussian_kernel(self):
        from propulate.propagators.abcpmc import _GaussianKernel

        return _GaussianKernel()

    def _epanechnikov_kernel(self):
        from propulate.propagators.abcpmc import _EpanechnikovKernel

        return _EpanechnikovKernel()

    def _converged_archive(self, n: int = 20, scale: float = 0.05):
        """An archive whose losses are tightly clustered near zero."""
        rng_np = np.random.default_rng(0)
        losses = np.abs(rng_np.normal(0.0, scale, size=n))
        return make_inds(list(map(float, losses)))

    def test_no_kernel_fn_falls_back_to_loss_quantile(self):
        """kernel_aware=True but kernel_fn=None → original quantile rule fires."""
        sched = QuantileScheduler(
            initial_tol=10.0,
            population_size=5,
            additional_needed_inds=0,
            percentile=50.0,
            kernel_aware=True,
            kernel_fn=None,
        )
        inds = make_inds([1.0, 2.0, 3.0, 4.0, 5.0])
        eps_aware = sched.compute(inds, 10.0)
        assert eps_aware == pytest.approx(3.0)  # 50th percentile lower-rank

    def test_hard_kernel_falls_back_to_loss_quantile(self):
        """Hard kernel is degenerate for ESS bisection → original rule fires."""
        from propulate.propagators.abcpmc import _HardKernel

        sched = QuantileScheduler(
            initial_tol=10.0,
            population_size=5,
            additional_needed_inds=0,
            percentile=50.0,
            kernel_aware=True,
            kernel_fn=_HardKernel(),
        )
        inds = make_inds([1.0, 2.0, 3.0, 4.0, 5.0])
        eps_aware = sched.compute(inds, 10.0)
        assert eps_aware == pytest.approx(3.0)

    def test_bisection_returns_eps_in_box(self):
        """Bisected ε must lie in [0, current_tol]."""
        sched = QuantileScheduler(
            initial_tol=1.0,
            population_size=5,
            additional_needed_inds=0,
            percentile=50.0,
            kernel_aware=True,
            kernel_fn=self._gaussian_kernel(),
        )
        inds = self._converged_archive(n=20, scale=0.05)
        eps_aware = sched.compute(inds, 1.0)
        assert 0.0 < eps_aware <= 1.0

    def test_bisection_achieves_target_ess(self):
        """Relative ESS at the bisected ε should match the target within tolerance."""
        kfn = self._gaussian_kernel()
        sched = QuantileScheduler(
            initial_tol=1.0,
            population_size=5,
            additional_needed_inds=0,
            percentile=50.0,
            kernel_aware=True,
            kernel_fn=kfn,
            ess_target=0.8,
        )
        inds = self._converged_archive(n=30, scale=0.1)
        eps_aware = sched.compute(inds, 1.0)
        # Manually compute relative ESS at the bisected ε.
        weights = np.ones(len(inds))
        losses = np.array([ind.loss for ind in inds])
        rel_ess = sched._relative_ess(weights, losses, kfn, eps_aware)
        # Bisection tolerance default 1e-4; allow 1e-3 for assertion slack.
        assert abs(rel_ess - 0.8) < 1e-3 or eps_aware == 1.0

    def test_bisection_lower_target_yields_tighter_eps(self):
        """Lower ess_target ⇒ bisection accepts a tighter ε."""
        kfn = self._gaussian_kernel()
        inds = self._converged_archive(n=30, scale=0.1)
        eps_high = QuantileScheduler(
            initial_tol=1.0, population_size=5, additional_needed_inds=0, percentile=50.0,
            kernel_aware=True, kernel_fn=kfn, ess_target=0.95,
        ).compute(inds, 1.0)
        eps_low = QuantileScheduler(
            initial_tol=1.0, population_size=5, additional_needed_inds=0, percentile=50.0,
            kernel_aware=True, kernel_fn=kfn, ess_target=0.5,
        ).compute(inds, 1.0)
        # ess_target=0.5 is more aggressive ⇒ smaller ε.
        assert eps_low < eps_high

    def test_skewed_core_weights_still_tighten(self):
        """Regression (R3): with dispersed core weights the kernel-weighted ESS
        at the loose bandwidth is already far below any absolute floor — exactly
        the condition under which the old absolute-ESS rule stalled (returned
        current_tol unchanged, freezing the bandwidth). The DMDJ successive-ratio
        rule must still produce a strictly tighter ε."""
        kfn = self._gaussian_kernel()
        sched = QuantileScheduler(
            initial_tol=1.0, population_size=5, additional_needed_inds=0,
            percentile=50.0, kernel_aware=True, kernel_fn=kfn, ess_target=0.95,
        )
        rng = np.random.default_rng(0)
        n = 40
        weights = np.ones(n)
        weights[0] = 1000.0  # heavily dispersed core weights → low ESS
        losses = np.abs(rng.normal(0.0, 0.05, size=n))
        current_tol = 1.0
        ess_current = sched._relative_ess(weights, losses, kfn, current_tol)
        assert ess_current < 0.95  # the exact stall condition for the old rule
        eps_new = sched._bisect_target_ess(weights, losses, current_tol)
        assert eps_new < current_tol  # ratio rule still tightens — no stall
        # Tightening never increases the effective sample size.
        assert sched._relative_ess(weights, losses, kfn, eps_new) <= ess_current + 1e-9

    def test_bisection_holds_when_too_few_accepted(self):
        """Not enough accepted inds ⇒ no tightening."""
        sched = QuantileScheduler(
            initial_tol=1.0,
            population_size=10,
            additional_needed_inds=0,
            percentile=50.0,
            kernel_aware=True,
            kernel_fn=self._gaussian_kernel(),
        )
        inds = make_inds([0.1, 0.2, 0.3])  # only 3 < pop=10
        eps_aware = sched.compute(inds, 1.0)
        assert eps_aware == pytest.approx(1.0)

    def test_cached_and_uncached_paths_agree(self):
        """compute_cached and compute must yield the same ε for the same history."""
        kfn = self._gaussian_kernel()
        sched_a = QuantileScheduler(
            initial_tol=1.0, population_size=5, additional_needed_inds=0, percentile=50.0,
            kernel_aware=True, kernel_fn=kfn, ess_target=0.9,
        )
        sched_b = QuantileScheduler(
            initial_tol=1.0, population_size=5, additional_needed_inds=0, percentile=50.0,
            kernel_aware=True, kernel_fn=kfn, ess_target=0.9,
        )
        inds = self._converged_archive(n=30, scale=0.1)
        eps_uncached = sched_a.compute(inds, 1.0)
        # Build the cached views the cached path expects.
        from sortedcontainers import SortedKeyList

        by_loss = SortedKeyList(inds, key=lambda i: i.loss)
        by_gen = SortedKeyList(inds, key=lambda i: i.generation)
        eps_cached = sched_b.compute_cached(inds, 1.0, by_loss, by_gen, by_gen)
        assert eps_cached == pytest.approx(eps_uncached, rel=1e-9)

    def test_bisection_respects_monotone_guarantee_via_propagator(self):
        """End-to-end through ABCPMC: child.tolerance ≤ current_tol always."""
        rng = random.Random(11)
        abc = ABCPMC(
            limits=LIMITS, k=5, tol=1.0,
            scheduler_type="quantile", additional_needed_inds=0, percentile=50.0,
            kernel="gaussian", rng=rng, ess_target=0.7,
        )
        inds = []
        rng_np = np.random.default_rng(42)
        prev = float("inf")
        for i in range(50):
            child = abc(inds)
            child.loss = float(rng_np.uniform(0.0, 1.0))
            child.generation = i
            inds.append(child)
            if child.tolerance is not None:
                assert child.tolerance <= prev + 1e-9
                prev = child.tolerance

    def test_works_with_geometric_decay_and_acceptance_rate(self):
        """Bisection dispatch fires on all three scheduler types."""
        kfn = self._gaussian_kernel()
        inds = self._converged_archive(n=30, scale=0.1)
        for st, extra in (
            ("quantile", {"percentile": 50.0}),
            ("geometric_decay", {"decay_factor": 0.9}),
            ("acceptance_rate", {"low_rate": 0.1, "high_rate": 0.3, "shrink_factor": 0.9}),
        ):
            sched = create_scheduler(
                st, 1.0, 5, 0, kernel_aware=True, kernel_fn=kfn, ess_target=0.9, **extra
            )
            eps_aware = sched.compute(inds, 1.0)
            assert 0.0 < eps_aware <= 1.0

    def test_ess_target_invalid_raises(self):
        with pytest.raises(ValueError, match="ess_target"):
            QuantileScheduler(1.0, 5, 0, ess_target=0.0)
        with pytest.raises(ValueError, match="ess_target"):
            QuantileScheduler(1.0, 5, 0, ess_target=1.5)

    def test_abcpmc_passes_kernel_to_scheduler(self):
        """ABCPMC must hand its kernel and ess_target to the scheduler factory."""
        abc = ABCPMC(
            LIMITS, k=5, tol=1.0,
            scheduler_type="quantile", additional_needed_inds=0, percentile=50.0,
            kernel="gaussian", ess_target=0.7,
        )
        sched = abc.tolerance_scheduler
        assert sched.kernel_fn is abc._kernel_fn
        assert sched.ess_target == pytest.approx(0.7)
        assert sched._use_kernel_aware() is True

    def test_abcpmc_hard_kernel_disables_bisection(self):
        abc = ABCPMC(
            LIMITS, k=5, tol=1.0,
            scheduler_type="quantile", additional_needed_inds=0, percentile=50.0,
            kernel="hard",
        )
        # Hard kernel: kernel_aware=False at the propagator level so the
        # scheduler falls back to its loss-quantile rule.
        assert abc.tolerance_scheduler._use_kernel_aware() is False

    def test_abcpmc_ess_target_invalid_raises(self):
        with pytest.raises(ValueError, match="ess_target"):
            ABCPMC(LIMITS, ess_target=0.0)


class TestTruncationCorrection:
    """Truncated-proposal density correction (R4): per-component in-box mass."""

    def test_log_box_mass_interior_is_near_zero(self):
        from propulate.propagators.abcpmc import _log_box_mass

        # Component at the box centre with tiny sigma: essentially all mass is
        # inside the box, so Z ≈ 1 and log Z ≈ 0.
        positions = np.array([[0.5, 0.5]])
        sigma = np.array([0.01, 0.01])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        lz = _log_box_mass(positions, sigma, lo, hi)
        assert lz.shape == (1,)
        assert lz[0] == pytest.approx(0.0, abs=1e-6)

    def test_log_box_mass_on_boundary_loses_half(self):
        from propulate.propagators.abcpmc import _log_box_mass

        # Mean sits exactly on the lower bound of dim 0 (sigma small relative to
        # the box) → ~half the marginal mass is outside → log Z ≈ log 0.5.
        positions = np.array([[0.0, 0.5]])
        sigma = np.array([0.1, 0.01])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        lz = _log_box_mass(positions, sigma, lo, hi)
        assert lz[0] == pytest.approx(np.log(0.5), abs=1e-3)

    def test_log_box_mass_diagonal_exactness(self):
        from propulate.propagators.abcpmc import _log_box_mass
        from scipy.special import ndtr

        # Diagonal covariance: the product-of-marginals equals the true box mass.
        positions = np.array([[0.3, 0.7]])
        sigma = np.array([0.2, 0.15])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        lz = _log_box_mass(positions, sigma, lo, hi)
        expected = 0.0
        for d in range(2):
            mass = ndtr((hi[d] - positions[0, d]) / sigma[d]) - ndtr(
                (lo[d] - positions[0, d]) / sigma[d]
            )
            expected += np.log(mass)
        assert lz[0] == pytest.approx(expected, abs=1e-12)

    def test_box_mass_precomputed_once_per_call(self, monkeypatch):
        """Hot-path constraint: Z_j is computed once per call (at component-build
        time), not per candidate and not per snapshot. ndtr-call count must be
        independent of the number of AMIS snapshots and reject-resample draws."""
        import propulate.propagators.abcpmc as mod

        abc = ABCPMC(
            LIMITS, k=5, tol=5.0, kernel="gaussian",
            amis_snapshots=5, amis_interval=1, additional_needed_inds=0,
        )
        history = []
        rng_np = np.random.default_rng(0)
        for i in range(20):
            child = abc(history)
            child.loss = float(rng_np.uniform(0.0, 1.0))
            child.generation = i
            history.append(child)
        assert len(abc._snapshots) > 1  # several snapshots accumulated

        calls = {"n": 0}
        orig = mod.ndtr

        def counting_ndtr(x):
            calls["n"] += 1
            return orig(x)

        monkeypatch.setattr(mod, "ndtr", counting_ndtr)
        abc(history)
        # _log_box_mass invokes ndtr exactly twice (z_hi, z_lo) for the single
        # current proposal; snapshots reuse their stored log_box_mass.
        assert calls["n"] == 2


class TestExtractPosterior:
    """Retroactive AMIS posterior estimator (R1) and its statelessness (R2)."""

    LIMITS_1D = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

    def _run_gaussian_mean(self, y, *, seed=0, n=600, k=40, ps=0.8):
        """Drive ABCPMC on a 1D Gaussian-mean ABC problem (loss = |x - y|)."""
        abc = ABCPMC(
            self.LIMITS_1D, k=k, tol=1.0, scheduler_type="acceptance_rate",
            kernel="gaussian", additional_needed_inds=0,
            perturbation_scale=ps, rng=random.Random(seed),
        )
        history = []
        for i in range(n):
            child = abc(history)
            child.loss = abs(child.position[0] - y)
            child.generation = i
            history.append(child)
        return abc, history

    def test_weights_normalized(self):
        abc, history = self._run_gaussian_mean(0.6)
        pos, w = abc.extract_posterior(history)
        assert pos.shape == (len(history), 2)
        assert w.shape == (len(history),)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0.0)

    def test_recovers_gaussian_mean(self):
        abc, history = self._run_gaussian_mean(0.6)
        pos, w = abc.extract_posterior(history)
        mean = np.average(pos, axis=0, weights=w)
        assert mean[0] == pytest.approx(0.6, abs=0.05)

    def test_crash_restart_invariance(self):
        """Estimator is a pure function of history: a fresh propagator with an
        empty snapshot buffer (as after a crash/restart) yields a bit-identical
        posterior. This is the statelessness / crash-recoverability claim (R2)."""
        abc, history = self._run_gaussian_mean(0.6, seed=0)
        pos1, w1 = abc.extract_posterior(history)
        fresh = ABCPMC(
            self.LIMITS_1D, k=40, tol=1.0, scheduler_type="acceptance_rate",
            kernel="gaussian", additional_needed_inds=0, rng=random.Random(999),
        )
        assert len(fresh._snapshots) == 0  # empty buffer, as after a restart
        pos2, w2 = fresh.extract_posterior(history)
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(w1, w2)

    def test_reweighting_is_retroactive(self):
        """Defining AMIS property: a particle's weight depends on the WHOLE
        history (the current cumulative mixture), not only proposals up to its
        arrival. With eps_final fixed, the weight ratio of two fixed particles
        changes as later history is appended — impossible for frozen forward
        weights."""
        abc, history = self._run_gaussian_mean(0.6, seed=0, n=800)
        eps = 0.05
        half = history[:400]
        order = sorted(range(400), key=lambda idx: half[idx].loss)
        i, j = order[0], order[1]
        _, w_half = abc.extract_posterior(half, eps_final=eps)
        _, w_full = abc.extract_posterior(history, eps_final=eps)
        ratio_half = w_half[i] / w_half[j]
        ratio_full = w_full[i] / w_full[j]
        assert not np.isclose(ratio_half, ratio_full, rtol=1e-6)

    def test_bootstrap_only_history_returns_uniform(self):
        """Before the archive phase (history < k) every draw is a uniform prior
        draw → equal weights."""
        abc = ABCPMC(self.LIMITS_1D, k=50, tol=1.0, kernel="gaussian")
        history = [make_ind(loss=1.0, generation=i) for i in range(10)]  # < k
        pos, w = abc.extract_posterior(history)
        assert np.allclose(w, 1.0 / len(history))

    def test_hard_kernel_extraction_runs(self):
        """Hard-kernel extraction produces a valid normalised posterior."""
        abc = ABCPMC(
            self.LIMITS_1D, k=20, tol=1.0, scheduler_type="quantile",
            kernel="hard", additional_needed_inds=0, percentile=50.0,
            rng=random.Random(3),
        )
        history = []
        for i in range(400):
            child = abc(history)
            child.loss = abs(child.position[0] - 0.4)
            child.generation = i
            history.append(child)
        _, w = abc.extract_posterior(history)
        assert w.sum() == pytest.approx(1.0)

    def test_truncation_reduces_near_boundary_weight(self, monkeypatch):
        """R4 end-to-end: with the box-mass correction active, near-boundary
        particles carry less of the posterior weight than under the untruncated
        density (which over-counts their proposal probability). The effect on the
        posterior *mean* is negligible for this self-tightening archive, so the
        regression targets the weight mechanism directly."""
        import propulate.propagators.abcpmc as mod

        abc, history = self._run_gaussian_mean(0.05, seed=0, n=800, k=40, ps=2.0)
        pos = np.array([ind.position for ind in history])
        near = pos[:, 0] < 0.12  # near the lower boundary
        _, w_corrected = abc.extract_posterior(history, eps_final=0.3)
        monkeypatch.setattr(mod, "_log_box_mass", lambda p, s, lo, hi: np.zeros(len(p)))
        _, w_uncorrected = abc.extract_posterior(history, eps_final=0.3)
        assert w_corrected[near].sum() < w_uncorrected[near].sum()


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
