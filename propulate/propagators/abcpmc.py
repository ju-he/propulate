import logging
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.linalg import solve_triangular
from sortedcontainers import SortedKeyList

from ..population import Individual
from .base import Propagator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ABC likelihood kernels
# ---------------------------------------------------------------------------
#
# The kernel K_eps(rho) turns the discrepancy rho between simulated and observed
# data into a smooth (or hard) likelihood approximation for the ABC posterior
#
#     pi_eps(theta) ~ pi(theta) * E[K_eps(rho)].
#
# Hard kernel reproduces the classical ABC rejection rule; Gaussian and
# Epanechnikov are smooth variants (Wilkinson 2013, Sisson-Fan-Beaumont 2018).
# Smooth kernels remove the prior-vs-archive phase discontinuity and unlock the
# AMIS / smooth-ABC consistency theory used by the streaming reweighting below.


class _Kernel(ABC):
    """ABC likelihood kernel K_eps(rho)."""

    name: str = ""

    @abstractmethod
    def log_weight(self, rho: np.ndarray, eps: float) -> np.ndarray:
        """Return log K_eps(rho), elementwise. May contain -inf entries."""

    def weight(self, rho: np.ndarray, eps: float) -> np.ndarray:
        """Return K_eps(rho), elementwise."""
        return np.exp(self.log_weight(rho, eps))

    def support_radius(self, eps: float) -> float:
        """
        Loss radius beyond which K_eps(rho) is treated as zero.

        Used to size the candidate pool for archive selection. For compactly
        supported kernels this is the true support boundary; for the Gaussian
        kernel it is the radius beyond which the weight is below ``1e-12``.
        """
        return float("inf")


class _HardKernel(_Kernel):
    """K_eps(rho) = 1 if rho < eps else 0 — the classical ABC indicator."""

    name = "hard"

    def log_weight(self, rho: np.ndarray, eps: float) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        out = np.where(rho < eps, 0.0, -np.inf)
        return out

    def support_radius(self, eps: float) -> float:
        return float(eps)


class _GaussianKernel(_Kernel):
    """K_eps(rho) = exp(-rho^2 / (2 eps^2)) — smooth, never zero."""

    name = "gaussian"

    def log_weight(self, rho: np.ndarray, eps: float) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        if eps <= 0.0:
            return np.where(rho == 0.0, 0.0, -np.inf)
        return -0.5 * (rho / eps) ** 2

    def support_radius(self, eps: float) -> float:
        # exp(-x^2/2) < 1e-12 for |x| > ~7.43; pad to 8 sigma.
        return 8.0 * float(eps)


class _EpanechnikovKernel(_Kernel):
    """K_eps(rho) = max(0, 1 - rho^2 / eps^2) — compactly supported."""

    name = "epanechnikov"

    def log_weight(self, rho: np.ndarray, eps: float) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        if eps <= 0.0:
            return np.where(rho == 0.0, 0.0, -np.inf)
        u = 1.0 - (rho / eps) ** 2
        # Use -inf for u <= 0; log of small positive u is safe via np.log here
        # because np.where evaluates both branches and we mask after the fact.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_u = np.log(np.where(u > 0.0, u, 1.0))
        return np.where(u > 0.0, log_u, -np.inf)

    def support_radius(self, eps: float) -> float:
        return float(eps)


_KERNEL_REGISTRY = {
    "hard": _HardKernel,
    "gaussian": _GaussianKernel,
    "epanechnikov": _EpanechnikovKernel,
}


def _make_kernel(name: str) -> _Kernel:
    """Construct a kernel by name. Valid names: 'hard', 'gaussian', 'epanechnikov'."""
    try:
        return _KERNEL_REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Unknown ABCPMC kernel '{name}'. "
            f"Valid kernels: {sorted(_KERNEL_REGISTRY)}"
        )


class _ArchiveSnapshot:
    """Frozen view of the proposal distribution used at one past call.

    Stores enough state to evaluate the past proposal density q_tau(theta) at
    any new theta: archive positions, normalised mixture weights, and the
    Cholesky factor of the perturbation covariance.
    """

    __slots__ = ("positions", "weights", "L", "log_norm")

    def __init__(self, positions: np.ndarray, weights: np.ndarray, L: np.ndarray) -> None:
        self.positions = positions
        self.weights = weights
        self.L = L
        d = positions.shape[1]
        self.log_norm = -0.5 * d * np.log(2.0 * np.pi) - np.log(np.diag(L)).sum()

    def log_pdf(self, theta: np.ndarray) -> np.ndarray:
        """Log-PDF of each mixture component at theta. Shape: (k,)."""
        diffs = theta - self.positions  # (k, d)
        z = solve_triangular(self.L, diffs.T, lower=True)  # (d, k)
        return self.log_norm - 0.5 * np.einsum("ij,ij->j", z, z)


class _IncrementalCache:
    """
    Performance cache for ``ABCPMC.__call__``.

    Maintains sorted views of the evaluated history, updated incrementally as
    new individuals arrive (O(log N) per update).  Falls back to a full rebuild
    (O(N log N)) when the history doesn't grow by exactly the expected amount
    (e.g. after crash recovery or when multiple evaluations arrive at once).

    This is a *transparent optimisation*: identical results are produced whether
    the cache is hit or rebuilt.
    """

    __slots__ = (
        "history_len",
        "tol_from_history",
        "_accepted_by_loss",
        "_by_gen",
        "_accepted_by_gen",
        "_initial_tol",
    )

    def __init__(self, initial_tol: float) -> None:
        self.history_len: int = -1  # sentinel: no history processed yet
        self.tol_from_history: float = initial_tol
        self._accepted_by_loss: SortedKeyList = SortedKeyList(key=lambda ind: ind.loss)
        self._by_gen: SortedKeyList = SortedKeyList(key=lambda ind: ind.generation)
        self._accepted_by_gen: SortedKeyList = SortedKeyList(key=lambda ind: ind.generation)
        self._initial_tol: float = initial_tol

    # -- full rebuild (fallback) -----------------------------------------------

    def rebuild(self, inds, initial_tol: float) -> None:
        """Reconstruct all cached state from scratch.  O(N log N)."""
        self._initial_tol = initial_tol
        self.history_len = len(inds)

        # tol_from_history
        tols = [ind.tolerance for ind in inds if ind.tolerance is not None]
        self.tol_from_history = min(tols) if tols else initial_tol

        # All inds sorted by loss for tolerance-threshold queries.
        self._accepted_by_loss = SortedKeyList(inds, key=lambda ind: ind.loss)

        # All inds sorted by generation
        self._by_gen = SortedKeyList(inds, key=lambda ind: ind.generation)

        # Accepted at initial_tol, sorted by generation
        acc_gen = [ind for ind in inds if ind.loss < initial_tol]
        self._accepted_by_gen = SortedKeyList(acc_gen, key=lambda ind: ind.generation)

    # -- incremental update ----------------------------------------------------

    def update(self, new_ind) -> None:
        """Process one new evaluated individual.  Amortised O(log N)."""
        self.history_len += 1

        # Update tol_from_history (running minimum)
        if new_ind.tolerance is not None and new_ind.tolerance < self.tol_from_history:
            self.tol_from_history = new_ind.tolerance

        self._accepted_by_loss.add(new_ind)

        # Insert into inds-by-generation
        self._by_gen.add(new_ind)

        # Insert into accepted-by-generation (for GeometricDecayScheduler)
        if new_ind.loss < self._initial_tol:
            self._accepted_by_gen.add(new_ind)

    # -- query methods ---------------------------------------------------------

    @property
    def n_accepted(self) -> int:
        """Number of accepted individuals at the current tolerance."""
        return self._accepted_by_loss.bisect_key_left(self.tol_from_history)

    def count_below(self, tol: float) -> int:
        """Count accepted individuals with loss strictly below *tol*."""
        return self._accepted_by_loss.bisect_key_left(tol)

    def get_archive(self, tol: float, k: int) -> list:
        """Return top-*k* individuals with loss < *tol*, sorted by loss."""
        n = self._accepted_by_loss.bisect_key_left(tol)
        return list(self._accepted_by_loss[: min(n, k)])


class ABCPMC(Propagator):
    """
    Steady-state asynchronous ABC-PMC propagator with configurable likelihood
    kernel and optional streaming-AMIS reweighting.

    The algorithm is fully stateless w.r.t. its algorithmic state: the
    effective bandwidth and the active archive are reconstructed from the
    evaluated-history list ``inds`` passed to ``__call__`` on every
    invocation. Internal caches and the AMIS snapshot ring buffer are
    *performance* state — they do not change the result for valid inputs.
    ``inds`` must be append-only between calls.

    Kernel modes
    ------------
    The ``kernel`` parameter selects the ABC likelihood approximation
    K_eps(rho):

    - ``"hard"`` — ``1[rho < eps]``. Classical rejection-style ABC-PMC. A
      prior phase samples uniformly from the search space until the archive
      contains ``k`` particles with ``loss < eps``; past that, the archive is
      the top-``k`` by lowest loss within the tolerance.
    - ``"gaussian"`` — ``exp(-rho^2 / 2 eps^2)``. Smooth-kernel ABC
      (Wilkinson 2013). Every particle contributes proportionally; the
      prior-vs-archive phase distinction is replaced by a bootstrap rule
      (uniform prior until ``len(history) >= k``).
    - ``"epanechnikov"`` — ``max(0, 1 - rho^2 / eps^2)``. Compactly
      supported smooth kernel; particles with ``rho > eps`` contribute zero.

    For smooth kernels (``"gaussian"`` / ``"epanechnikov"``) the bandwidth
    ``eps`` is selected by the scheduler in its *kernel-aware* mode (e.g.
    target-ESS bisection for ``QuantileScheduler``).

    AMIS reweighting
    ----------------
    When ``amis_snapshots > 0``, the propagator maintains a ring buffer of
    past proposal distributions. New particles' importance weights use the
    balance-heuristic denominator

        q_bar_n(theta) = (1 / S) * sum_s q_s(theta)

    over snapshots s, in addition to the current proposal. This is the
    streaming variant of Cornuet et al. 2012's AMIS scheme and gives a
    coherent reweighting against the cumulative proposal mixture rather than
    the moment-of-arrival proposal. Set ``amis_snapshots=0`` for the legacy
    single-current-proposal weighting.

    Stored ``Individual`` weights are the *core* importance weights
    ``pi(theta) / denom(theta)`` measured at proposal time; the kernel
    factor ``K_eps(rho)`` is applied separately at use time, so changes to
    eps automatically reweight the archive without re-evaluating the
    simulator.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    _MAX_RESAMPLE_ATTEMPTS = 1000
    _MIN_DENOM = 1e-12  # numerical floor for the importance-weight denominator

    def __init__(
        self,
        limits: Dict,
        perturbation_scale: float = 0.8,
        k: int = 100,
        tol: float = 600.0,
        scheduler_type: str = "acceptance_rate",
        additional_needed_inds: Optional[int] = None,
        min_tol: Optional[float] = None,
        rng: Optional[random.Random] = None,
        kernel: str = "hard",
        amis_snapshots: int = 0,
        amis_interval: Optional[int] = None,
        **kwargs: Union[float, int, str],
    ) -> None:
        """
        Initialize the ABCPMC propagator.

        Parameters
        ----------
        limits : Dict
            Search-space limits for each gene (float intervals only).
        perturbation_scale : float
            Scale factor for the perturbation covariance.
        k : int
            Archive size: number of accepted individuals used to build the
            mixture proposal.
        tol : float
            Initial tolerance / bandwidth. **Never mutated** after
            construction; serves as the fallback when history is empty.
        scheduler_type : str
            Bandwidth scheduler. One of ``'quantile'``, ``'geometric_decay'``,
            ``'acceptance_rate'``.
        additional_needed_inds : int, optional
            Minimum extra accepted individuals beyond ``k`` before the
            scheduler proposes an update. Defaults to ``k``.
        rng : random.Random, optional
            Random number generator forwarded to the base ``Propagator``.
        min_tol : float, optional
            Lower bound applied to the effective bandwidth after scheduler
            proposals and fallback reconstruction.
        kernel : str
            ABC likelihood kernel. One of ``'hard'`` (classical),
            ``'gaussian'`` (smooth, recommended), ``'epanechnikov'``
            (compactly supported). Default ``'hard'`` for back-compat.
        amis_snapshots : int
            Number of past proposal snapshots to retain for streaming-AMIS
            balance-heuristic reweighting (Cornuet et al. 2012). Default 0
            (legacy single-proposal weighting). Recommended for smooth
            kernels: ``20``.
        amis_interval : int, optional
            Number of ``__call__`` invocations between snapshots. Defaults
            to ``k`` so each snapshot represents one archive turnover.
        **kwargs
            Additional parameters forwarded to the scheduler constructor
            (e.g. ``percentile`` for quantile, ``decay_factor`` for geometric
            decay, ``low_rate``/``high_rate`` for acceptance rate).
        """
        super().__init__(-1, 1, rng=rng)
        self.limits = limits
        self.perturbation_scale = perturbation_scale
        self.k = k
        self.tol = tol  # read-only initial tolerance; never mutated
        if min_tol is not None and min_tol < 0:
            raise ValueError("min_tol must be >= 0.")
        self.min_tol = min_tol
        if additional_needed_inds is None:
            self.additional_needed_inds = k
        else:
            self.additional_needed_inds = additional_needed_inds
        self.kernel_name = kernel
        self._kernel_fn: _Kernel = _make_kernel(kernel)
        self._kernel_aware = kernel != "hard"
        self.tolerance_scheduler = create_scheduler(
            scheduler_type,
            tol,
            k,
            self.additional_needed_inds,
            kernel_aware=self._kernel_aware,
            **kwargs,
        )
        self.rng_np = np.random.default_rng(
            self.rng.getrandbits(128)
        )  # Derive NumPy seed from Propagator RNG
        # Uniform prior density = 1 / volume (float limits only)
        float_limits = {key: v for key, v in self.limits.items() if isinstance(v[0], (float, np.floating))}
        if len(float_limits) != len(self.limits):
            raise ValueError("ABCPMC requires all search-space limits to be continuous (float) intervals.")
        volumes = [hi - lo for lo, hi in float_limits.values()]
        self.prior_density = 1.0 / float(np.prod(volumes))
        self._cache = _IncrementalCache(self.tol)

        # AMIS snapshot ring buffer (performance state, not algorithmic state).
        if amis_snapshots < 0:
            raise ValueError("amis_snapshots must be >= 0.")
        self._amis_snapshots = amis_snapshots
        self._amis_interval = amis_interval if amis_interval is not None else max(1, self.k)
        self._snapshots: deque = deque(maxlen=amis_snapshots) if amis_snapshots > 0 else deque(maxlen=1)
        self._calls_since_snapshot = 0

    def _update_cache(self, inds: List[Individual]) -> None:
        """Incrementally update the internal performance cache."""
        n = len(inds)
        cached_len = self._cache.history_len
        if cached_len < 0 or n < cached_len:
            # First call or history shrunk — full rebuild
            self._cache.rebuild(inds, self.tol)
            self.tolerance_scheduler.reset_cache()
        elif n > cached_len:
            # New individuals — process incrementally
            for new_ind in inds[cached_len:]:
                self._cache.update(new_ind)
        # n == cached_len: no change, use cached state as-is

    def filter_by_tolerance(self, inds: List[Individual], tol: float) -> List[Individual]:
        """
        Return individuals whose loss is strictly below *tol*.

        Parameters
        ----------
        inds : List[Individual]
            Candidate individuals.
        tol : float
            Tolerance threshold.

        Returns
        -------
        List[Individual]
            Filtered list.
        """
        return [ind for ind in inds if ind.loss < tol]

    def select_archive(self, inds: List[Individual], tol: float) -> List[Individual]:
        """
        Return up to ``k`` best accepted individuals as the active archive.

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history.
        tol : float
            Current effective tolerance.

        Returns
        -------
        List[Individual]
            Top-k accepted individuals sorted by loss (ascending).
        """
        accepted = self.filter_by_tolerance(inds, tol)
        return sorted(accepted, key=lambda ind: ind.loss)[: self.k]

    def weighted_covariance(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Unbiased weighted covariance with bias correction.

        Returns zero matrix if there is only one effective sample (denominator <= 0).

        Parameters
        ----------
        values : np.ndarray
            The values for which to compute the covariance.
        weights : np.ndarray
            The weights for each value.

        Returns
        -------
        np.ndarray
            The weighted covariance matrix.
        """
        values = np.asarray(values)
        weights = np.asarray(weights)
        mean = np.average(values, axis=0, weights=weights)
        w = weights.sum()
        denom = w**2 - np.sum(weights**2)
        if denom <= 0:
            return np.zeros((values.shape[1], values.shape[1]))
        factor = w / denom
        diffs = values - mean
        wd = weights[:, None] * diffs
        cov = wd.T @ diffs
        return factor * cov

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Generate a new candidate individual.

        The algorithm is stateless w.r.t. its algorithmic state: effective
        bandwidth and archive are reconstructed from ``inds`` on every call.
        The kernel mode (``self.kernel_name``) determines whether
        ``effective_tol`` is a hard rejection threshold or a smooth-kernel
        bandwidth. When the AMIS snapshot buffer is enabled the importance
        weight denominator is the balance-heuristic average over snapshots
        plus the current proposal (Veach 1997, Cornuet et al. 2012).

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history passed by Propulate.

        Returns
        -------
        Individual
            The next candidate (unevaluated, ``loss == inf``).
        """
        # 1. Reconstruct effective bandwidth from history.
        self._update_cache(inds)
        tol_from_history = self._cache.tol_from_history
        current_tol = tol_from_history
        if self.min_tol is not None:
            current_tol = max(current_tol, self.min_tol)

        # 2. Bootstrap (prior) phase. Hard kernel preserves the classical
        #    rule (need k particles below current threshold); smooth kernels
        #    use the simpler "len(history) >= k" rule since the kernel
        #    handles acceptance smoothly with no hard discontinuity.
        if self.kernel_name == "hard":
            need_more_particles = self._cache.count_below(current_tol) < self.k
        else:
            history_len = max(0, self._cache.history_len)
            need_more_particles = history_len < self.k

        if need_more_particles:
            sample = {key: self.rng.uniform(limit[0], limit[1]) for key, limit in self.limits.items()}
            child = Individual(position=sample, limits=self.limits)
            child.weight = 1.0
            return child

        # 3. Scheduler proposes a tighter bandwidth (monotone guarantee enforced below).
        proposed_tol = self.tolerance_scheduler.compute_cached(
            inds, current_tol,
            self._cache._accepted_by_loss,
            self._cache._by_gen,
            self._cache._accepted_by_gen,
        )
        candidate_tol = min(current_tol, proposed_tol)  # monotone guarantee
        if self.min_tol is not None:
            candidate_tol = max(candidate_tol, self.min_tol)

        # 4. Accept the tighter bandwidth. Under the hard kernel we only
        #    tighten if the archive remains full at the new threshold;
        #    under smooth kernels the archive is always top-k by lowest
        #    loss and the kernel weights handle the rest.
        if self.kernel_name == "hard":
            if self._cache.count_below(candidate_tol) >= self.k:
                effective_tol = candidate_tol
            else:
                effective_tol = current_tol
        else:
            effective_tol = candidate_tol
        if self.min_tol is not None:
            effective_tol = max(effective_tol, self.min_tol)

        # 5. Select archive. Hard kernel: top-k below threshold. Smooth
        #    kernels: top-k by lowest loss (kernel weight handles cutoff).
        if self.kernel_name == "hard":
            archive = self._cache.get_archive(effective_tol, self.k)
        else:
            archive = self._cache.get_archive(float("inf"), self.k)

        if len(archive) < self.k:
            # Defensive: shouldn't happen after the prior-phase guards above,
            # but recover gracefully by re-emitting a uniform prior draw.
            sample = {key: self.rng.uniform(limit[0], limit[1]) for key, limit in self.limits.items()}
            child = Individual(position=sample, limits=self.limits)
            child.weight = 1.0
            return child

        # 6. Compute effective archive weights = stored_weight * K_eps(loss).
        #    Done in log-space for numerical stability (Gaussian kernel
        #    weights can span many orders of magnitude).
        if any(ind.weight is None for ind in archive):
            if not getattr(self, "_warned_none", False):
                logger.warning(
                    "ABCPMC: one or more archive individuals have weight=None "
                    "(likely from an external propagator). Falling back to weight=1.0 "
                    "for those individuals; importance weights will be approximate."
                )
                self._warned_none = True
        raw = np.fromiter(
            (1.0 if ind.weight is None else max(float(ind.weight), 0.0) for ind in archive),
            dtype=float,
            count=len(archive),
        )
        losses_arr = np.fromiter(
            (float(ind.loss) for ind in archive), dtype=float, count=len(archive)
        )
        log_kernel = self._kernel_fn.log_weight(losses_arr, effective_tol)
        with np.errstate(divide="ignore"):
            log_raw = np.where(raw > 0.0, np.log(np.where(raw > 0.0, raw, 1.0)), -np.inf)
        log_effective = log_raw + log_kernel

        finite_mask = np.isfinite(log_effective)
        if not finite_mask.any():
            # All archive members have zero smooth weight (e.g. Epanechnikov
            # support too tight). Fall back to uniform mixture weights so the
            # algorithm continues to make progress.
            weights = np.full(len(archive), 1.0 / len(archive))
        else:
            log_eff_max = float(np.max(log_effective[finite_mask]))
            shifted = np.where(finite_mask, log_effective - log_eff_max, -np.inf)
            weights = np.exp(shifted)
            wsum = weights.sum()
            if wsum <= 0.0:
                weights = np.full(len(archive), 1.0 / len(archive))
            else:
                weights = weights / wsum

        positions = np.stack([ind.position for ind in archive])
        cov = self.weighted_covariance(positions, weights)
        cov += 1e-6 * np.eye(positions.shape[1])
        kernel_cov = self.perturbation_scale * cov
        kernel_cov = 0.5 * (kernel_cov + kernel_cov.T)
        try:
            L = np.linalg.cholesky(kernel_cov)
        except np.linalg.LinAlgError:
            kernel_cov += 1e-7 * np.eye(positions.shape[1])
            L = np.linalg.cholesky(kernel_cov)

        # 7. Sample candidate. Batched reject-resample amortises the
        #    BLAS-call overhead across many candidates.
        idx = int(self.rng_np.choice(len(archive), p=weights))
        parent = archive[idx]

        lo = np.array([lim[0] for lim in self.limits.values()], dtype=float)
        hi = np.array([lim[1] for lim in self.limits.values()], dtype=float)
        d = positions.shape[1]
        BATCH = 16
        candidate_pos = parent.position
        found = False
        for _ in range(self._MAX_RESAMPLE_ATTEMPTS // BATCH):
            z = self.rng_np.standard_normal((BATCH, d))
            cands = parent.position + z @ L.T
            in_box = np.all((cands >= lo) & (cands <= hi), axis=1)
            if in_box.any():
                candidate_pos = cands[int(np.argmax(in_box))]
                found = True
                break
        if not found:
            candidate_pos = np.clip(candidate_pos, lo, hi)
            logger.debug(
                "ABCPMC: reject-resample exhausted %d attempts; falling back to boundary clipping.",
                self._MAX_RESAMPLE_ATTEMPTS,
            )

        child = Individual(position=candidate_pos, limits=self.limits)
        child.tolerance = effective_tol  # stamped for future history reconstruction

        # 8. Compute importance weight via the balance-heuristic (AMIS)
        #    denominator: average of current proposal + snapshot proposals.
        diffs = child.position - positions                      # (k, d)
        z_solve = solve_triangular(L, diffs.T, lower=True)      # (d, k)
        log_norm = -0.5 * d * np.log(2.0 * np.pi) - np.log(np.diag(L)).sum()
        log_pdfs = log_norm - 0.5 * np.einsum("ij,ij->j", z_solve, z_solve)
        pdfs = np.exp(log_pdfs)
        current_proposal_pdf = float(np.dot(weights, pdfs))

        if self._amis_snapshots > 0 and len(self._snapshots) > 0:
            # Balance heuristic: average proposal density across snapshots
            # plus the current proposal. Removes the importance-weight
            # staleness that arises when the moving archive makes the
            # proposal at proposal time differ from the current proposal.
            total = current_proposal_pdf
            for snap in self._snapshots:
                snap_log_pdfs = snap.log_pdf(child.position)
                snap_pdfs = np.exp(snap_log_pdfs)
                total += float(np.dot(snap.weights, snap_pdfs))
            denom = total / (1.0 + len(self._snapshots))
        else:
            denom = current_proposal_pdf

        if denom < self._MIN_DENOM:
            warnings.warn(
                "ABCPMC: importance weight denominator below floor "
                "(child outside kernel mixture support). Assigning fallback weight.",
                RuntimeWarning,
                stacklevel=2,
            )
            denom = self._MIN_DENOM
        child.weight = self.prior_density / denom

        # 9. Snapshot the current proposal for future AMIS denominators.
        self._calls_since_snapshot += 1
        if self._amis_snapshots > 0 and self._calls_since_snapshot >= self._amis_interval:
            self._snapshots.append(
                _ArchiveSnapshot(positions.copy(), weights.copy(), L.copy())
            )
            self._calls_since_snapshot = 0

        return child


class EpsilonScheduler(ABC):
    """
    Base class for bandwidth scheduling in ABC-PMC.

    Subclasses must implement ``compute(inds, current_tol)`` — a **pure function**
    of the evaluated history and the current effective tolerance/bandwidth.
    No mutable state should be modified by ``compute``; all scheduling logic
    must be derivable from ``inds`` alone. Subclasses may additionally
    override ``compute_cached(...)`` to exploit append-only history with
    internal performance caches, but that cached path must remain equivalent
    to ``compute(...)`` for valid inputs.

    Kernel-aware mode
    -----------------
    When ``kernel_aware=True`` the scheduler is told that the propagator is
    using a smooth ABC kernel, in which case the selected value ``eps``
    plays the role of a kernel bandwidth rather than a hard rejection
    threshold. The selection logic itself uses raw loss statistics in both
    modes — under the paper's consistency theorem any monotone-decreasing
    bandwidth schedule satisfying (C3) is admissible, and the loss-quantile
    rule meets that. ESS-based selection (Del Moral, Doucet & Jasra 2012)
    is left as a future refinement.
    """

    def __init__(
        self,
        initial_tol: float,
        population_size: int,
        additional_needed_inds: int,
        *,
        kernel_aware: bool = False,
    ):
        self.initial_tol = initial_tol
        self.current_tol = initial_tol  # kept for backward-compat with deprecated update()
        self.population_size = population_size
        self.additional_needed_inds = additional_needed_inds
        self.kernel_aware = kernel_aware

    @abstractmethod
    def compute(self, inds: List[Individual], current_tol: float) -> float:
        """
        Propose the next tolerance as a pure function of history.

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history (all individuals, pre-filtered or not).
        current_tol : float
            The current effective tolerance reconstructed from history.

        Returns
        -------
        float
            Proposed new tolerance.  The caller (``ABCPMC.__call__``) enforces
            the monotone guarantee via ``min(current_tol, proposed)``.
        """
        ...

    def compute_cached(
        self,
        inds: List[Individual],
        current_tol: float,
        accepted_by_loss: list,
        inds_by_gen: list,
        accepted_by_gen: list,
    ) -> float:
        """
        Compute using pre-built cached data.

        Subclasses override this for O(1) fast paths.  The default
        implementation ignores the cached data and delegates to
        :meth:`compute`, so external schedulers work unchanged.
        """
        return self.compute(inds, current_tol)

    def reset_cache(self) -> None:
        """Reset internal performance caches (called on full history rebuild)."""
        pass

    def update(
        self,
        accepted_inds: Optional[List[Individual]] = None,
        all_inds: Optional[List[Individual]] = None,
    ) -> float:
        """Deprecated. Use ``compute(inds, current_tol)`` instead."""
        warnings.warn(
            "EpsilonScheduler.update() is deprecated and will be removed in a future release. "
            "Use compute(inds, current_tol) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        all_inds = all_inds or []
        accepted_inds = accepted_inds or []
        current_tol = self.current_tol
        proposed = self.compute(all_inds, current_tol)
        self.current_tol = min(current_tol, proposed)
        return self.current_tol


class QuantileScheduler(EpsilonScheduler):
    """
    Shrinks tolerance to a given percentile of the losses of *accepted* individuals.

    Unlike the prior stateful implementation, ``compute`` operates only on
    individuals already accepted at ``current_tol``, avoiding the all-history
    bias that arises when prior-phase samples (with large losses) are included.

    Uses lower-rank percentile (``losses_sorted[int(p/100 * n)]``) — no
    interpolation between adjacent ranks. This makes ``compute_cached`` an
    ``O(1)`` indexed access on the already-sorted history view, and keeps the
    cached and uncached paths in exact agreement.
    """

    def __init__(
        self,
        initial_tol: float,
        population_size: int,
        additional_needed_inds: int,
        percentile: float = 50.0,
        *,
        kernel_aware: bool = False,
    ):
        super().__init__(
            initial_tol, population_size, additional_needed_inds, kernel_aware=kernel_aware
        )
        if not (0 < percentile < 100):
            raise ValueError("Percentile must be between 0 and 100.")
        self.percentile = percentile

    def compute(self, inds: List[Individual], current_tol: float) -> float:
        accepted = [ind for ind in inds if ind.loss < current_tol]
        n = len(accepted)
        if n < self.population_size + self.additional_needed_inds:
            return current_tol
        losses_sorted = sorted(ind.loss for ind in accepted)
        return float(losses_sorted[int(self.percentile / 100.0 * n)])

    def compute_cached(self, inds, current_tol, accepted_by_loss, inds_by_gen, accepted_by_gen):
        n = accepted_by_loss.bisect_key_left(current_tol)
        if n < self.population_size + self.additional_needed_inds:
            return current_tol
        idx = int(self.percentile / 100.0 * n)
        return float(accepted_by_loss[idx].loss)


class GeometricDecayScheduler(EpsilonScheduler):
    """
    Shrinks tolerance by a fixed multiplicative factor per completed epoch.

    An *epoch* is a batch of ``population_size + additional_needed_inds``
    accepted individuals (those with ``loss < initial_tol``), processed in
    generation order.  The tolerance for epoch ``n`` is
    ``initial_tol * decay_factor^n``, but only if enough individuals in each
    batch survive the tightened threshold.

    The tolerance reconstruction is derived from the accepted history divided
    into batches by generation order. The cached fast path maintains
    incremental bookkeeping for append-only histories but must remain
    equivalent to a full replay from history.
    """

    def __init__(
        self,
        initial_tol: float,
        population_size: int,
        additional_needed_inds: int,
        decay_factor: float = 0.9,
        *,
        kernel_aware: bool = False,
    ):
        super().__init__(
            initial_tol, population_size, additional_needed_inds, kernel_aware=kernel_aware
        )
        if not (0 < decay_factor < 1):
            raise ValueError("Decay factor must be between 0 and 1.")
        self.decay_factor = decay_factor
        self._cached_consumed = 0
        self._cached_tol = initial_tol

    def compute(self, inds: List[Individual], current_tol: float) -> float:
        """
        Propose the next tolerance by replaying epoch history from ``initial_tol``.

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history.
        current_tol : float
            The current effective tolerance.  **Intentionally unused.**  This
            scheduler is fully stateless and reconstructs the epoch count by
            replaying batches of accepted individuals starting from
            ``self.initial_tol``.  This makes ``compute`` idempotent across
            repeated calls on the same history snapshot, at the cost of ignoring
            any tolerance value externally imposed outside the geometric decay
            schedule.  The monotone guarantee in ``ABCPMC.__call__`` ensures the
            returned value never exceeds the actual current tolerance.

        Returns
        -------
        float
            Proposed tolerance for the next epoch.
        """
        batch_size = self.population_size + self.additional_needed_inds
        accepted_all = sorted(
            [ind for ind in inds if ind.loss < self.initial_tol],
            key=lambda i: i.generation,
        )
        tol = self.initial_tol
        consumed = 0
        while consumed + batch_size <= len(accepted_all):
            batch = accepted_all[consumed : consumed + batch_size]
            next_tol = self.decay_factor * tol
            surviving = [i for i in batch if i.loss < next_tol]
            if len(surviving) >= self.population_size:
                tol = next_tol
            consumed += batch_size
        return tol

    def compute_cached(self, inds, current_tol, accepted_by_loss, inds_by_gen, accepted_by_gen):
        batch_size = self.population_size + self.additional_needed_inds
        tol = self._cached_tol
        consumed = self._cached_consumed
        while consumed + batch_size <= len(accepted_by_gen):
            batch = accepted_by_gen[consumed : consumed + batch_size]
            next_tol = self.decay_factor * tol
            surviving = [i for i in batch if i.loss < next_tol]
            if len(surviving) >= self.population_size:
                tol = next_tol
            consumed += batch_size
        self._cached_consumed = consumed
        self._cached_tol = tol
        return tol

    def reset_cache(self) -> None:
        self._cached_consumed = 0
        self._cached_tol = self.initial_tol


class AcceptanceRateScheduler(EpsilonScheduler):
    """
    Adjusts tolerance based on the acceptance rate in a recent sliding window.

    The window size is ``population_size + additional_needed_inds``.  If the
    most recent window contains more accepted individuals than ``high_rate``
    allows, the tolerance is tightened by ``shrink_factor``.  When the
    acceptance rate is below ``low_rate`` the scheduler holds the current
    tolerance unchanged.

    .. note::
        Tolerance expansion is architecturally impossible in this design: the
        monotone guarantee in ``ABCPMC.__call__`` clips any proposed value
        above ``tol_from_history`` to ``tol_from_history``.  An ``expand_factor``
        parameter was previously present but removed because it was silently
        discarded on every call.  If the acceptance rate drops too low, the
        algorithm holds the tolerance and waits for more accepted individuals.
    """

    def __init__(
        self,
        initial_tol: float,
        population_size: int,
        additional_needed_inds: int,
        low_rate: float = 0.1,
        high_rate: float = 0.3,
        shrink_factor: float = 0.9,
        *,
        kernel_aware: bool = False,
    ):
        super().__init__(
            initial_tol, population_size, additional_needed_inds, kernel_aware=kernel_aware
        )
        if not (0 < low_rate < high_rate < 1):
            raise ValueError("0 < low_rate < high_rate < 1 required.")
        self.low_rate = low_rate
        self.high_rate = high_rate
        self.shrink_factor = shrink_factor

    def compute(self, inds: List[Individual], current_tol: float) -> float:
        window_size = self.population_size + self.additional_needed_inds
        if len(inds) < window_size:
            return current_tol
        recent = sorted(inds, key=lambda i: i.generation)[-window_size:]
        accepted = [i for i in recent if i.loss < current_tol]
        rate = len(accepted) / len(recent)
        if rate > self.high_rate:
            return current_tol * self.shrink_factor
        return current_tol

    def compute_cached(self, inds, current_tol, accepted_by_loss, inds_by_gen, accepted_by_gen):
        window_size = self.population_size + self.additional_needed_inds
        if len(inds_by_gen) < window_size:
            return current_tol
        recent = inds_by_gen[-window_size:]
        accepted = sum(1 for i in recent if i.loss < current_tol)
        rate = accepted / len(recent)
        if rate > self.high_rate:
            return current_tol * self.shrink_factor
        return current_tol


class EpsilonSchedulerType(Enum):
    QUANTILE = "quantile"
    GEOMETRIC_DECAY = "geometric_decay"
    ACCEPTANCE_RATE = "acceptance_rate"


def create_scheduler(
    scheduler_type: str,
    initial_tol: float,
    population_size: int,
    additional_needed_inds: int,
    *,
    kernel_aware: bool = False,
    **kwargs,
) -> EpsilonScheduler:
    """
    Factory to create a bandwidth scheduler by name.

    Parameters
    ----------
    scheduler_type : str
        One of ``'quantile'``, ``'geometric_decay'``, ``'acceptance_rate'``.
    initial_tol : float
        Starting bandwidth value.
    population_size : int
        Size of the population used for scheduling decisions.
    additional_needed_inds : int
        Minimum extra accepted individuals beyond ``population_size`` required
        before a bandwidth update is proposed.
    kernel_aware : bool
        If True, the scheduler is informed that the propagator is using a
        smooth ABC kernel. The current implementation keeps the same
        loss-quantile selection logic regardless; ESS-based bandwidth
        selection is planned future work.
    **kwargs
        Additional parameters passed to the scheduler constructor.

    Returns
    -------
    EpsilonScheduler
        An instance of the requested scheduler.
    """
    try:
        st = EpsilonSchedulerType(scheduler_type)
    except ValueError:
        valid = [e.value for e in EpsilonSchedulerType]
        raise ValueError(f"Unknown scheduler type '{scheduler_type}'. Valid types: {valid}")

    if st == EpsilonSchedulerType.QUANTILE:
        return QuantileScheduler(
            initial_tol, population_size, additional_needed_inds, kernel_aware=kernel_aware, **kwargs
        )
    elif st == EpsilonSchedulerType.GEOMETRIC_DECAY:
        return GeometricDecayScheduler(
            initial_tol, population_size, additional_needed_inds, kernel_aware=kernel_aware, **kwargs
        )
    elif st == EpsilonSchedulerType.ACCEPTANCE_RATE:
        return AcceptanceRateScheduler(
            initial_tol, population_size, additional_needed_inds, kernel_aware=kernel_aware, **kwargs
        )
