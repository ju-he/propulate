import logging
import random
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.linalg import solve_triangular
from sortedcontainers import SortedKeyList

from ..population import Individual
from .base import Propagator

logger = logging.getLogger(__name__)


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
    Steady-state asynchronous ABC-PMC propagator.

    The algorithm is fully stateless: all algorithm state (effective tolerance,
    active archive) is reconstructed from the evaluated-history list ``inds``
    passed to ``__call__`` on every invocation. Internal caches may be updated
    as a performance optimization, but they do not carry algorithmic state.
    This requires ``inds`` to be append-only: previously seen ``Individual``
    instances must not be mutated in place or reordered between calls.

    Tolerance memory is carried by each proposed ``Individual`` via the
    ``Individual.tolerance`` field.  The effective tolerance at any call is
    ``min(ind.tolerance for ind in inds if ind.tolerance is not None)``,
    falling back to the constructor argument ``tol`` when history is empty.
    A monotone decrease is guaranteed by taking the min of the history value
    and the scheduler's proposed value.

    .. note::
        **Weight staleness**: importance weights (``child.weight``) are computed
        against the archive state at proposal time.  In asynchronous execution
        results may arrive after the archive has evolved; the weights are
        therefore an approximation that degrades gracefully for slowly-changing
        archives.

    .. note::
        **Prior-phase weight mixing**: prior-phase individuals sampled before
        the archive fills are assigned ``weight=1.0`` (uniform prior).  Once
        the archive phase begins, these individuals may coexist in the archive
        alongside importance-weighted particles, mixing two weighting schemes.
        This is a benign approximation for large archives but importance-weight
        statistics should be treated as unreliable while the archive is newly
        formed.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    _MAX_RESAMPLE_ATTEMPTS = 1000

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
        **kwargs: Union[float, int, str],
    ) -> None:
        """
        Initialize the ABCPMC propagator.

        Parameters
        ----------
        limits : Dict
            Search-space limits for each gene (float intervals only).
        perturbation_scale : float
            Scale factor for the Gaussian perturbation covariance.
        k : int
            Archive size: number of best accepted individuals used to build
            the mixture proposal.
        tol : float
            Initial tolerance.  This value is **never mutated** after
            construction; it serves only as the fallback when history is empty.
        scheduler_type : str
            Tolerance scheduler.  One of ``'quantile'``,
            ``'geometric_decay'``, ``'acceptance_rate'``.
        additional_needed_inds : int, optional
            Minimum number of *extra* accepted individuals beyond ``k`` before
            the scheduler proposes a tolerance update.  Defaults to ``k``.
        rng : random.Random, optional
            Random number generator forwarded to the base ``Propagator``.
        min_tol : float, optional
            Lower bound applied to the effective tolerance after scheduler
            proposals and fallback reconstruction.
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
        self.tolerance_scheduler = create_scheduler(
            scheduler_type, tol, k, self.additional_needed_inds, **kwargs
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
        cov = np.einsum("i,ij,ik->jk", weights, diffs, diffs)
        return factor * cov

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Generate a new candidate individual.

        The algorithm is fully stateless: all required state is derived from
        *inds* on every call.

        Steps
        -----
        1. Reconstruct effective tolerance from ``ind.tolerance`` fields.
        2. Check preliminary archive at current tolerance; if too small → prior phase.
        3. Call scheduler to propose a tighter tolerance (monotone guarantee).
        4. Accept tighter tolerance only if the archive remains full; otherwise hold.
        5. Build perturbation kernel from archive.
        6. Sample candidate by perturbing a weighted-random archive member.
        7. Stamp candidate with effective tolerance and compute importance weight.

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history passed by Propulate.

        Returns
        -------
        Individual
            The next candidate (unevaluated, ``loss == inf``).
        """
        # 1. Update incremental cache; reconstructs tol_from_history.
        self._update_cache(inds)
        tol_from_history = self._cache.tol_from_history
        current_tol = tol_from_history
        if self.min_tol is not None:
            current_tol = max(current_tol, self.min_tol)

        # 2. Prior-phase guard: check archive BEFORE calling scheduler to avoid
        #    premature tolerance tightening that could oscillate the archive below k.
        if self._cache.count_below(current_tol) < self.k:
            sample = {key: self.rng.uniform(limit[0], limit[1]) for key, limit in self.limits.items()}
            child = Individual(position=sample, limits=self.limits)
            child.weight = 1.0
            return child

        # 3. Archive is full — safe to call scheduler and propose a tighter tolerance
        proposed_tol = self.tolerance_scheduler.compute_cached(
            inds, current_tol,
            self._cache._accepted_by_loss,
            self._cache._by_gen,
            self._cache._accepted_by_gen,
        )
        candidate_tol = min(current_tol, proposed_tol)  # monotone guarantee
        if self.min_tol is not None:
            candidate_tol = max(candidate_tol, self.min_tol)

        # 4. Accept tighter tolerance only if the archive remains full; otherwise hold
        if self._cache.count_below(candidate_tol) >= self.k:
            effective_tol = candidate_tol
        else:
            effective_tol = current_tol
        if self.min_tol is not None:
            effective_tol = max(effective_tol, self.min_tol)

        archive = self._cache.get_archive(effective_tol, self.k)

        # 5. Build perturbation kernel from archive
        _raw_weights = []
        _warned_none = False
        for ind in archive:
            if ind.weight is None:
                if not _warned_none:
                    logger.warning(
                        "ABCPMC: one or more archive individuals have weight=None "
                        "(likely from an external propagator). Falling back to weight=1.0 "
                        "for those individuals; importance weights will be approximate."
                    )
                    _warned_none = True
                _raw_weights.append(1.0)
            else:
                _raw_weights.append(ind.weight)
        weights = np.array(_raw_weights, dtype=float)
        weights /= weights.sum()

        positions = np.stack([ind.position for ind in archive])
        cov = self.weighted_covariance(positions, weights)
        cov += 1e-6 * np.eye(positions.shape[1])
        kernel_cov = self.perturbation_scale * cov
        kernel_cov = 0.5 * (kernel_cov + kernel_cov.T)
        # Cholesky-factorise once; L is reused for both sampling and the
        # log-PDF evaluation below.
        try:
            L = np.linalg.cholesky(kernel_cov)
        except np.linalg.LinAlgError:
            kernel_cov += 1e-7 * np.eye(positions.shape[1])
            L = np.linalg.cholesky(kernel_cov)

        # 6. Sample candidate — reuse L to avoid repeated Cholesky in the loop
        idx = int(self.rng_np.choice(len(archive), p=weights))
        parent = archive[idx]

        lo = np.array([lim[0] for lim in self.limits.values()], dtype=float)
        hi = np.array([lim[1] for lim in self.limits.values()], dtype=float)
        candidate_pos = parent.position
        d = positions.shape[1]
        for _attempt in range(self._MAX_RESAMPLE_ATTEMPTS):
            candidate_pos = parent.position + L @ self.rng_np.standard_normal(d)
            if np.all(candidate_pos >= lo) and np.all(candidate_pos <= hi):
                break
        else:
            candidate_pos = np.clip(candidate_pos, lo, hi)
            logger.debug(
                "ABCPMC: reject-resample exhausted %d attempts; falling back to boundary clipping.",
                self._MAX_RESAMPLE_ATTEMPTS,
            )

        child = Individual(position=candidate_pos, limits=self.limits)
        child.tolerance = effective_tol  # stamped for future history reconstruction

        # 7. Compute importance weight w* = pi(theta*) / q_n(theta*)
        # Manual multivariate-normal log-PDF using the existing Cholesky factor
        # L: a single triangular solve gives all k Mahalanobis distances and
        # avoids re-factorising the covariance inside scipy.
        diffs = child.position - positions                      # (k, d)
        z = solve_triangular(L, diffs.T, lower=True)            # (d, k)
        log_norm = -0.5 * d * np.log(2.0 * np.pi) - np.log(np.diag(L)).sum()
        log_pdfs = log_norm - 0.5 * np.einsum("ij,ij->j", z, z)  # (k,)
        pdfs = np.exp(log_pdfs)

        denom = float(np.dot(weights, pdfs))
        if denom < 1e-12:
            warnings.warn(
                "ABCPMC: importance weight denominator is zero (child is outside kernel support). "
                "Assigning fallback weight; consider re-sampling.",
                RuntimeWarning,
                stacklevel=2,
            )
            denom = 1e-12
        child.weight = self.prior_density / denom

        return child


class EpsilonScheduler(ABC):
    """
    Base class for tolerance scheduling in ABC-PMC.

    Subclasses must implement ``compute(inds, current_tol)`` — a **pure function**
    of the evaluated history and the current effective tolerance.  No mutable
    state should be modified by ``compute``; all scheduling logic must be
    derivable from ``inds`` alone. Subclasses may additionally override
    ``compute_cached(...)`` to exploit append-only history with internal
    performance caches, but that cached path must remain equivalent to
    ``compute(...)`` for valid inputs.
    """

    def __init__(self, initial_tol: float, population_size: int, additional_needed_inds: int):
        self.initial_tol = initial_tol
        self.current_tol = initial_tol  # kept for backward-compat with deprecated update()
        self.population_size = population_size
        self.additional_needed_inds = additional_needed_inds

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
    ):
        super().__init__(initial_tol, population_size, additional_needed_inds)
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
    ):
        super().__init__(initial_tol, population_size, additional_needed_inds)
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
    ):
        super().__init__(initial_tol, population_size, additional_needed_inds)
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
    scheduler_type: str, initial_tol: float, population_size: int, additional_needed_inds: int, **kwargs
) -> EpsilonScheduler:
    """
    Factory to create a tolerance scheduler by name.

    Parameters
    ----------
    scheduler_type : str
        One of ``'quantile'``, ``'geometric_decay'``, ``'acceptance_rate'``.
    initial_tol : float
        Starting tolerance value.
    population_size : int
        Size of the population used for scheduling decisions.
    additional_needed_inds : int
        Minimum extra accepted individuals beyond ``population_size`` required
        before a tolerance update is proposed.
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
        return QuantileScheduler(initial_tol, population_size, additional_needed_inds, **kwargs)
    elif st == EpsilonSchedulerType.GEOMETRIC_DECAY:
        return GeometricDecayScheduler(initial_tol, population_size, additional_needed_inds, **kwargs)
    elif st == EpsilonSchedulerType.ACCEPTANCE_RATE:
        return AcceptanceRateScheduler(initial_tol, population_size, additional_needed_inds, **kwargs)
