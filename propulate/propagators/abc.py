import logging
import random
import warnings
from abc import ABC as AbstractBase
from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.stats import multivariate_normal

from ..population import Individual
from .base import Propagator

logger = logging.getLogger(__name__)


class ABC(Propagator):
    """
    Steady-state asynchronous ABC propagator.

    The algorithm is fully stateless: all algorithm state (effective tolerance,
    active archive) is reconstructed from the evaluated-history list ``inds``
    passed to ``__call__`` on every invocation.  No mutable instance variables
    are modified after construction, making the propagator safe for asynchronous
    HPC execution without synchronization barriers.

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

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(
        self,
        limits: Dict,
        perturbation_scale: float = 0.8,
        k: int = 100,
        tol: float = 600.0,
        scheduler_type: str = "acceptance_rate",
        additional_needed_inds: Optional[int] = None,
        rng: Optional[random.Random] = None,
        **kwargs: Dict[str, Union[float, int, str]],
    ) -> None:
        """
        Initialize the ABC propagator.

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
        if additional_needed_inds is None:
            self.additional_needed_inds = k
        else:
            self.additional_needed_inds = additional_needed_inds
        self.tolerance_scheduler = create_scheduler(
            scheduler_type, tol, k, self.additional_needed_inds, **kwargs
        )
        self.rng_np = np.random.default_rng()
        # Uniform prior density = 1 / volume (float limits only)
        float_limits = {key: v for key, v in self.limits.items() if isinstance(v[0], float)}
        if len(float_limits) != len(self.limits):
            raise ValueError("ABC requires all search-space limits to be continuous (float) intervals.")
        volumes = [hi - lo for lo, hi in float_limits.values()]
        self.prior_density = 1.0 / float(np.prod(volumes))

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
        cov = np.zeros((values.shape[1], values.shape[1]))
        for i in range(len(weights)):
            cov += weights[i] * np.outer(diffs[i], diffs[i])
        return factor * cov

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Generate a new candidate individual.

        The algorithm is fully stateless: all required state is derived from
        *inds* on every call.

        Steps
        -----
        1. Reconstruct effective tolerance from ``ind.tolerance`` fields.
        2. Propose lower tolerance via the scheduler (monotone guarantee).
        3. Build archive = top-k accepted individuals.
        4. If archive too small → sample from prior.
        5. Otherwise build Gaussian mixture kernel and sample candidate.
        6. Stamp candidate with effective tolerance and compute importance weight.

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history passed by Propulate.

        Returns
        -------
        Individual
            The next candidate (unevaluated, ``loss == inf``).
        """
        # 1. Reconstruct effective tolerance from stamped history
        tol_from_history = min(
            (ind.tolerance for ind in inds if ind.tolerance is not None),
            default=self.tol,
        )
        proposed_tol = self.tolerance_scheduler.compute(inds, tol_from_history)
        effective_tol = min(tol_from_history, proposed_tol)  # monotone guarantee

        # 2. Build archive
        archive = self.select_archive(inds, effective_tol)

        # 3. Prior phase: archive not yet large enough
        if len(archive) < self.k:
            sample = {key: self.rng.uniform(limit[0], limit[1]) for key, limit in self.limits.items()}
            child = Individual(position=sample, limits=self.limits)
            child.weight = 1.0
            return child

        # 4. Build perturbation kernel from archive
        weights = np.array([ind.weight for ind in archive], dtype=float)
        weights /= weights.sum()

        positions = np.stack([ind.position for ind in archive])
        cov = self.weighted_covariance(positions, weights)
        cov += 1e-6 * np.eye(positions.shape[1])
        kernel_cov = self.perturbation_scale * cov
        kernel_cov = 0.5 * (kernel_cov + kernel_cov.T)
        eigs = np.linalg.eigvalsh(kernel_cov)
        if eigs.min() <= 0:
            kernel_cov += (-eigs.min() + 1e-8) * np.eye(positions.shape[1])

        # 5. Sample candidate
        idx = self.rng.choices(range(len(archive)), weights=weights.tolist())[0]
        parent = archive[idx]

        lo = np.array([lim[0] for lim in self.limits.values()], dtype=float)
        hi = np.array([lim[1] for lim in self.limits.values()], dtype=float)
        candidate_pos = parent.position + self.rng_np.multivariate_normal(
            mean=np.zeros(positions.shape[1]), cov=kernel_cov
        )
        candidate_pos = np.clip(candidate_pos, lo, hi)

        child = Individual(position=candidate_pos, limits=self.limits)
        child.tolerance = effective_tol  # stamped for future history reconstruction

        # 6. Compute importance weight w* = pi(theta*) / q_n(theta*)
        pdfs = []
        for wp in archive:
            try:
                p = multivariate_normal.pdf(
                    child.position, mean=wp.position, cov=kernel_cov, allow_singular=True
                )
            except np.linalg.LinAlgError:
                p = multivariate_normal.pdf(
                    child.position,
                    mean=wp.position,
                    cov=kernel_cov + 1e-6 * np.eye(positions.shape[1]),
                    allow_singular=True,
                )
            pdfs.append(p)

        denom = float(np.dot(weights, pdfs))
        if denom == 0:
            warnings.warn(
                "ABC: importance weight denominator is zero (child is outside kernel support). "
                "Assigning fallback weight; consider re-sampling.",
                RuntimeWarning,
                stacklevel=2,
            )
            denom = 1e-12
        child.weight = self.prior_density / denom

        return child


class ToleranceScheduler(AbstractBase):
    """
    Base class for tolerance scheduling in ABC-PMC.

    Subclasses must implement ``compute(inds, current_tol)`` — a **pure function**
    of the evaluated history and the current effective tolerance.  No mutable
    state should be modified by ``compute``; all scheduling logic must be
    derivable from ``inds`` alone.
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
            Proposed new tolerance.  The caller (``ABC.__call__``) enforces
            the monotone guarantee via ``min(current_tol, proposed)``.
        """
        ...

    def update(
        self,
        accepted_inds: Optional[List[Individual]] = None,
        all_inds: Optional[List[Individual]] = None,
    ) -> float:
        """Deprecated. Use ``compute(inds, current_tol)`` instead."""
        warnings.warn(
            "ToleranceScheduler.update() is deprecated and will be removed in a future release. "
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


class QuantileToleranceScheduler(ToleranceScheduler):
    """
    Shrinks tolerance to a given percentile of the losses of *accepted* individuals.

    Unlike the prior stateful implementation, ``compute`` operates only on
    individuals already accepted at ``current_tol``, avoiding the all-history
    bias that arises when prior-phase samples (with large losses) are included.
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
        if len(accepted) < self.population_size + self.additional_needed_inds:
            return current_tol
        losses = [ind.loss for ind in accepted]
        return float(np.percentile(losses, self.percentile))


class GeometricDecayToleranceScheduler(ToleranceScheduler):
    """
    Shrinks tolerance by a fixed multiplicative factor per completed epoch.

    An *epoch* is a batch of ``population_size + additional_needed_inds``
    accepted individuals (those with ``loss < initial_tol``), processed in
    generation order.  The tolerance for epoch ``n`` is
    ``initial_tol * decay_factor^n``, but only if enough individuals in each
    batch survive the tightened threshold.

    The reconstruction is fully stateless: epoch count is derived from the
    length of accepted history divided by the batch size.
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

    def compute(self, inds: List[Individual], current_tol: float) -> float:
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


class AcceptanceRateToleranceScheduler(ToleranceScheduler):
    """
    Adjusts tolerance based on the acceptance rate in a recent sliding window.

    The window size is ``population_size + additional_needed_inds``.  If the
    most recent window contains too many accepted individuals the tolerance is
    tightened; if too few, it is relaxed.
    """

    def __init__(
        self,
        initial_tol: float,
        population_size: int,
        additional_needed_inds: int,
        low_rate: float = 0.1,
        high_rate: float = 0.3,
        shrink_factor: float = 0.9,
        expand_factor: float = 1.1,
    ):
        super().__init__(initial_tol, population_size, additional_needed_inds)
        if not (0 < low_rate < high_rate < 1):
            raise ValueError("0 < low_rate < high_rate < 1 required.")
        self.low_rate = low_rate
        self.high_rate = high_rate
        self.shrink_factor = shrink_factor
        self.expand_factor = expand_factor

    def compute(self, inds: List[Individual], current_tol: float) -> float:
        window_size = self.population_size + self.additional_needed_inds
        if len(inds) < window_size:
            return current_tol
        recent = sorted(inds, key=lambda i: i.generation)[-window_size:]
        accepted = [i for i in recent if i.loss < current_tol]
        rate = len(accepted) / len(recent)
        if rate > self.high_rate:
            return current_tol * self.shrink_factor
        elif rate < self.low_rate:
            return current_tol * self.expand_factor
        return current_tol


class SchedulerType(Enum):
    QUANTILE = "quantile"
    GEOMETRIC_DECAY = "geometric_decay"
    ACCEPTANCE_RATE = "acceptance_rate"


def create_scheduler(
    scheduler_type: str, initial_tol: float, population_size: int, additional_needed_inds: int, **kwargs
) -> ToleranceScheduler:
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
    ToleranceScheduler
        An instance of the requested scheduler.
    """
    try:
        st = SchedulerType(scheduler_type)
    except ValueError:
        valid = [e.value for e in SchedulerType]
        raise ValueError(f"Unknown scheduler type '{scheduler_type}'. Valid types: {valid}")

    if st == SchedulerType.QUANTILE:
        return QuantileToleranceScheduler(initial_tol, population_size, additional_needed_inds, **kwargs)
    elif st == SchedulerType.GEOMETRIC_DECAY:
        return GeometricDecayToleranceScheduler(initial_tol, population_size, additional_needed_inds, **kwargs)
    elif st == SchedulerType.ACCEPTANCE_RATE:
        return AcceptanceRateToleranceScheduler(initial_tol, population_size, additional_needed_inds, **kwargs)
