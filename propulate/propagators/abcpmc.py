import logging
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtr
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


class _HardKernel(_Kernel):
    """K_eps(rho) = 1 if rho < eps else 0 — the classical ABC indicator."""

    name = "hard"

    def log_weight(self, rho: np.ndarray, eps: float) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        out = np.where(rho < eps, 0.0, -np.inf)
        return out


class _GaussianKernel(_Kernel):
    """K_eps(rho) = exp(-rho^2 / (2 eps^2)) — smooth, never zero."""

    name = "gaussian"

    def log_weight(self, rho: np.ndarray, eps: float) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        if eps <= 0.0:
            return np.where(rho == 0.0, 0.0, -np.inf)
        return -0.5 * (rho / eps) ** 2


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


def _log_box_mass(
    positions: np.ndarray, sigma: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Log of the per-component in-box probability mass ``log Z_j``.

    Candidates are rejection-sampled *inside* the box, so each perturbation
    component is a Gaussian **truncated** to ``[lo, hi]``. Its true density is the
    untruncated Gaussian divided by its in-box mass ``Z_j = P(component j ∈ box)``;
    omitting ``Z_j`` from the importance-sampling denominator inflates the weights
    of near-boundary particles by ``1 / Z_j``.

    For a component centred at ``positions[j]`` with marginal standard deviations
    ``sigma`` (shared across components — all share the perturbation covariance),

        log Z_j ≈ Σ_d log( Φ((hi_d - μ_jd)/σ_d) − Φ((lo_d - μ_jd)/σ_d) ).

    This is **exact for a diagonal covariance** and a **diagonal-marginal
    approximation for a full covariance** (it ignores off-diagonal correlations;
    a box has no closed-form Gaussian mass under full covariance). Component means
    lie inside the box, so ``Z_j`` is bounded away from zero and the direct
    ``ndtr`` difference is numerically safe.
    """
    sigma = np.where(sigma > 0.0, sigma, 1.0)  # guard zero-variance dims
    z_hi = (hi - positions) / sigma  # (k, d)
    z_lo = (lo - positions) / sigma
    mass = np.clip(ndtr(z_hi) - ndtr(z_lo), 1e-300, 1.0)  # (k, d)
    return np.log(mass).sum(axis=1)  # (k,)


# Rows processed per chunk in ABCPMC.extract_posterior. Bounds the retroactive
# AMIS reweighting to O(chunk · k) memory so a fast simulator's multi-million
# individual history does not allocate an (n, k) matrix of tens of GB. Large
# enough that the per-chunk Python/NumPy overhead is negligible.
_EXTRACT_POSTERIOR_CHUNK = 65_536


class _ArchiveSnapshot:
    """Frozen view of the proposal distribution used at one call.

    Stores the raw mixture description (archive positions, normalised mixture
    weights, the Cholesky factor of the perturbation covariance, and the
    per-component in-box mass ``log Z_j``) plus derived quantities precomputed
    **once at construction** so that density evaluation on the per-``__call__``
    hot path is pure NumPy:

    - ``L_inv`` — explicit inverse of the triangular factor. Whitening becomes
      a small matmul instead of a LAPACK ``trtrs`` call per query; scipy's
      ``solve_triangular`` wrapper overhead (validation, ``asarray_chkfinite``)
      dominates at the tiny ``d`` encountered here. The factor is jittered SPD
      (see ``_build_proposal``), so the explicit triangular inverse is
      well-conditioned for density evaluation.
    - ``mu_w`` — whitened component means ``L⁻¹ μ_j``, shape ``(k, d)``.
    - ``log_comp_const`` — per-component constant
      ``log w_j + log_norm − log Z_j`` (−inf for zero-weight components), so
      ``log(w_j · N_j(θ)) = log_comp_const_j − ½‖L⁻¹θ − mu_w_j‖²`` and a full
      mixture density is a single logsumexp over these terms.
    """

    __slots__ = (
        "positions", "weights", "L", "log_norm", "log_box_mass",
        "L_inv", "mu_w", "log_comp_const",
    )

    def __init__(
        self,
        positions: np.ndarray,
        weights: np.ndarray,
        L: np.ndarray,
        log_box_mass: np.ndarray,
    ) -> None:
        self.positions = positions
        self.weights = weights
        self.L = L
        self.log_box_mass = log_box_mass
        d = positions.shape[1]
        self.log_norm = -0.5 * d * np.log(2.0 * np.pi) - np.log(np.diag(L)).sum()
        self.L_inv = solve_triangular(L, np.eye(d), lower=True)  # (d, d)
        self.mu_w = positions @ self.L_inv.T  # (k, d) whitened means
        with np.errstate(divide="ignore"):
            log_w = np.where(weights > 0.0, np.log(np.where(weights > 0.0, weights, 1.0)), -np.inf)
        self.log_comp_const = log_w + self.log_norm - log_box_mass  # (k,)

    def log_component_terms(self, theta: np.ndarray) -> np.ndarray:
        """``log(w_j · N_j(theta))`` per component, shape ``(k,)``.

        Pure NumPy (one small matmul + one einsum); the hot-path building
        block for the balance-heuristic denominator — the mixture-of-mixtures
        over [current proposal + snapshots] flattens into one logsumexp over
        the concatenated component terms.
        """
        theta_w = self.L_inv @ theta  # (d,)
        diff = self.mu_w - theta_w  # (k, d)
        return self.log_comp_const - 0.5 * np.einsum("ij,ij->i", diff, diff)

    def log_pdf(self, theta: np.ndarray) -> np.ndarray:
        """Log-PDF of each truncated mixture component at theta. Shape: (k,)."""
        theta_w = self.L_inv @ theta
        diff = self.mu_w - theta_w
        return self.log_norm - 0.5 * np.einsum("ij,ij->i", diff, diff) - self.log_box_mass

    def log_mixture_density(self, thetas: np.ndarray) -> np.ndarray:
        """Log mixture density ``log q(theta)`` for a batch of points.

        Vectorised over a ``(n, d)`` batch, returning ``(n,)``. All components
        share the covariance ``L Lᵀ``, so the Mahalanobis distance equals the
        Euclidean distance after whitening; the whitened means and combined
        per-component constants are precomputed at construction, so a batch
        costs one ``(n, d) @ (d, d)`` whitening matmul plus the ``(n, k)``
        distance matrix. Used by :meth:`ABCPMC.extract_posterior` for the
        retroactive reweighting (called once per chunk — no per-chunk solves).
        """
        theta_w = thetas @ self.L_inv.T  # (n, d)
        sq = (
            (theta_w ** 2).sum(1)[:, None]
            + (self.mu_w ** 2).sum(1)[None, :]
            - 2.0 * theta_w @ self.mu_w.T
        )  # (n, k) squared Mahalanobis distances
        return logsumexp(self.log_comp_const[None, :] - 0.5 * sq, axis=1)  # (n,)


def _gen_order(ind: "Individual") -> tuple:
    """Total generation order: ``(generation, island, rank)``.

    Generations collide across workers (every rank counts 0, 1, 2, ...), so a
    bare ``generation`` key leaves the order of ties to arrival order, which
    is rank-dependent under MPI. Breaking ties by ``(island, rank)`` makes
    every generation-ordered view a deterministic function of the history
    *set* — identical on every rank — so the epoch structure of
    :class:`GeometricDecayScheduler` and the sliding window of
    :class:`AcceptanceRateScheduler` do not depend on message-arrival
    interleaving, and the cached and uncached scheduler paths agree exactly.
    """
    return (ind.generation, ind.island, ind.rank)


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
        "_last_ind",
    )

    def __init__(self, initial_tol: float) -> None:
        self.history_len: int = -1  # sentinel: no history processed yet
        self.tol_from_history: float = initial_tol
        self._accepted_by_loss: SortedKeyList = SortedKeyList(key=lambda ind: ind.loss)
        self._by_gen: SortedKeyList = SortedKeyList(key=_gen_order)
        self._accepted_by_gen: SortedKeyList = SortedKeyList(key=_gen_order)
        self._initial_tol: float = initial_tol
        # Identity of the last individual processed, used to verify that the
        # previously-cached prefix is still intact before taking the
        # append-only fast path (see ABCPMC._update_cache).
        self._last_ind = None

    # -- full rebuild (fallback) -----------------------------------------------

    def rebuild(self, inds, initial_tol: float) -> None:
        """Reconstruct all cached state from scratch.  O(N log N).

        ``tol_from_history`` is kept **monotone across rebuilds** within the
        lifetime of this cache: rebuilds are triggered by island migration,
        where an emigrant is *deactivated* on the source island but continues
        to exist on its destination — if that emigrant was the sole carrier of
        the tightest stamped tolerance, recomputing the minimum from the
        remaining active history alone would silently *loosen* the effective
        bandwidth. Taking ``min(previous, min over new history)`` preserves
        the monotone-decrease guarantee across migration events.

        Trade-off: after a crash/restart the cache is a fresh instance, so the
        tolerance reconstructs from the checkpointed history alone and may
        transiently loosen if the min-carrier had emigrated — consistent with
        the "no persisted algorithmic state" design; the next archive-phase
        breed restamps a tightened tolerance.
        """
        self._initial_tol = initial_tol
        self.history_len = len(inds)

        # tol_from_history: running minimum, monotone across rebuilds.
        tols = [ind.tolerance for ind in inds if ind.tolerance is not None]
        new_min = min(tols) if tols else initial_tol
        self.tol_from_history = min(self.tol_from_history, new_min)

        # All inds sorted by loss for tolerance-threshold queries.
        self._accepted_by_loss = SortedKeyList(inds, key=lambda ind: ind.loss)

        # All inds sorted by generation
        self._by_gen = SortedKeyList(inds, key=_gen_order)

        # Accepted at initial_tol, sorted by generation
        acc_gen = [ind for ind in inds if ind.loss < initial_tol]
        self._accepted_by_gen = SortedKeyList(acc_gen, key=_gen_order)

        self._last_ind = inds[-1] if inds else None

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

        self._last_ind = new_ind

    # -- query methods ---------------------------------------------------------

    @property
    def n_accepted(self) -> int:
        """Number of accepted individuals at the current tolerance."""
        return self._accepted_by_loss.bisect_key_left(self.tol_from_history)

    def count_below(self, tol: float) -> int:
        """Count accepted individuals with loss strictly below *tol*."""
        return self._accepted_by_loss.bisect_key_left(tol)

    def get_archive(self, tol: float, k: int) -> list:
        """Return top-*k* mixture-eligible individuals with loss < *tol*, by loss.

        Individuals with ``weight == 0`` (underflow-exhausted candidates) are
        skipped: a zero-weight particle contributes nothing to the proposal
        mixture and can never be selected as a parent, so letting it occupy
        one of the ``k`` slots would silently shrink the effective mixture.
        ``weight is None`` (external propagator) stays eligible — it falls
        back to 1.0 in the proposal build. Zero-weight individuals are rare,
        so the lazy scan stays O(k) in practice.
        """
        out = []
        for ind in self._accepted_by_loss.irange_key(None, tol, inclusive=(True, False)):
            if ind.weight is None or ind.weight > 0.0:
                out.append(ind)
                if len(out) == k:
                    break
        return out


class ABCPMC(Propagator):
    """
    Steady-state asynchronous ABC-PMC propagator with configurable likelihood
    kernel and optional streaming-AMIS reweighting.

    Statelessness and the posterior estimator
    -----------------------------------------
    The algorithm is stateless w.r.t. its algorithmic state: the effective
    bandwidth and the active archive are reconstructed from the evaluated-history
    list ``inds`` on every ``__call__``. One qualification: the effective
    bandwidth is additionally kept *monotone across migration-induced cache
    rebuilds* within the propagator's lifetime, because migration deactivates
    emigrants — if the sole carrier of the tightest stamped tolerance
    emigrates, a pure reconstruction from the remaining active history would
    loosen the bandwidth (see :meth:`_IncrementalCache.rebuild`). The posterior the paper reports is
    :meth:`extract_posterior`, which is a **pure function of the history** — it
    replays the proposal sequence deterministically — so it is identical whether
    computed in one run or after a crash/restart with an empty buffer. That is
    what makes "stateless / crash-recoverable" hold for the *estimator*, not just
    the archive.

    The AMIS snapshot ring buffer is **proposal-side performance state**: it
    supplies the proposal-time importance denominator and never feeds
    :meth:`extract_posterior`, so a stale or empty buffer changes only proposal
    quality, never the reported posterior. The buffer is cleared on a genuine
    history rollback (shrink or non-append-only mutation under island
    migration) so a rolled-back future cannot leak forward. The per-call
    ``Individual.weight`` is subtler: it is never used *as a particle's
    importance weight* in the reported posterior (that weight is recomputed
    retroactively), but it **is** read during the proposal reconstruction —
    :meth:`extract_posterior` replays past proposals via
    :meth:`_build_proposal`, whose mixture weights are built from the stored
    per-individual weights. That is deliberate: the stored weight is part of
    the history that determines which proposal was actually used, and it is
    checkpointed with the individual, so the replay stays faithful across a
    restart. ``inds`` is append-only in single-island runs; under migration
    the cache detects the non-append-only case and rebuilds.

    Kernel modes
    ------------
    The ``kernel`` parameter selects the ABC likelihood approximation
    K_eps(rho):

    - ``"hard"`` — ``1[rho < eps]``. Classical rejection-style ABC-PMC. A
      prior phase samples uniformly from the search space until the archive
      contains ``k`` particles with ``loss < eps``; past that, the archive is
      the **top-k by lowest loss among individuals with ``loss < eps``** —
      a quantile-trimmed rejection rule, not strict ``loss < eps`` retention
      (see ``select_archive``). At steady state with k particles below eps
      the two coincide, but when the scheduler tightens eps the top-k rule
      can keep accepted particles that the strict rule would drop on the
      next call.
    - ``"gaussian"`` — ``exp(-rho^2 / 2 eps^2)``. Smooth-kernel ABC
      (Wilkinson 2013). Every particle contributes proportionally; the
      prior-vs-archive phase distinction is replaced by a bootstrap rule
      (uniform prior until ``len(history) >= k``).
    - ``"epanechnikov"`` — ``max(0, 1 - rho^2 / eps^2)``. Compactly
      supported smooth kernel; particles with ``rho > eps`` contribute zero.

    For smooth kernels (``"gaussian"`` / ``"epanechnikov"``) the bandwidth
    ``eps`` is selected by the scheduler in its *kernel-aware* mode, which is
    active by default for smooth kernels (``kernel_aware`` is set from the
    kernel choice). The scheduler chooses ``eps`` by a bracketed search on the
    kernel-weighted effective sample size, targeting a fixed per-step ESS
    retention ratio (the Del Moral, Doucet & Jasra 2012 successive-population
    rule), subject to the monotone-decrease guarantee and a per-call
    tightening cap (``max_tighten_factor``). The hard kernel retains the
    loss-quantile / acceptance-rate / geometric-decay rule.

    AMIS reweighting
    ----------------
    Two distinct weightings live here; keep them separate:

    - The *proposal-time* weight stored on each ``Individual`` is the core
      importance weight ``pi(theta) / q_bar_tau(theta)`` measured against the
      balance-heuristic denominator at proposal time, where (when
      ``amis_snapshots > 0``) ``q_bar`` averages the current proposal with a ring
      buffer of past proposal snapshots (the streaming variant of Cornuet et al.
      2012's AMIS scheme). This weight only steers parent selection and the next
      proposal covariance; ``amis_snapshots=0`` recovers the single-current-
      proposal weighting.
    - The *reported posterior* is :meth:`extract_posterior`, which applies the
      **retroactive** balance heuristic: every particle is reweighted against the
      cumulative proposal mixture reconstructed at the END of the run (evaluated
      at every past particle), times the kernel factor ``K_{eps}(rho)``. This is
      the estimator the consistency + CLT are stated for; the frozen proposal-time
      weight is not.

    The kernel factor ``K_eps(rho)`` is applied at use time, so changing eps
    reweights the archive without re-evaluating the simulator.

    Performance note (BLAS threading)
    ---------------------------------
    The per-call hot path works on tiny matrices (``d × d`` triangular
    solves, ``k``-component mixtures), where multi-threaded BLAS is pure
    overhead: with default OpenBLAS/MKL settings the thread pool spin-waits
    on every ``solve_triangular`` call (observed: ~30× per-call slowdown at
    ``d = 2`` on a 12-core node). Pin BLAS to one thread per process
    (``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1``) — for MPI runs with one
    worker per rank this is standard practice anyway.

    See Also
    --------
    :class:`Propagator` : The parent class.
    :meth:`extract_posterior` : The retroactive AMIS posterior estimator.
    """

    _MAX_RESAMPLE_ATTEMPTS = 1000
    _MIN_DENOM = 1e-12  # numerical floor for the importance-weight denominator
    _MAX_WEIGHT_RETRIES = 5  # retries when the AMIS denominator underflows the floor

    def __init__(
        self,
        limits: Dict,
        perturbation_scale: float = 0.8,
        k: int = 100,
        tol: Optional[float] = None,
        scheduler_type: str = "acceptance_rate",
        additional_needed_inds: Optional[int] = None,
        min_tol: Optional[float] = None,
        rng: Optional[random.Random] = None,
        kernel: str = "hard",
        amis_snapshots: int = 20,
        amis_interval: Optional[int] = None,
        ess_target: float = 0.95,
        max_tighten_factor: float = 0.5,
        bisect_interval: Optional[int] = None,
        **kwargs: Union[float, int, str],
    ) -> None:
        """
        Initialize the ABCPMC propagator.

        Parameters
        ----------
        limits : Dict
            Search-space limits for each gene (float intervals only).
        perturbation_scale : float
            Scale factor applied to the weighted archive covariance to form
            the perturbation kernel covariance. PMC optimality (Beaumont et
            al. 2009; Filippi et al. 2013) corresponds to roughly twice the
            weighted covariance (``perturbation_scale ≈ 2.0``); the default
            ``0.8`` is deliberately narrower (best-fit / optimisation-leaning
            behaviour) and under-explores the tails for posterior
            approximation. Sweep this for posterior-quality work.
        k : int
            Archive size: number of accepted individuals used to build the
            mixture proposal.
        tol : float, optional
            Initial tolerance / bandwidth. **Never mutated** after
            construction. Required (explicit) for the hard kernel. For smooth
            kernels the default ``None`` selects the bandwidth data-driven:
            the median loss of the first ``k`` finite-loss individuals in
            generation order — a pure function of the history set (identical
            after a crash/restart; rank-invariant given the set thanks to the
            deterministic generation tie-break). The data-driven value is
            only consulted until the first stamped tolerance enters the
            history, after which the running minimum takes over.
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
            Number of past proposal snapshots retained for the streaming-AMIS
            balance-heuristic denominator used at *proposal time* (Cornuet et al.
            2012), and the default number of proposals
            :meth:`extract_posterior` reconstructs for the *reported* posterior.
            Default ``20``. Set to ``0`` for the legacy single-current-proposal
            proposal-time weighting (this does not affect
            :meth:`extract_posterior`, which always reconstructs the cumulative
            mixture from history).
        amis_interval : int, optional
            Number of ``__call__`` invocations between snapshots. Defaults
            to ``k`` so each snapshot represents one archive turnover.
        ess_target : float
            Per-step ESS *retention ratio* for the kernel-aware bandwidth
            search (Del Moral, Doucet & Jasra 2012): each tightening step
            picks ε so that ``ESS(ε) ≈ ess_target · ESS(current_tol)``. Active
            only when ``kernel`` is a smooth kernel (``"gaussian"`` or
            ``"epanechnikov"``); ignored for the hard kernel which retains the
            loss-quantile / acceptance-rate / geometric-decay schedule. Must
            lie in ``(0, 1]``; default ``0.95`` (retain 95% of ESS per step).

            Semantics: tightening only proceeds while at least
            ``k + additional_needed_inds`` individuals satisfy ``loss < ε``
            (the scheduler's gate), so the bandwidth equilibrates near the
            ``(k + additional_needed_inds)``-th smallest loss seen so far and
            keeps tracking that order statistic as evaluations accumulate.
            ``ess_target`` therefore controls the *approach speed* toward that
            equilibrium, not the final posterior sharpness — use ``min_tol``
            to pin an explicit floor. Note also that the retention rule fires
            per ``__call__`` (per bred individual), not once per SMC
            population as in the original formulation, so the same
            ``ess_target`` tightens faster with more workers (see
            :class:`EpsilonScheduler`).
        max_tighten_factor : float
            Per-call tightening cap for the kernel-aware bandwidth search:
            any proposed ε is floored at ``max_tighten_factor · current_tol``,
            bounding the worst-case single-call tightening regardless of the
            shape of the ESS(ε) curve (which is not monotone in general and
            can be flat, e.g. for tied losses from discrete summary
            statistics). Must lie in ``(0, 1)``; default ``0.5`` (ε can at
            most halve per call).
        bisect_interval : int, optional
            Number of ``__call__`` invocations between kernel-aware ESS
            searches. The search costs O(n_accepted) per evaluation, so
            running it every call would make the otherwise O(log n) hot path
            linear in the accepted count; in between, the last proposal is
            held (bounded staleness — tightening is only delayed, never
            reversed, thanks to the monotone guarantee). Defaults to ``k``,
            matching one archive turnover. Only affects the cached scheduler
            path; the pure ``compute`` reference path is never throttled.
        **kwargs
            Additional parameters forwarded to the scheduler constructor
            (e.g. ``percentile`` for quantile, ``decay_factor`` for geometric
            decay, ``high_rate``/``shrink_factor`` for acceptance rate).

        Raises
        ------
        ValueError
            On invalid configuration: non-float limits, ``lo >= hi`` for any
            dimension, ``k < 1``, ``tol <= 0``, ``perturbation_scale <= 0``,
            ``additional_needed_inds < 0``, ``min_tol < 0``,
            ``amis_snapshots < 0``, ``amis_interval < 1``, ``ess_target``
            outside ``(0, 1]``, ``max_tighten_factor`` outside ``(0, 1)``, or
            ``bisect_interval < 1``.
            Additionally raised from ``__call__`` when an individual carries a
            negative or NaN loss (ABC requires a nonnegative discrepancy).
        """
        super().__init__(-1, 1, rng=rng)
        self.limits = limits
        # Validate the primary input (limits) first, then independent scalar
        # arguments; the cross-parameter tol-requires-hard-kernel check comes
        # last so a wrong argument is never masked by a merely missing one.
        float_limits = {key: v for key, v in limits.items() if isinstance(v[0], (float, np.floating))}
        if len(float_limits) != len(limits):
            raise ValueError("ABCPMC requires all search-space limits to be continuous (float) intervals.")
        if any(hi <= lo for lo, hi in float_limits.values()):
            raise ValueError("ABCPMC limits must satisfy lo < hi for every dimension.")
        if perturbation_scale <= 0.0:
            raise ValueError("perturbation_scale must be > 0.")
        self.perturbation_scale = perturbation_scale
        if k < 1:
            raise ValueError("k (archive size) must be >= 1.")
        self.k = k
        if min_tol is not None and min_tol < 0:
            raise ValueError("min_tol must be >= 0.")
        self.min_tol = min_tol
        if additional_needed_inds is None:
            self.additional_needed_inds = k
        elif additional_needed_inds < 0:
            raise ValueError("additional_needed_inds must be >= 0.")
        else:
            self.additional_needed_inds = additional_needed_inds
        if not (0.0 < ess_target <= 1.0):
            raise ValueError("ess_target must be in (0, 1].")
        self.ess_target = float(ess_target)
        if not (0.0 < max_tighten_factor < 1.0):
            raise ValueError("max_tighten_factor must be in (0, 1).")
        self.max_tighten_factor = float(max_tighten_factor)
        if bisect_interval is not None and bisect_interval < 1:
            raise ValueError("bisect_interval must be >= 1.")
        self.kernel_name = kernel
        self._kernel_fn: _Kernel = _make_kernel(kernel)
        self._kernel_aware = kernel != "hard"
        if tol is None:
            if not self._kernel_aware:
                raise ValueError(
                    "ABCPMC with the hard kernel requires an explicit tol (initial "
                    "rejection threshold); the data-driven default (tol=None) is "
                    "available only for smooth kernels."
                )
        elif tol <= 0.0:
            raise ValueError("tol (initial tolerance/bandwidth) must be > 0.")
        self.tol = tol  # read-only; None = data-driven initial bandwidth (smooth kernels)
        # Internal fallback where a numeric tolerance is structurally required
        # (cache construction/rebuild, scheduler initial_tol). +inf encodes
        # "no bandwidth fixed yet"; the data-driven value replaces it in
        # __call__ once k finite-loss individuals exist.
        self._tol_fallback = tol if tol is not None else float("inf")
        self.tolerance_scheduler = create_scheduler(
            scheduler_type,
            self._tol_fallback,
            k,
            self.additional_needed_inds,
            kernel_aware=self._kernel_aware,
            kernel_fn=self._kernel_fn,
            ess_target=self.ess_target,
            max_tighten_factor=self.max_tighten_factor,
            bisect_interval=bisect_interval,
            **kwargs,
        )
        if self._kernel_aware:
            logger.info(
                "ABCPMC: smooth kernel '%s' selects the bandwidth by target-ESS "
                "search; scheduler_type='%s' applies only under the hard kernel.",
                kernel,
                scheduler_type,
            )
        self.rng_np = np.random.default_rng(
            self.rng.getrandbits(128)
        )  # Derive NumPy seed from Propagator RNG
        # Uniform prior density = 1 / volume (float limits validated above).
        volumes = [hi - lo for lo, hi in float_limits.values()]
        self.prior_density = 1.0 / float(np.prod(volumes))
        # Constant box bounds, used by the proposal build and box-mass correction.
        self._lo = np.array([lim[0] for lim in self.limits.values()], dtype=float)
        self._hi = np.array([lim[1] for lim in self.limits.values()], dtype=float)
        # Mean squared box width; sets the absolute floor of the scale-aware
        # covariance jitter in _build_proposal.
        self._mean_box_sq = float(np.mean((self._hi - self._lo) ** 2))
        self._cache = _IncrementalCache(self._tol_fallback)
        # Live-proposal memo: (effective_tol, archive ids, proposal, cdf,
        # archive refs). See _get_proposal. Never needs invalidation beyond
        # key mismatch — the proposal is a pure function of the key.
        self._proposal_cache: Optional[tuple] = None

        # AMIS snapshot ring buffer (performance state, not algorithmic state).
        if amis_snapshots < 0:
            raise ValueError("amis_snapshots must be >= 0.")
        self._amis_snapshots = amis_snapshots
        if amis_interval is not None and amis_interval < 1:
            raise ValueError("amis_interval must be >= 1.")
        self._amis_interval = amis_interval if amis_interval is not None else max(1, self.k)
        self._snapshots: deque = deque(maxlen=amis_snapshots) if amis_snapshots > 0 else deque(maxlen=1)
        self._calls_since_snapshot = 0

        # One-shot warning flags (set on first emission).
        self._warned_none: bool = False
        self._warned_zero_weight: bool = False

    @staticmethod
    def _check_loss(ind: Individual) -> None:
        """Reject losses outside ABC semantics.

        ABC interprets the loss as a nonnegative discrepancy
        ``rho(simulated, observed)``; every kernel, the quantile schedule and
        the ESS search silently misbehave on signed losses, so fail fast.
        ``inf`` is allowed (a failed simulation is an infinitely bad
        discrepancy); NaN and negative values are not.
        """
        loss = ind.loss
        if loss != loss or loss < 0.0:  # NaN or negative
            raise ValueError(
                f"ABCPMC received an individual with loss={loss}. ABC requires the "
                "loss to be a nonnegative discrepancy rho(simulated, observed); "
                "shift or redefine your distance function so that rho >= 0."
            )

    def _update_cache(self, inds: List[Individual]) -> None:
        """Incrementally update the internal performance cache.

        The append-only fast path is valid only when the previously-cached
        prefix is still intact. The propagator receives the *active* subset of
        the population, which is **not** append-only under the island model:
        migration deactivates emigrants, removing them from arbitrary positions
        in the active list while new arrivals are appended at the end. The old
        prefix is intact iff the element now at index ``cached_len - 1`` is still
        the last individual the cache processed — any deactivation inside the
        prefix shifts or drops it, changing that identity. This O(1) check also
        catches the equal-length content-change case (one emigrant deactivated +
        one immigrant appended), which a length-only test would silently miss.
        On any mismatch (or a shrink) we rebuild; this fires only on migration
        calls, never on the append-only steady state.
        """
        n = len(inds)
        cached_len = self._cache.history_len
        prefix_intact = (
            cached_len >= 1
            and n >= cached_len
            and inds[cached_len - 1] is self._cache._last_ind
        )
        if cached_len < 0 or n < cached_len or not prefix_intact:
            # First call, history shrunk, or non-append-only mutation — rebuild.
            if cached_len >= 0:
                # A genuine rollback (shrink or content change), not the initial
                # build. The AMIS snapshot buffer is a proposal-side performance
                # cache; drop it so snapshots taken in a now-rolled-back future
                # cannot leak into subsequent proposal denominators. (The
                # posterior estimator does not depend on the buffer — see
                # extract_posterior — so this only affects proposal quality.)
                self._snapshots.clear()
                self._calls_since_snapshot = 0
            for ind in inds:
                self._check_loss(ind)
            self._cache.rebuild(inds, self._tol_fallback)
            self.tolerance_scheduler.reset_cache()
        elif n > cached_len:
            # Pure append — process the new tail incrementally.
            for new_ind in inds[cached_len:]:
                self._check_loss(new_ind)
                self._cache.update(new_ind)
        # n == cached_len and prefix intact: genuinely no change.

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
        Return the top-``k`` accepted individuals (by lowest loss, ascending).

        This implements a **quantile-trimmed rejection rule**: from the
        individuals with ``loss < tol`` we keep only the best ``k`` by loss,
        not all of them. At steady state with the archive saturated below
        ``tol`` the two rules coincide; under a tightening schedule the
        top-k rule biases the archive slightly toward the mode relative to
        a strict ``loss < tol`` rule. Documented in §3.2 of the
        asynchronous-ABC paper plan; under smooth kernels the kernel weight
        ``K_eps(rho)`` provides the additional decay.

        Individuals with ``weight == 0`` are not eligible: they contribute
        nothing to the proposal mixture, so they must not occupy archive
        slots (see :meth:`_IncrementalCache.get_archive`).

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history.
        tol : float
            Current effective tolerance.

        Returns
        -------
        List[Individual]
            Top-k accepted, mixture-eligible individuals sorted by loss
            (ascending).
        """
        accepted = self.filter_by_tolerance(inds, tol)
        eligible = [ind for ind in accepted if ind.weight is None or ind.weight > 0.0]
        return sorted(eligible, key=lambda ind: ind.loss)[: self.k]

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

    def _build_proposal(self, archive: List[Individual], effective_tol: float) -> _ArchiveSnapshot:
        """Build the truncated Gaussian-mixture proposal density from an archive.

        Pure function of ``(archive, effective_tol)``. Shared by :meth:`__call__`
        (the live proposal, memoised via :meth:`_get_proposal`) and
        :meth:`extract_posterior` (history replay) so the reconstructed proposal
        sequence matches the one actually used. Computes the effective mixture
        weights ``∝ stored_weight · K_eps(loss)`` in log-space, the perturbation
        Cholesky factor, and the per-component in-box mass.

        Returns
        -------
        _ArchiveSnapshot
            The proposal with all density-evaluation precomputes (whitened
            means, combined per-component log-constants, inverse factor).
        """
        if any(ind.weight is None for ind in archive):
            if not self._warned_none:
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
        d = positions.shape[1]
        # Scale-aware jitter: proportional to the mean marginal variance of the
        # archive so that a tightly-converged posterior (std ~1e-4 → var ~1e-8)
        # is not swamped — an absolute 1e-6*I floor would dominate such an
        # archive and permanently cap proposal sharpness. The box-scale term
        # keeps the matrix factorable for a degenerate archive (all positions
        # identical → zero covariance) at a negligible ~1e-6 · box-width
        # perturbation scale.
        jitter = 1e-9 * max(float(np.trace(cov)), 0.0) / d + 1e-12 * self._mean_box_sq
        cov += jitter * np.eye(d)
        kernel_cov = self.perturbation_scale * cov
        kernel_cov = 0.5 * (kernel_cov + kernel_cov.T)
        try:
            L = np.linalg.cholesky(kernel_cov)
        except np.linalg.LinAlgError:
            kernel_cov += (1e3 * jitter) * np.eye(d)
            L = np.linalg.cholesky(kernel_cov)
        # Per-component in-box mass log Z_j (candidate-independent; computed once
        # per proposal, never per candidate — see _log_box_mass).
        sigma = np.sqrt(np.diag(kernel_cov))
        log_box_mass = _log_box_mass(positions, sigma, self._lo, self._hi)
        return _ArchiveSnapshot(positions, weights, L, log_box_mass)

    def _get_proposal(self, archive: List[Individual], effective_tol: float) -> _ArchiveSnapshot:
        """Memoised :meth:`_build_proposal` for the live proposal.

        The proposal is a pure function of ``(archive, effective_tol)``, and at
        steady state consecutive calls see the identical archive and bandwidth
        — the top-k changes only when a better individual arrives and the
        bandwidth only on a (throttled) scheduler tightening — so rebuilding
        the covariance + Cholesky + box mass every call (~40% of hard-kernel
        call time) is wasted. Cache the last build keyed on the archive
        members' identities and the bandwidth. The cache entry holds the
        archive list itself, so the member ids cannot be recycled while the
        fingerprint is alive; a stale entry can never alias a different
        archive. Parent-selection state (the weight CDF) lives alongside.
        """
        key_ids = tuple(map(id, archive))
        cached = self._proposal_cache
        if (
            cached is not None
            and cached[0] == effective_tol
            and cached[1] == key_ids
        ):
            return cached[2]
        proposal = self._build_proposal(archive, effective_tol)
        cdf = np.cumsum(proposal.weights)
        self._proposal_cache = (effective_tol, key_ids, proposal, cdf, archive)
        return proposal

    def _initial_bandwidth_from_history(self) -> Optional[float]:
        """Data-driven initial bandwidth for ``tol=None`` (smooth kernels).

        Median loss of the first ``k`` finite-loss individuals in generation
        order; ``None`` while fewer than ``k`` exist (still bootstrap). A pure
        function of the history *set* (generation order is total after the
        ``(generation, island, rank)`` tie-break), so it is identical after a
        crash/restart. Only consulted until the first stamped tolerance
        enters the history, after which the running minimum takes over —
        so late-arriving stragglers with early generation order can shift it
        only during the short pre-stamp window.
        """
        finite: List[float] = []
        for ind in self._cache._by_gen:
            if np.isfinite(ind.loss):
                finite.append(float(ind.loss))
                if len(finite) == self.k:
                    return float(np.median(finite))
        return None

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
        tol_from_history = self._cache.tol_from_history  # +inf while unfixed in data-driven mode
        current_tol = tol_from_history
        if self.min_tol is not None:
            current_tol = max(current_tol, self.min_tol)

        # 2. Bootstrap (prior) phase. Hard kernel preserves the classical
        #    rule (need k particles below current threshold); smooth kernels
        #    use the simpler "len(history) >= k" rule since the kernel
        #    handles acceptance smoothly with no hard discontinuity. In
        #    data-driven mode (tol=None) no bandwidth is fixed until a
        #    stamped tolerance exists; derive it from the bootstrap losses,
        #    staying in the prior phase until k finite-loss individuals
        #    exist.
        if self.kernel_name == "hard":
            need_more_particles = self._cache.count_below(current_tol) < self.k
        elif self.tol is None and not np.isfinite(current_tol):
            eps0 = self._initial_bandwidth_from_history()
            if eps0 is None:
                need_more_particles = True
            else:
                current_tol = eps0 if self.min_tol is None else max(eps0, self.min_tol)
                need_more_particles = False
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

        # 6+7. Build (or reuse) the truncated Gaussian-mixture proposal for
        #      this (archive, bandwidth). Memoised — see _get_proposal; shared
        #      with extract_posterior so the reconstructed proposal sequence
        #      matches the one used live.
        proposal = self._get_proposal(archive, effective_tol)
        cdf = self._proposal_cache[3]  # parent-selection weight CDF, cached alongside
        L = proposal.L

        # 7+8. Sample candidate and assign its importance weight.
        #
        # Retry up to _MAX_WEIGHT_RETRIES times if the AMIS denominator
        # underflows the floor: each retry draws a fresh parent + a fresh
        # perturbation. This replaces the prior _MIN_DENOM floor (which
        # produced outlier weights ~prior/1e-12 capable of swamping the
        # mixture). On final exhaustion the candidate keeps weight = 0,
        # which drops out of weighted_covariance and resampling.
        lo = self._lo
        hi = self._hi
        d = proposal.positions.shape[1]
        BATCH = 16
        log_min = np.log(self._MIN_DENOM)

        candidate_pos: np.ndarray
        fallback_to_prior = False
        log_denom = -np.inf
        weight_ok = False

        for retry in range(self._MAX_WEIGHT_RETRIES):
            # Parent selection via the cached weight CDF (inverse-CDF draw —
            # avoids Generator.choice's per-call O(k) validation of p), then
            # batched reject-resample inside the box.
            u = self.rng_np.random() * cdf[-1]
            idx = min(int(np.searchsorted(cdf, u, side="right")), len(archive) - 1)
            parent = archive[idx]
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
                # Kernel cov wider than the box: truncated-Gaussian mixture
                # has effectively no mass inside. Fall back to a uniform
                # prior draw (paper §3.4); the proposal equals the prior,
                # IS weight pi/pi = 1.0. No further retries needed.
                sample = {key: self.rng.uniform(limit[0], limit[1]) for key, limit in self.limits.items()}
                candidate_pos = np.array([sample[k] for k in self.limits], dtype=float)
                logger.debug(
                    "ABCPMC: reject-resample exhausted %d attempts; falling back to uniform-prior draw.",
                    self._MAX_RESAMPLE_ATTEMPTS,
                )
                fallback_to_prior = True
                weight_ok = True
                break

            # Balance-heuristic (AMIS) denominator in log-space. The mixture
            # of mixtures over [current proposal + snapshots] flattens into a
            # SINGLE logsumexp over all component terms (the equal 1/(S+1)
            # outer weights factor out as a constant): one scipy call instead
            # of one per snapshot, and each term set is pure NumPy against
            # the snapshot's precomputed whitened means. logsumexp keeps the
            # mixture stable when Gaussian-kernel PDFs span many decades
            # (typical in d >= 10).
            terms = [proposal.log_component_terms(candidate_pos)]
            for snap in self._snapshots:
                terms.append(snap.log_component_terms(candidate_pos))
            log_denom = float(logsumexp(np.concatenate(terms))) - np.log(len(terms))

            if np.isfinite(log_denom) and log_denom >= log_min:
                weight_ok = True
                break
            # otherwise: retry with a fresh parent + fresh perturbation

        child = Individual(position=candidate_pos, limits=self.limits)
        child.tolerance = effective_tol  # stamped for future history reconstruction

        if fallback_to_prior:
            child.weight = 1.0
        elif weight_ok:
            child.weight = float(self.prior_density * np.exp(-log_denom))
        else:
            # All retries underflowed: the kernel mixture has effectively no
            # support at the sampled neighbourhood. Drop this candidate from
            # the proposal mixture by assigning weight = 0; downstream
            # weighted_covariance and resampling already tolerate zero rows.
            if not self._warned_zero_weight:
                warnings.warn(
                    "ABCPMC: importance weight denominator below floor across "
                    f"{self._MAX_WEIGHT_RETRIES} retries; assigning weight=0.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_zero_weight = True
            child.weight = 0.0

        # 9. Snapshot the current proposal for future AMIS denominators. The
        #    proposal object is immutable by convention (every consumer is
        #    read-only), so the ring buffer shares it with the memo cache —
        #    no copies; all precomputes come along for free.
        self._calls_since_snapshot += 1
        if self._amis_snapshots > 0 and self._calls_since_snapshot >= self._amis_interval:
            self._snapshots.append(proposal)
            self._calls_since_snapshot = 0

        return child

    def _reconstruct_archive(
        self, prefix: List[Individual], eps: float
    ) -> Optional[List[Individual]]:
        """Reconstruct the top-k archive a call would have selected on *prefix*.

        Mirrors :meth:`select_archive` / the cache's ``get_archive``: hard kernel
        keeps the top-k by loss among ``loss < eps``; smooth kernels keep the
        top-k by loss overall; ``weight == 0`` individuals are not eligible
        (same rule as the live selection, so the replay stays faithful).
        Returns ``None`` if fewer than ``k`` candidates exist (the call would
        still have been in the bootstrap phase).
        """
        if self.kernel_name == "hard":
            accepted = [ind for ind in prefix if ind.loss < eps]
        else:
            accepted = prefix
        eligible = [ind for ind in accepted if ind.weight is None or ind.weight > 0.0]
        archive = sorted(eligible, key=lambda i: i.loss)[: self.k]
        if len(archive) < self.k:
            return None
        return archive

    def extract_posterior(
        self,
        inds: List[Individual],
        *,
        eps_final: Optional[float] = None,
        n_proposals: Optional[int] = None,
    ):
        """Retroactive AMIS posterior estimate from the evaluated history.

        This is the estimator the consistency + CLT are stated for. Every
        particle is reweighted against the **current cumulative proposal
        mixture** (the balance heuristic of Cornuet et al. 2012), not against the
        proposal-time mixture frozen onto it during the run. The per-call
        ``Individual.weight`` is never used *as a particle's importance weight*
        here — it enters only through the proposal reconstruction
        (:meth:`_build_proposal` rebuilds each past mixture's component weights
        from the stored per-individual weights, which is what makes the replay
        faithful). The posterior the paper reports is this quantity.

        It is a **pure function of** ``inds`` — the proposal sequence is replayed
        deterministically from history — so the result is identical whether
        computed in a single run or after a crash/restart with an empty snapshot
        buffer. That is what makes "stateless / crash-recoverable" hold for the
        estimator (not merely the archive).

        Runs in the analysis phase (once per run, cost ``O(n · n_proposals · k)``);
        **never call it inside the timed inference loop** — it is deliberately off
        the per-``__call__`` hot path.

        Parameters
        ----------
        inds : List[Individual]
            Full evaluated history.
        eps_final : float, optional
            Bandwidth at which to report the posterior. Defaults to the tightest
            bandwidth reached over the history (the running minimum of the
            stamped tolerances).
        n_proposals : int, optional
            Number of past proposals to reconstruct for the cumulative-mixture
            denominator. Defaults to ``amis_snapshots`` — the fixed-S balance
            heuristic, i.e. the (C4) approximation to the full cumulative mixture.
            Mixture components are weighted by the number of draws they stand in
            for (draw-proportional deterministic mixture): each snapshot covers
            its segment of archive-phase draws, the prior component covers the
            bootstrap draws, floored at half an equal share (``0.5/(m+1)``) so
            the importance weights stay bounded on the whole box.

        Returns
        -------
        positions : np.ndarray, shape (n, d)
            Particle positions in history order.
        weights : np.ndarray, shape (n,)
            Self-normalised importance weights
            ``w_i ∝ π(θ_i) · K_{ε_final}(ρ_i) / q̄(θ_i)`` summing to 1.

        Notes
        -----
        Three approximations scope the consistency claims:

        * **Asynchronous replay.** Under MPI each rank's history interleaves
          its own and received individuals in arrival order, so ``inds`` — and
          therefore the replayed prefixes ``inds[:tau]`` — are rank-dependent
          and need not equal the breeding worker's actual view at call ``tau``.
          "Pure function of the history" (which grounds crash-recoverability)
          is exact; "faithful replay of the proposal sequence" is exact only
          single-rank and an approximation under asynchrony. The balance
          heuristic's consistency requires ``q̄`` to approximate the true draw
          mixture, so the rank-to-rank variation of the reported posterior
          bounds this error empirically (see the order-sensitivity experiment
          in ``tests/benchmarks/bench_abcpmc.py``); proposals evolve slowly —
          the top-k archive is stable to local reordering — which keeps the
          effect small in practice.
        * **Underflow retries.** ``__call__`` redraws a candidate (up to
          ``_MAX_WEIGHT_RETRIES`` times) when the proposal-time denominator
          underflows ``_MIN_DENOM``, conditioning the effective proposal on
          ``q̄ ≥ floor`` without a weight correction. The affected region
          carries proposal density below 1e-12 — negligible proposal mass.
        * **Prior fallback.** When reject-resampling exhausts its attempts,
          the candidate is drawn from the prior and weighted as a pure prior
          draw (weight 1), ignoring the two-stage mixture structure of the
          fallback event. Exact when the fallback probability given the
          covariance is ≈ 0 or ≈ 1 (the typical regimes, since the in-box
          mass makes the 1000-attempt failure probability effectively 0/1);
          approximate in between.
        """
        n = len(inds)
        d = len(self.limits)
        if n == 0:
            return np.empty((0, d)), np.empty(0)

        positions = np.stack([ind.position for ind in inds])  # (n, d)
        losses = np.array([float(ind.loss) for ind in inds])

        # Archive-phase calls: those whose proposal stamped a tolerance. Bootstrap
        # (uniform-prior) draws have tolerance is None.
        archive_idx = [i for i, ind in enumerate(inds) if ind.tolerance is not None]
        if not archive_idx:
            # Never left the bootstrap phase: every draw is a uniform-prior draw,
            # so the posterior is the (flat) prior — equal weights.
            return positions, np.full(n, 1.0 / n)

        if eps_final is None:
            # Tightest bandwidth reached over the history. archive_idx is
            # nonempty here, so at least one stamped tolerance exists (this
            # must not fall back to self.tol, which is None in data-driven
            # mode).
            eps_final = min(inds[i].tolerance for i in archive_idx)

        if n_proposals is None:
            n_proposals = self._amis_snapshots if self._amis_snapshots > 0 else 1
        n_proposals = max(1, min(n_proposals, len(archive_idx)))
        step = max(1, len(archive_idx) // n_proposals)
        picks = list(archive_idx[::step][:n_proposals])
        if archive_idx[-1] not in picks:
            picks.append(archive_idx[-1])

        # Reconstruct each chosen past proposal q_tau from the history prefix the
        # call saw (inds[:tau]) at the bandwidth it stamped (inds[tau].tolerance),
        # reusing the live proposal builder so the reconstruction is faithful.
        snapshots = []
        snap_taus = []
        for tau in picks:
            eps_tau = inds[tau].tolerance
            archive = self._reconstruct_archive(inds[:tau], eps_tau)
            if archive is None:
                continue
            snapshots.append(self._build_proposal(archive, eps_tau))
            snap_taus.append(tau)

        if not snapshots:
            return positions, np.full(n, 1.0 / n)

        # Cumulative proposal mixture q̄, evaluated at EVERY particle (the
        # retroactive step). The deterministic-mixture balance heuristic (Owen
        # & Zhou 2000) wants each component weighted by the fraction of draws
        # it stands in for, so the mixture weights are DRAW-PROPORTIONAL: each
        # snapshot represents the segment of archive-phase draws from its pick
        # up to the next surviving pick (the first also covers the draws before
        # it), and the uniform-prior component represents the bootstrap draws
        # (``tolerance is None``). The prior mass is floored at ``0.5/(m+1)``
        # so q̄ stays bounded away from zero relative to the prior everywhere
        # on the box even for bootstrap-free histories — this keeps the
        # importance weights bounded (w_i <= 2(m+1) · K before normalisation).
        #
        # Evaluated in CHUNKS over the n history points so peak memory is
        # O(chunk · k) instead of O(n · k). Each snapshot's log_mixture_density
        # materialises an (n, k) Mahalanobis-distance matrix; for a fast
        # simulator whose evaluated history reaches many millions of individuals
        # (e.g. Gaussian-mean: ~1e3 sims/s/worker × 48 workers × 300 s ≈ 1e7),
        # the unchunked (n, k) allocation is tens of GB and OOM-kills the rank
        # in the post-run analysis phase. logsumexp is applied per row, so the
        # chunked result is bit-for-bit the same as the unchunked computation.
        m = len(snapshots)
        arch = np.asarray(archive_idx)
        # Segment of each archive-phase draw: index of the last surviving
        # snapshot at or before it (draws before the first pick map to 0).
        seg = np.searchsorted(np.asarray(snap_taus[1:]), arch, side="right")
        counts = np.bincount(seg, minlength=m).astype(float)  # sums to len(arch) >= 1
        n_prior_draws = n - len(archive_idx)
        w_prior = max(n_prior_draws / n, 0.5 / (m + 1))
        mix_w = np.append(counts * ((1.0 - w_prior) / counts.sum()), w_prior)
        log_mix_w = np.log(mix_w)  # all entries > 0 (every segment holds its own pick)
        log_prior = float(np.log(self.prior_density))
        log_kernel = self._kernel_fn.log_weight(losses, eps_final)
        log_w = np.empty(n)
        for start in range(0, n, _EXTRACT_POSTERIOR_CHUNK):
            stop = min(start + _EXTRACT_POSTERIOR_CHUNK, n)
            comp = np.empty((m + 1, stop - start))
            for s, snap in enumerate(snapshots):
                comp[s] = snap.log_mixture_density(positions[start:stop])
            comp[m] = log_prior
            log_qbar = logsumexp(comp + log_mix_w[:, None], axis=0)
            log_w[start:stop] = log_prior + log_kernel[start:stop] - log_qbar
        finite = np.isfinite(log_w)
        if not finite.any():
            return positions, np.full(n, 1.0 / n)
        log_w = np.where(finite, log_w - np.max(log_w[finite]), -np.inf)
        weights = np.where(finite, np.exp(log_w), 0.0)
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(n, 1.0 / n)
        return positions, weights


class EpsilonScheduler(ABC):
    """
    Base class for bandwidth scheduling in ABC-PMC.

    Subclasses must implement ``compute(inds, current_tol)`` — a **pure function**
    of the evaluated history and the current effective tolerance/bandwidth.
    No mutable state should be modified by ``compute``; all scheduling logic
    must be derivable from ``inds`` alone. Subclasses may additionally
    override ``compute_cached(...)`` to exploit append-only history with
    internal performance caches; that cached path must remain equivalent
    to ``compute(...)`` for valid inputs, **modulo bounded staleness** on the
    kernel-aware path: the ESS search runs only every ``bisect_interval``
    cached calls with the last proposal held in between, so ``compute_cached``
    may lag ``compute`` by up to ``bisect_interval`` calls (never violating
    monotonicity — the caller clips).

    Kernel-aware mode
    -----------------
    When ``kernel_aware=True`` and a smooth kernel function is supplied, the
    scheduler selects ε by a bracketed search on the kernel-weighted effective
    sample size rather than from raw loss statistics. ``ess_target`` (default
    0.95) is the **per-step ESS retention ratio**: at each call the scheduler
    proposes the largest ε ≤ ``current_tol`` such that

        ESS(ε) ≈ ess_target · ESS(current_tol),   ESS(ε) = (Σ w·K_ε)² / Σ(w·K_ε)²

    where ``w`` is the stored core importance weight and ``K_ε(ρ)`` is the
    smooth ABC kernel. This is the Del Moral, Doucet & Jasra 2012
    successive-population rule (tighten so each population retains a fixed
    fraction of the previous ESS) restricted to smooth-kernel ABC. Targeting a
    *ratio* of the current ESS — rather than an absolute ``ESS/N`` floor —
    guarantees a tightening direction always exists, so the bandwidth keeps
    sharpening instead of stalling once the absolute ESS drops below the floor.

    ESS(ε) is **not monotone in ε** in general: skewed core weights (a heavy
    importance weight on a high-loss particle — the generic case for
    heavy-tailed IS weights) put a bump in the curve. The search therefore
    scans a geometric grid downward from ``current_tol`` and takes the
    *largest* ε meeting the goal (see :meth:`_bisect_target_ess`), and every
    proposal is floored at ``max_tighten_factor · current_tol`` so a flat ESS
    curve (e.g. tied losses from discrete summary statistics) tightens at a
    bounded rate instead of collapsing.

    Cadence: the retention rule fires on every ``compute`` call — per bred
    individual in the asynchronous steady state, not once per SMC population
    as in the original formulation — so the same ``ess_target`` tightens
    faster with more workers. The acceptance gate (at least
    ``population_size + additional_needed_inds`` individuals with
    ``loss < ε``) throttles this: after tightening, the accepted count drops
    and the bandwidth holds until enough new evaluations accumulate below the
    new ε, so ε equilibrates near that order statistic of the losses and
    ``ess_target`` (with ``max_tighten_factor``) sets only how fast it is
    approached.

    For the hard kernel or ``kernel_aware=False`` the schedulers retain
    their original loss-quantile / acceptance-rate / geometric-decay
    selection rules.
    """

    # Search settings; protected so subclasses can tune them.
    _ESS_GRID_POINTS = 32  # geometric grid resolution for the downward scan
    _BISECT_ITERATIONS = 32
    _BISECT_TOL = 1e-4  # absolute tolerance on relative ESS

    def __init__(
        self,
        initial_tol: float,
        population_size: int,
        additional_needed_inds: int,
        *,
        kernel_aware: bool = False,
        kernel_fn: Optional["_Kernel"] = None,
        ess_target: float = 0.95,
        max_tighten_factor: float = 0.5,
        bisect_interval: Optional[int] = None,
    ):
        if not (0.0 < ess_target <= 1.0):
            raise ValueError("ess_target must be in (0, 1].")
        if not (0.0 < max_tighten_factor < 1.0):
            raise ValueError("max_tighten_factor must be in (0, 1).")
        if bisect_interval is not None and bisect_interval < 1:
            raise ValueError("bisect_interval must be >= 1.")
        self.initial_tol = initial_tol
        self.current_tol = initial_tol  # kept for backward-compat with deprecated update()
        self.population_size = population_size
        self.additional_needed_inds = additional_needed_inds
        self.kernel_aware = kernel_aware
        self.kernel_fn = kernel_fn
        self.ess_target = float(ess_target)
        self.max_tighten_factor = float(max_tighten_factor)
        # Kernel-aware throttle (cached path only): run the O(n_accepted) ESS
        # search every `bisect_interval` calls, hold the last proposal between.
        self._bisect_interval = bisect_interval if bisect_interval is not None else max(1, population_size)
        self._calls_since_bisect = 0
        self._held_eps: Optional[float] = None

    # ------------------------------------------------------------------
    # Kernel-aware target-ESS bisection (Del Moral, Doucet & Jasra 2012)
    # ------------------------------------------------------------------

    def _use_kernel_aware(self) -> bool:
        """Return True when the kernel-aware ESS bisection path is active."""
        if not self.kernel_aware:
            return False
        kfn = self.kernel_fn
        if kfn is None:
            return False
        # Hard kernel is degenerate — ESS is a step function in ε, so
        # bisection isn't meaningful. Fall back to the quantile/etc. rule.
        return getattr(kfn, "name", "") != "hard"

    @staticmethod
    def _relative_ess(weights: np.ndarray, losses: np.ndarray, kfn: "_Kernel", eps: float) -> float:
        """Relative kernel-weighted ESS in [0, 1] at bandwidth *eps*.

        Computed in log-space via logsumexp to remain stable for Gaussian
        kernels in regimes where K_eps(ρ) spans many decades. Returns 0.0
        when the effective weight vector is all zero.
        """
        if weights.size == 0:
            return 0.0
        log_w = np.where(weights > 0.0, np.log(np.where(weights > 0.0, weights, 1.0)), -np.inf)
        log_k = kfn.log_weight(losses, eps)
        log_eff = log_w + log_k
        if not np.isfinite(log_eff).any():
            return 0.0
        log_sum = float(logsumexp(log_eff))
        log_sum_sq = float(logsumexp(2.0 * log_eff))
        # ESS = exp(2 log_sum - log_sum_sq); divide by N for the relative form.
        return float(np.exp(2.0 * log_sum - log_sum_sq) / weights.size)

    def _bisect_target_ess(
        self, weights: np.ndarray, losses: np.ndarray, current_tol: float
    ) -> float:
        """Return the largest ε ≤ current_tol retaining a fixed fraction of the ESS.

        This is the Del Moral, Doucet & Jasra 2012 *successive-population* rule:
        pick ε_new so that ``ESS(w·K_{ε_new}) = α · ESS(w·K_{ε_current})`` with
        ``α = ess_target`` (default 0.95), i.e. each step retains a fixed
        fraction of the current effective sample size. Targeting a *ratio* of
        the current ESS — rather than an absolute ``ESS/N`` floor — means a
        tightening direction always exists, so the bandwidth keeps sharpening
        instead of stalling once the absolute ESS drops below the floor.

        ESS(ε) is **not monotone in ε** in general. With skewed core weights —
        a heavy importance weight sitting on a high-loss particle, which is the
        *generic* case for heavy-tailed IS weights, not a corner — tightening ε
        first kills the heavy component and *raises* the ESS before the usual
        concentration on the lowest-loss particles brings it down again. A
        plain bisection on ``[~0, current_tol]`` brackets across that bump and
        can land orders of magnitude too tight. The search here instead scans a
        geometric grid downward from ``current_tol`` and takes the **first
        (largest) ε whose ESS meets the goal**, then refines by bisection
        inside that single grid interval, where the crossing is bracketed.

        Every proposal is floored at ``max_tighten_factor · current_tol``. If
        no grid point above the floor meets the goal — a flat ESS curve, e.g.
        losses tied at identical values as produced by discrete summary
        statistics — the floor itself is returned: bounded per-call tightening
        instead of the unbounded collapse a "maximal tightening" clip would
        produce.

        The ``/N`` in :meth:`_relative_ess` cancels in the ratio, so the relative
        form is used throughout.
        """
        kfn = self.kernel_fn
        assert kfn is not None  # guarded by _use_kernel_aware
        if current_tol <= 0.0 or weights.size == 0:
            return current_tol

        ess_current = self._relative_ess(weights, losses, kfn, current_tol)
        if ess_current <= 0.0:
            # Degenerate: no effective weight at current_tol (e.g. all archive
            # members already outside a compact kernel's support). Nothing to do.
            return current_tol
        ess_goal = self.ess_target * ess_current

        # Per-call tightening cap: never propose below this floor.
        floor = self.max_tighten_factor * current_tol

        # Downward scan: find the first (largest) grid point meeting the goal.
        # Invariant: ESS(high) > ess_goal (holds at ε = current_tol since
        # ess_goal < ess_current).
        ratio = self.max_tighten_factor ** (1.0 / (self._ESS_GRID_POINTS - 1))
        high = current_tol
        bracket_low = None
        for i in range(1, self._ESS_GRID_POINTS):
            eps = current_tol * ratio**i
            if self._relative_ess(weights, losses, kfn, eps) <= ess_goal:
                bracket_low = eps
                break
            high = eps
        if bracket_low is None:
            # Even the maximal allowed tightening retains more ESS than the
            # goal. Tighten at the bounded rate.
            return floor

        # Refine within the single grid interval [bracket_low, high] where the
        # crossing is bracketed: ESS(high) > ess_goal >= ESS(bracket_low).
        low = bracket_low
        for _ in range(self._BISECT_ITERATIONS):
            mid = 0.5 * (low + high)
            ess_mid = self._relative_ess(weights, losses, kfn, mid)
            if abs(ess_mid - ess_goal) <= self._BISECT_TOL:
                return mid
            if ess_mid > ess_goal:
                # ε too loose — too much ESS retained; tighten.
                high = mid
            else:
                # ε too tight — too little ESS retained; relax.
                low = mid
        return 0.5 * (low + high)

    def _kernel_aware_from_accepted(
        self,
        current_tol: float,
        accepted: List[Individual],
    ) -> float:
        """Shared dispatch point for subclasses' kernel-aware path."""
        if len(accepted) < self.population_size + self.additional_needed_inds:
            return current_tol
        # Use stored core weight; fall back to 1.0 for individuals from an
        # external propagator (matches the same fallback in ABCPMC.__call__).
        weights = np.fromiter(
            (1.0 if ind.weight is None else max(float(ind.weight), 0.0) for ind in accepted),
            dtype=float,
            count=len(accepted),
        )
        losses = np.fromiter(
            (float(ind.loss) for ind in accepted), dtype=float, count=len(accepted)
        )
        return self._bisect_target_ess(weights, losses, current_tol)

    def _kernel_aware_throttled(self, current_tol: float, gather) -> float:
        """Throttled kernel-aware dispatch for the cached path.

        The ESS search costs O(n_accepted) per evaluation (the ``gather``
        materialisation plus up to ``_ESS_GRID_POINTS + _BISECT_ITERATIONS``
        logsumexp passes), which would make the otherwise O(log n) hot path
        linear in the accepted count on *every* call. Run the search only
        every ``bisect_interval`` calls and hold the last proposal in between.
        Staleness only *delays* tightening by at most ``bisect_interval``
        calls — a held (possibly looser) value is safe under the monotone
        clip in ``ABCPMC.__call__`` — and ``gather`` is invoked only on
        refresh. The uncached :meth:`compute` reference path is never
        throttled, so it remains a pure function of history.
        """
        self._calls_since_bisect += 1
        if self._held_eps is not None and self._calls_since_bisect < self._bisect_interval:
            return self._held_eps
        self._calls_since_bisect = 0
        eps = self._kernel_aware_from_accepted(current_tol, gather())
        self._held_eps = eps
        return eps

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
        self._calls_since_bisect = 0
        self._held_eps = None

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
        kernel_fn: Optional["_Kernel"] = None,
        ess_target: float = 0.95,
        max_tighten_factor: float = 0.5,
        bisect_interval: Optional[int] = None,
    ):
        super().__init__(
            initial_tol,
            population_size,
            additional_needed_inds,
            kernel_aware=kernel_aware,
            kernel_fn=kernel_fn,
            ess_target=ess_target,
            max_tighten_factor=max_tighten_factor,
            bisect_interval=bisect_interval,
        )
        if not (0 < percentile < 100):
            raise ValueError("Percentile must be between 0 and 100.")
        self.percentile = percentile

    def compute(self, inds: List[Individual], current_tol: float) -> float:
        accepted = [ind for ind in inds if ind.loss < current_tol]
        if self._use_kernel_aware():
            return self._kernel_aware_from_accepted(current_tol, accepted)
        n = len(accepted)
        if n < self.population_size + self.additional_needed_inds:
            return current_tol
        losses_sorted = sorted(ind.loss for ind in accepted)
        return float(losses_sorted[int(self.percentile / 100.0 * n)])

    def compute_cached(self, inds, current_tol, accepted_by_loss, inds_by_gen, accepted_by_gen):
        if self._use_kernel_aware():
            return self._kernel_aware_throttled(
                current_tol,
                lambda: list(accepted_by_loss[: accepted_by_loss.bisect_key_left(current_tol)]),
            )
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
        kernel_fn: Optional["_Kernel"] = None,
        ess_target: float = 0.95,
        max_tighten_factor: float = 0.5,
        bisect_interval: Optional[int] = None,
    ):
        super().__init__(
            initial_tol,
            population_size,
            additional_needed_inds,
            kernel_aware=kernel_aware,
            kernel_fn=kernel_fn,
            ess_target=ess_target,
            max_tighten_factor=max_tighten_factor,
            bisect_interval=bisect_interval,
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
        if self._use_kernel_aware():
            accepted = [ind for ind in inds if ind.loss < current_tol]
            return self._kernel_aware_from_accepted(current_tol, accepted)
        batch_size = self.population_size + self.additional_needed_inds
        accepted_all = sorted(
            [ind for ind in inds if ind.loss < self.initial_tol],
            key=_gen_order,
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
        if self._use_kernel_aware():
            return self._kernel_aware_throttled(
                current_tol,
                lambda: list(accepted_by_loss[: accepted_by_loss.bisect_key_left(current_tol)]),
            )
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
        super().reset_cache()
        self._cached_consumed = 0
        self._cached_tol = self.initial_tol


class AcceptanceRateScheduler(EpsilonScheduler):
    """
    Adjusts tolerance based on the acceptance rate in a recent sliding window.

    The window size is ``population_size + additional_needed_inds``.  If the
    most recent window contains more accepted individuals than ``high_rate``
    allows, the tolerance is tightened by ``shrink_factor``; otherwise it is
    held unchanged.

    .. note::
        Tolerance expansion is architecturally impossible in this design: the
        monotone guarantee in ``ABCPMC.__call__`` clips any proposed value
        above ``tol_from_history`` to ``tol_from_history``.  ``expand_factor``
        and ``low_rate`` parameters were previously present but removed
        because they never had any effect (expansion was silently discarded;
        rates below ``low_rate`` behaved identically to holding).  If the
        acceptance rate drops too low, the algorithm holds the tolerance and
        waits for more accepted individuals.
    """

    def __init__(
        self,
        initial_tol: float,
        population_size: int,
        additional_needed_inds: int,
        high_rate: float = 0.3,
        shrink_factor: float = 0.9,
        *,
        kernel_aware: bool = False,
        kernel_fn: Optional["_Kernel"] = None,
        ess_target: float = 0.95,
        max_tighten_factor: float = 0.5,
        bisect_interval: Optional[int] = None,
    ):
        super().__init__(
            initial_tol,
            population_size,
            additional_needed_inds,
            kernel_aware=kernel_aware,
            kernel_fn=kernel_fn,
            ess_target=ess_target,
            max_tighten_factor=max_tighten_factor,
            bisect_interval=bisect_interval,
        )
        if not (0 < high_rate < 1):
            raise ValueError("0 < high_rate < 1 required.")
        if not (0 < shrink_factor < 1):
            raise ValueError("0 < shrink_factor < 1 required.")
        self.high_rate = high_rate
        self.shrink_factor = shrink_factor

    def compute(self, inds: List[Individual], current_tol: float) -> float:
        if self._use_kernel_aware():
            accepted = [ind for ind in inds if ind.loss < current_tol]
            return self._kernel_aware_from_accepted(current_tol, accepted)
        window_size = self.population_size + self.additional_needed_inds
        if len(inds) < window_size:
            return current_tol
        recent = sorted(inds, key=_gen_order)[-window_size:]
        accepted = [i for i in recent if i.loss < current_tol]
        rate = len(accepted) / len(recent)
        if rate > self.high_rate:
            return current_tol * self.shrink_factor
        return current_tol

    def compute_cached(self, inds, current_tol, accepted_by_loss, inds_by_gen, accepted_by_gen):
        if self._use_kernel_aware():
            return self._kernel_aware_throttled(
                current_tol,
                lambda: list(accepted_by_loss[: accepted_by_loss.bisect_key_left(current_tol)]),
            )
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
    kernel_fn: Optional["_Kernel"] = None,
    ess_target: float = 0.95,
    max_tighten_factor: float = 0.5,
    bisect_interval: Optional[int] = None,
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
        When True (and ``kernel_fn`` is a smooth kernel), the scheduler
        selects ε by target-ESS bisection (Del Moral, Doucet & Jasra 2012)
        rather than from raw loss statistics. The original selection rule
        is preserved as the fallback for hard kernels and for
        ``kernel_aware=False``.
    kernel_fn : :class:`_Kernel`, optional
        The smooth ABC kernel used by the propagator. Required for
        ``kernel_aware=True`` bisection; ignored otherwise.
    ess_target : float
        Target relative ESS retention ratio for the kernel-aware search
        (default 0.95). Must be in ``(0, 1]``.
    max_tighten_factor : float
        Per-call tightening cap for the kernel-aware search: any proposed ε
        is floored at ``max_tighten_factor · current_tol`` (default 0.5).
        Must be in ``(0, 1)``.
    bisect_interval : int, optional
        Calls between kernel-aware ESS searches on the cached path; the last
        proposal is held in between (bounded staleness). Defaults to
        ``population_size``. Must be ``>= 1``.
    **kwargs
        Additional parameters passed to the scheduler constructor
        (e.g. ``percentile``, ``decay_factor``, ``high_rate``/``shrink_factor``).

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

    common = dict(
        kernel_aware=kernel_aware,
        kernel_fn=kernel_fn,
        ess_target=ess_target,
        max_tighten_factor=max_tighten_factor,
        bisect_interval=bisect_interval,
    )
    if st == EpsilonSchedulerType.QUANTILE:
        return QuantileScheduler(
            initial_tol, population_size, additional_needed_inds, **common, **kwargs
        )
    elif st == EpsilonSchedulerType.GEOMETRIC_DECAY:
        return GeometricDecayScheduler(
            initial_tol, population_size, additional_needed_inds, **common, **kwargs
        )
    elif st == EpsilonSchedulerType.ACCEPTANCE_RATE:
        return AcceptanceRateScheduler(
            initial_tol, population_size, additional_needed_inds, **common, **kwargs
        )
