from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from pyparsing import abstractmethod
from scipy.stats import multivariate_normal

from ..population import Individual
from .base import Propagator


class ABC(Propagator):
    """
    This propagator implements approximate bayesian computation.

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
        **kwargs: Dict[str, Union[float, int, str]],  # Additional parameters for the scheduler
    ) -> None:
        """
        Initialize the ABC propagator.

        Parameters
        ----------
        loss_fn : Union[Callable, Generator[float, None, None]]
            The loss function to be minimized.
        limits : Dict
            Search-space limits for each gene.
        perturbation_scale : float
            Scale factor for the Gaussian perturbation covariance.
        k : int
            Number of best individuals to use as the population.
        tol : float
            Distance tolerance for accepting a new sample (default=5.0).
        scheduler_type : str
            The type of tolerance scheduler to use. Options are 'quantile', 'geometric_decay', 'acceptance_rate'.
        additional_needed_inds : int, optional
            How many additional individuals are needed to update the tolerance.
            If None, defaults to k.
        **kwargs : Dict[str, Union[float, int, str]]
            Additional parameters for the scheduler, such as percentile for quantile scheduling,
            decay factor for geometric decay, or low/high rates for acceptance rate scheduling.
        """
        self.limits = limits
        self.perturbation_scale = perturbation_scale
        self.k = k
        self.tol = tol
        if additional_needed_inds is None:
            self.additional_needed_inds = k
        else:
            self.additional_needed_inds = additional_needed_inds
        self.tolerance_scheduler = create_scheduler(scheduler_type, tol, k, self.additional_needed_inds, **kwargs)
        # compute uniform prior density = 1 / volume
        volumes = [(hi - lo) for lo, hi in self.limits.values()]
        self.prior_density = 1.0 / float(np.prod(volumes))

    def filter_by_tolerance(self, inds) -> List[Individual]:
        """
        Return a list of individuals from inds whose loss < tol.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        List[propulate.population.Individual]
            The individuals filtered by loss smaller than tolerance.
        """
        return [ind for ind in inds if ind.loss < self.tol]

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
        # Compute weighted mean
        mean = np.average(values, axis=0, weights=weights)
        w = weights.sum()
        # Denominator for bias correction
        denom = w**2 - np.sum(weights**2)
        if denom <= 0:
            # Cannot compute unbiased covariance with <=1 effective samples
            return np.zeros((values.shape[1], values.shape[1]))
        factor = w / denom
        # Compute weighted scatter
        diffs = values - mean
        cov = np.zeros((values.shape[1], values.shape[1]))
        for i in range(len(weights)):
            cov += weights[i] * np.outer(diffs[i], diffs[i])
        return factor * cov

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Generate a new individual using the ABC algorithm.

        If no parents provided, generate the first individual from the prior.
        Otherwise, take the k newest parents, build a perturbation kernel,
        and sample until the distance is <= tol.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The individual after application of the propagator.
        """
        # Initial generations: population not big enough

        usable_pop = self.filter_by_tolerance(inds)

        if len(usable_pop) < self.k:
            sample = {}
            for key, limit in self.limits.items():
                sample[key] = np.random.uniform(limit[0], limit[1])
            child = Individual(position=sample, limits=self.limits)
            child.weight = 1.0
            return child

        # Update tolerance based on the scheduler for next generation
        self.tol = self.tolerance_scheduler.update(usable_pop, inds)

        # Select k newest individuals
        working_pop = usable_pop[-self.k :]

        weights = np.array([ind.weight for ind in working_pop], dtype=float)
        weights /= weights.sum()  # normalize

        positions = np.stack([ind.position for ind in working_pop])
        cov = self.weighted_covariance(positions, weights)
        cov += 1e-6 * np.eye(positions.shape[1])
        kernel_cov = self.perturbation_scale * cov

        kernel_cov = self.perturbation_scale * cov
        # ensure true SPD
        kernel_cov = 0.5 * (kernel_cov + kernel_cov.T)
        eigs = np.linalg.eigvalsh(kernel_cov)
        if eigs.min() <= 0:
            kernel_cov += (-eigs.min() + 1e-8) * np.eye(positions.shape[1])

        idx = np.random.choice(len(working_pop), p=weights)
        parent = working_pop[idx]

        candidate_pos = parent.position + np.random.multivariate_normal(mean=np.zeros(positions.shape[1]), cov=kernel_cov)

        child = Individual(position=candidate_pos, limits=self.limits)
        child.tolerance = self.tol

        pdfs = []
        for wp in working_pop:
            try:
                p = multivariate_normal.pdf(child.position, mean=wp.position, cov=kernel_cov, allow_singular=True)
            except np.linalg.LinAlgError:
                p = multivariate_normal.pdf(
                    child.position, mean=wp.position, cov=kernel_cov + 1e-6 * np.eye(positions.shape[1]), allow_singular=True
                )
            pdfs.append(p)

        # pdfs = [
        #     multivariate_normal.pdf(child.position, mean=wp.position, cov=kernel_cov)
        #     for wp in working_pop
        # ]
        denom = float(np.dot(weights, pdfs))
        if denom == 0:
            # If denominator is zero, we cannot compute a valid weight
            # This can happen if all pdfs are zero (e.g., child is outside limits)
            denom = 1e-12
        child.weight = self.prior_density / denom

        return child


class ToleranceScheduler(ABC):
    """
    Base class for tolerance scheduling in ABC-PMC.

    All schedulers implement `update(accepted_inds, all_inds)` with a unified signature.
    Subclasses should use only the parameters they need.
    """

    def __init__(self, initial_tol: float, population_size: int, additional_needed_inds: int):
        self.current_tol = initial_tol
        self.population_size = population_size
        self.additional_needed_inds = additional_needed_inds  # how often to update tolerance

    def condition_fulfilled(self, accepted_inds: List[Individual]) -> bool:
        """
        Increment the internal call counter, and only
        return True every self.update_interval calls.
        """
        if len(accepted_inds) >= self.population_size + self.additional_needed_inds:
            self._call_count = 0
            return True
        return False

    @abstractmethod
    def update(self, accepted_inds: Optional[List[Individual]] = None, all_inds: Optional[List[Individual]] = None) -> float:
        """
        Compute and set the next tolerance value.

        Parameters
        ----------
        accepted_inds : Optional[List[Any]]
            Individuals accepted in the last generation (soft generation of size k).
        all_inds : Optional[List[Any]]
            All proposed individuals in the last batch (used for acceptance rate).

        Returns
        -------
        float
            The updated tolerance.
        """
        ...


class QuantileToleranceScheduler(ToleranceScheduler):
    """
    Shrinks tolerance to a given percentile of the previous generation's losses.
    """

    def __init__(self, initial_tol: float, population_size: int, additional_needed_inds: int, percentile: float = 50.0):
        super().__init__(initial_tol, population_size, additional_needed_inds)
        if not (0 < percentile < 100):
            raise ValueError("Percentile must be between 0 and 100.")
        self.percentile = percentile

    def update(self, accepted_inds: List[Individual], all_inds: Optional[List[Individual]] = None) -> float:
        if not self.condition_fulfilled(accepted_inds):
            return self.current_tol
        if accepted_inds is None or len(accepted_inds) == 0:
            raise ValueError("`accepted_inds` must be a non-empty list for quantile scheduling.")

        # gather losses of accepted individuals
        losses = [ind.loss for ind in accepted_inds]
        # compute new tolerance as the specified percentile of losses
        new_tol = float(np.percentile(losses, self.percentile))
        self.current_tol = new_tol
        return self.current_tol


class GeometricDecayToleranceScheduler(ToleranceScheduler):
    """
    Shrinks tolerance by a fixed multiplicative factor each generation.
    """

    def __init__(self, initial_tol: float, population_size: int, additional_needed_inds: int, decay_factor: float = 0.9):
        super().__init__(initial_tol, population_size, additional_needed_inds)
        if not (0 < decay_factor < 1):
            raise ValueError("Decay factor must be between 0 and 1.")
        self.decay_factor = decay_factor 

    def update(self, accepted_inds: List[Individual], all_inds: Optional[List[Individual]] = None) -> float:
        # ignore all_inds
        if accepted_inds is None or len(accepted_inds) == 0:
            raise ValueError("`accepted_inds` must be a non-empty list for quantile scheduling.")
        next_tol = self.decay_factor * self.current_tol
        if len([ind for ind in accepted_inds if ind.loss < next_tol]) >= self.population_size:
            # if enough individuals accepted, shrink tolerance
            self.current_tol = next_tol
        return self.current_tol


class AcceptanceRateToleranceScheduler(ToleranceScheduler):
    """
    Adjusts tolerance based on the acceptance rate computed from the sizes
    of accepted_inds and all_inds.
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

    def update(self, accepted_inds: List[Individual], all_inds: Optional[List[Individual]] = None) -> float:
        if not self.condition_fulfilled(accepted_inds):
            return self.current_tol
        if accepted_inds is None or all_inds is None or len(all_inds) == 0:
            raise ValueError("Both `accepted_inds` and `all_inds` must be provided and non-empty to compute acceptance rate.")

        # compute acceptance rate
        acceptance_rate = len(accepted_inds) / len(all_inds)

        # adjust tolerance based on rate
        if acceptance_rate > self.high_rate:
            # too many accepted, tighten tolerance
            self.current_tol *= self.shrink_factor
        elif acceptance_rate < self.low_rate:
            # too few accepted, relax tolerance
            self.current_tol *= self.expand_factor
        # else, keep tolerance unchanged
        return self.current_tol


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
        One of 'quantile', 'geometric_decay', 'acceptance_rate'.
    initial_tol : float
        Starting tolerance value.
    population_size : int
        Size of the population used for scheduling.
    additional_needed_inds : int
        How many additional individuals are needed to update the tolerance.
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
