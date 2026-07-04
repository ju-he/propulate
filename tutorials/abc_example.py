import pathlib
import random

import numpy as np
from mpi4py import MPI

from propulate import Propulator
from propulate.propagators.abcpmc import ABCPMC
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import (
    get_function_search_space,
    parse_arguments,
)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(
            "#################################################\n"
            "# PROPULATE: Parallel Propagator of Populations #\n"
            "#################################################\n"
        )

    # Parse command-line arguments.
    config, _ = parse_arguments(comm)

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=config.logging_level,  # logging level
        log_file=f"{config.checkpoint}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    rng = random.Random(config.seed + comm.rank)  # Separate random number generator for optimization.
    function, limits = get_function_search_space(config.function)  # Get callable function + search-space limits.

    # ABC-PMC propagator in posterior mode: smooth Gaussian kernel with a
    # data-driven initial bandwidth (tol=None, the default) and the
    # PMC-optimal perturbation scale (~2x the weighted archive covariance,
    # Beaumont et al. 2009). The default perturbation_scale=0.8 is narrower
    # and optimisation-leaning — use it if you only want a best fit; sweep it
    # for posterior-quality work.
    #
    # NOTE: pass a rank-offset rng so proposal sampling is reproducible under
    # a fixed --seed; without it the proposal stream is seeded from OS
    # entropy and runs are not repeatable.
    propagator = ABCPMC(
        limits=limits,
        kernel="gaussian",
        perturbation_scale=2.0,
        rng=random.Random(config.seed + comm.rank),
    )
    # Set up Propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        propulate_comm=comm,
        generations=config.generations,
        checkpoint_path=config.checkpoint,
    )

    # Run optimization and print summary of results.
    propulator.propulate(logging_interval=config.logging_interval, debug=config.verbosity)
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)

    # Posterior extraction (analysis phase — run once, after the loop): the
    # retroactive AMIS estimator reweights every evaluated individual against
    # the reconstructed proposal mixture. The weight/tolerance stamps travel
    # with the individuals, so any rank's population copy works.
    if comm.rank == 0:
        positions, weights = propagator.extract_posterior(propulator.population)
        posterior_mean = np.average(positions, axis=0, weights=weights)
        ess = 1.0 / float(np.sum(weights**2))
        print(f"Posterior mean: {dict(zip(limits.keys(), (float(x) for x in posterior_mean)))}")
        print(f"Posterior ESS : {ess:.1f} of {len(weights)} particles")
        np.save(f"{config.checkpoint}/abc_posterior_positions.npy", positions)
        np.save(f"{config.checkpoint}/abc_posterior_weights.npy", weights)
