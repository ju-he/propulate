"""
Standalone microbenchmark for ``ABCPMC.__call__``.

Not collected by pytest (the file does not match ``test_*.py``).  Run
manually before/after a refactor to capture wall-time and profile
breakdowns:

    python tests/benchmarks/bench_abcpmc.py

The fabricated workload approximates a mid-run snapshot:

* ``d``  ≈ 10 (continuous parameters)
* ``k``  ≈ 100 (archive size)
* ``n_accepted`` ≈ 5000 (inds with loss below the initial tolerance)
* mixed prior-phase + accepted history so all hot-path branches fire
"""

from __future__ import annotations

import cProfile
import os
import pstats
import random
import time
from io import StringIO

import numpy as np

from propulate.population import Individual
from propulate.propagators.abcpmc import ABCPMC


def make_workload(d: int = 10, n_accepted: int = 5000, n_rejected: int = 200, seed: int = 0):
    """Fabricate a representative evaluated history."""
    rng = np.random.default_rng(seed)
    limits = {f"p{i}": (0.0, 1.0) for i in range(d)}
    initial_tol = 600.0

    inds: list[Individual] = []
    # Accepted: losses uniformly spread on (0, initial_tol).
    losses = rng.uniform(1.0, initial_tol - 1.0, size=n_accepted)
    losses.sort()
    for gen, loss in enumerate(losses):
        position = {f"p{i}": float(rng.uniform(0.0, 1.0)) for i in range(d)}
        ind = Individual(position, limits, tolerance=initial_tol, generation=gen)
        ind.loss = float(loss)
        ind.weight = 1.0
        inds.append(ind)

    # Rejected: a handful of high-loss individuals (history of failed proposals).
    for j in range(n_rejected):
        position = {f"p{i}": float(rng.uniform(0.0, 1.0)) for i in range(d)}
        ind = Individual(position, limits, tolerance=initial_tol, generation=n_accepted + j)
        ind.loss = float(rng.uniform(initial_tol + 1.0, 2.0 * initial_tol))
        ind.weight = 1.0
        inds.append(ind)

    return limits, initial_tol, inds


def time_call(prop: ABCPMC, inds: list[Individual], n_calls: int = 200, warmup: int = 20):
    """Run the propagator ``n_calls`` times and report timing stats."""
    for _ in range(warmup):
        prop(inds)

    samples = np.empty(n_calls, dtype=float)
    for i in range(n_calls):
        t0 = time.perf_counter()
        prop(inds)
        samples[i] = time.perf_counter() - t0

    samples_us = samples * 1e6
    print(f"  n_calls       : {n_calls}")
    print(f"  mean / call   : {samples_us.mean():9.1f} µs")
    print(f"  median        : {np.median(samples_us):9.1f} µs")
    print(f"  P95           : {np.percentile(samples_us, 95):9.1f} µs")
    print(f"  P99           : {np.percentile(samples_us, 99):9.1f} µs")
    print(f"  max           : {samples_us.max():9.1f} µs")
    print(f"  total         : {samples.sum() * 1000:9.1f} ms")


def profile_call(prop: ABCPMC, inds: list[Individual], n_calls: int = 50):
    """Run cProfile on the hot path and print the top entries by cumtime."""
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_calls):
        prop(inds)
    profiler.disable()

    stream = StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(25)
    print(stream.getvalue())


def order_sensitivity_experiment(n: int = 2000, k: int = 50, window: int = 20, seed: int = 0) -> None:
    """Quantify the rank-order sensitivity of ``extract_posterior``.

    Under MPI, each rank's population interleaves its own and received
    individuals in arrival order, so the history handed to
    ``extract_posterior`` is rank-dependent. The retroactive reweighting
    replays proposals from *that* order (see the Notes section of
    ``extract_posterior``), so the reported posterior varies slightly from
    rank to rank. This experiment bounds the effect: it drives a
    single-worker gaussian-mean run, re-runs the extraction on a locally
    shuffled copy of the same individuals (permutation within windows of
    ``window`` — a model of message-arrival skew, which reorders nearby
    arrivals but preserves global progress), and reports the discrepancy in
    posterior mean plus the total-variation distance between the two weight
    vectors.
    """
    limits = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    abc = ABCPMC(
        limits, k=k, tol=1.0, kernel="gaussian", scheduler_type="acceptance_rate",
        additional_needed_inds=0, rng=random.Random(seed),
    )
    history: list[Individual] = []
    for i in range(n):
        child = abc(history)
        child.loss = abs(child.position[0] - 0.6)
        child.generation = i
        history.append(child)

    rng = np.random.default_rng(seed)
    shuffled = list(history)
    for start in range(0, n, window):
        stop = min(start + window, n)
        perm = rng.permutation(stop - start)
        shuffled[start:stop] = [shuffled[start + int(p)] for p in perm]

    eps = min(ind.tolerance for ind in history if ind.tolerance is not None)
    pos1, w1 = abc.extract_posterior(history, eps_final=eps)
    pos2, w2 = abc.extract_posterior(shuffled, eps_final=eps)

    mean1 = np.average(pos1, axis=0, weights=w1)
    mean2 = np.average(pos2, axis=0, weights=w2)

    # Match weights particle-by-particle (by identity) for the TV distance.
    index_of = {id(ind): i for i, ind in enumerate(history)}
    w2_aligned = np.empty_like(w2)
    for j, ind in enumerate(shuffled):
        w2_aligned[index_of[id(ind)]] = w2[j]
    tv = 0.5 * float(np.abs(w1 - w2_aligned).sum())

    print(f"  history n                 : {n} (k={k}, local shuffle window={window})")
    print(f"  posterior mean (orig)     : {mean1}")
    print(f"  posterior mean (shuffled) : {mean2}")
    print(f"  |dmean|                   : {np.abs(mean1 - mean2)}")
    print(f"  TV(w_orig, w_shuffled)    : {tv:.4f}")


def main() -> None:
    print("ABCPMC __call__ microbenchmark")
    print("=" * 70)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        print(
            "WARNING: OMP_NUM_THREADS != 1. Multi-threaded BLAS spin-waits on the\n"
            "tiny per-call triangular solves and can inflate timings ~30x; run with\n"
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 for representative numbers.\n"
        )
    limits, initial_tol, inds = make_workload()
    print(
        f"Workload: d={len(limits)}, history={len(inds)}, accepted~5000, rejected~200, "
        f"initial_tol={initial_tol}"
    )

    rng = random.Random(42)
    prop = ABCPMC(limits=limits, k=100, tol=initial_tol, rng=rng)

    print("\n-- timing --")
    time_call(prop, inds, n_calls=200, warmup=20)

    print("\n-- cProfile (top 25 by cumtime) --")
    profile_call(prop, inds, n_calls=50)

    print("\n-- extract_posterior order sensitivity (async-replay approximation) --")
    order_sensitivity_experiment()


if __name__ == "__main__":
    main()
