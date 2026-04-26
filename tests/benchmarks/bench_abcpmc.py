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


def main() -> None:
    print("ABCPMC __call__ microbenchmark")
    print("=" * 70)
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


if __name__ == "__main__":
    main()
