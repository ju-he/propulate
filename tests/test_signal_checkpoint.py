import os
import pathlib
import pickle
import random
import signal
import time

import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.utils import get_default_propagator
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.mark.timeout(30)
def test_emergency_checkpoint_on_signal(mpi_tmp_path: pathlib.Path) -> None:
    """
    Verify that sending SIGINT triggers an emergency checkpoint written by island rank 0
    without interrupting the process, and that backup rotation occurs on subsequent signals.
    """
    rng = random.Random(123 + MPI.COMM_WORLD.rank)
    func, limits = get_function_search_space("sphere")

    propagator = get_default_propagator(pop_size=2, limits=limits, rng=rng)

    prop = Propulator(
        loss_fn=func,
        propagator=propagator,
        rng=rng,
        generations=10,
        checkpoint_path=mpi_tmp_path,
    )

    # Prepare expected checkpoint paths for island 0
    ckpt = mpi_tmp_path / "island_0_ckpt.pickle"
    bkp = ckpt.with_suffix(".bkp")
    # Clean any leftovers if present
    if ckpt.exists() and MPI.COMM_WORLD.rank == 0:
        ckpt.unlink()
    if bkp.exists() and MPI.COMM_WORLD.rank == 0:
        bkp.unlink()
    MPI.COMM_WORLD.barrier()

    # Trigger the emergency checkpoint via SIGINT
    os.kill(os.getpid(), signal.SIGINT)
    MPI.COMM_WORLD.barrier()

    # Only island rank 0 is expected to write the file
    if prop.island_comm.rank == 0:
        assert ckpt.exists(), "Emergency checkpoint file was not created by island rank 0 on SIGINT"
        # Validate it can be unpickled and is a list (population)
        with open(ckpt, "rb") as f:
            population = pickle.load(f)
        assert isinstance(population, list)

        # Send another signal to exercise .bkp rotation
        os.kill(os.getpid(), signal.SIGINT)
        # Give a tiny moment for filesystem flush (best-effort)
        time.sleep(0.05)
        assert ckpt.exists()
        assert bkp.exists(), "Backup .bkp file not found after second SIGINT"
    else:
        # Non-zero island ranks should not write the checkpoint.
        # No assertion here; rank 0's existence check covers success.
        pass
