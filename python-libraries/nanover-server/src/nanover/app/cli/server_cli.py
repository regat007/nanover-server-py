"""
Command line interface for nanover.omni.
"""

import ctypes
import logging
import os
import shutil
import ssl
import subprocess
import sys
import time
import textwrap
import argparse
from contextlib import contextmanager
from glob import glob
from typing import Iterable

# ---------------------------------------------------------------------------
# Windows DLL pre-loading
#
# LAMMPS (MSMPI build) is compiled with MinGW-GCC and ships its own copies of
# libgomp-1.dll, libgcc_s_seh-1.dll, libstdc++-6.dll etc.  The conda
# environment also contains different versions of these same DLLs (in
# Library/bin).  If any other nanover dependency (e.g. OpenMM) loads the conda
# versions first, liblammps.dll will fail to load with WinError 127 because
# the already-resident DLL is the wrong version.
#
# Fix: register the LAMMPS bin and MS-MPI bin via os.add_dll_directory() and
# pre-load the GCC runtime DLLs from the LAMMPS bin *before* importing OpenMM
# or any other conda package that might pull in conflicting versions.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "4"


_lammps_dll_path = os.environ.get("LAMMPSDLLPATH") or os.environ.get("LAMMPSHOME")
if _lammps_dll_path:
    _lammps_bin = (
        _lammps_dll_path
        if os.path.isfile(os.path.join(_lammps_dll_path, "liblammps.dll"))
        else os.path.join(_lammps_dll_path, "bin")
    )
    if hasattr(os, "add_dll_directory") and os.path.isdir(_lammps_bin):
        os.add_dll_directory(_lammps_bin)
    # Pre-load GCC runtime DLLs from the LAMMPS bin so they take precedence
    # over any conda-env copies that OpenMM or numpy might otherwise load first.
    for _dll_name in ("libgomp-1.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll",
                      "libwinpthread-1.dll"):
        _dll_full = os.path.join(_lammps_bin, _dll_name)
        if os.path.isfile(_dll_full):
            try:
                ctypes.CDLL(_dll_full)
            except OSError:
                pass

_msmpi_bin = os.environ.get("MSMPI_BIN") or r"C:\Program Files\Microsoft MPI\Bin"
if hasattr(os, "add_dll_directory") and os.path.isdir(_msmpi_bin):
    os.add_dll_directory(_msmpi_bin)

# Pre-load msmpi.dll by importing mpi4py before any other package that might
# otherwise trigger liblammps.dll to load without msmpi.dll already resident.
try:
    from mpi4py import MPI as _MPI_preload  # noqa: F401
except ImportError:
    pass

from nanover.omni import OmniRunner
from nanover.mdanalysis import UniverseSimulation
from nanover.openmm import OpenMMSimulation
from nanover.recording import PlaybackSimulation
from nanover.utilities.cli import suppress_keyboard_interrupt_as_cancellation
from nanover.websocket.discovery import DiscoveryClient
from nanover.websocket.record import record_from_runner

try:
    from nanover.lammps import LAMMPSSimulation
except Exception as e:
    print(f"Could not import LAMMPS module: {e}")
    LAMMPSSimulation = None


def _detect_mpi():
    """
    Attempt to detect an MPI environment via mpi4py.

    :return: ``(comm, rank)`` — the MPI communicator and this process's rank.
        Returns ``(None, 0)`` when mpi4py is not installed or MPI is not active.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank()
    except ImportError:
        return None, 0


def _is_running_under_mpi() -> bool:
    """
    Return True if this process was launched by an MPI launcher.

    Checks environment variables set by common MPI implementations so that the
    auto-relaunch logic is never triggered inside an already-MPI job.
    """
    mpi_env_vars = (
        "PMI_RANK",               # MPICH / Intel MPI
        "OMPI_COMM_WORLD_RANK",   # Open MPI
        "MV2_COMM_WORLD_RANK",    # MVAPICH2
        "SLURM_PROCID",           # SLURM + srun
        "PMI_ID",                 # Some MPICH variants
        "MPIEXEC_HOSTNAME",       # MS-MPI (Windows)
    )
    return any(var in os.environ for var in mpi_env_vars)


def _find_mpi_launcher() -> str | None:
    """Return the path to ``mpiexec`` or ``mpirun``, whichever is found first."""
    for launcher in ("mpiexec", "mpirun"):
        if shutil.which(launcher):
            return launcher
    return None


def _relaunch_with_mpi(launcher: str, n_procs: int) -> int:
    """
    Re-launch the current command under *launcher* with *n_procs* processes.

    Blocks until the child job finishes and returns its exit code.
    stdout/stderr pass through to the terminal unchanged.
    """
    # Use sys.executable to ensure child processes inherit the same Python
    # environment (conda env, virtual env, etc.) — important on Windows where
    # mpiexec may not inherit PATH/DLL search paths from the parent process.
    cmd = [launcher, "-np", str(n_procs), sys.executable, "-m", "nanover.app.cli.server_cli"] + sys.argv[1:]
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0


def handle_user_arguments(args=None) -> argparse.Namespace:
    """
    Parse the arguments from the command line.

    :return: The namespace of arguments read from the command line.
    """
    description = textwrap.dedent(
        """\
    Multi-simulation server for NanoVer
    """
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--lammps",
        dest="lammps_entries",
        action="append",
        nargs="+",
        default=[],
        metavar="PATH [FRAME_INTERVAL]",
        help=(
            "Simulation to run via LAMMPS. Optionally append an integer frame "
            "interval to control how many LAMMPS steps are run per frame sent "
            "to clients (default: 1). Example: --lammps sim.in 20"
        ),
    )

    parser.add_argument(
        "--omm",
        dest="openmm_xml_entries",
        action="append",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Simulations to run via OpenMM (XML format)",
    )

    parser.add_argument(
        "--mda",
        dest="mdanalysis_entries",
        action="append",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Structures to load via MDanalysis",
    )

    parser.add_argument(
        "--playback",
        dest="recording_entries",
        action="append",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Recorded sessions to playback",
    )

    parser.add_argument(
        "--record",
        dest="record_to_path",
        nargs="?",
        default=None,
        const="",
        metavar="PATH",
        help="Record trajectory and state to files.",
    )

    parser.add_argument(
        "--lammps-quiet",
        dest="lammps_quiet",
        action="store_true",
        default=False,
        help="Suppress LAMMPS console output (thermo, timing lines) when running LAMMPS simulations.",
    )

    parser.add_argument(
        "-q",
        "--include-velocities",
        action="store_true",
        default=False,
        help="Optionally include the particle velocities in the frame data for OMM and ASE simulations.",
    )
    parser.add_argument(
        "-k",
        "--include-forces",
        action="store_true",
        default=False,
        help="Optionally include the particle forces in the frame data for OMM and ASE simulations.",
    )

    parser.add_argument(
        "-s",
        "--ssl",
        nargs=3,
        metavar=("CERTFILE", "KEYFILE", "PASSWORD"),
        help="Offer SSL in addition to insecure connections.",
    )

    parser.add_argument(
        "--cloud-discovery",
        dest="cloud_discovery_host",
        metavar="ENDPOINT",
        help="Advertise server on cloud discovery.",
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Give a friendly name to the server.",
    )
    parser.add_argument("-p", "--port", type=int, default=None)
    parser.add_argument("-a", "--address", default=None)

    parser.add_argument(
        "--np",
        dest="n_procs",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Number of MPI processes for LAMMPS simulations "
            "(default: 4). "
            "The server auto-launches with mpiexec/mpirun if not already "
            "running under MPI. Requires mpi4py and mpiexec/mpirun in PATH."
        ),
    )

    parser.add_argument(
        "--omp-num-threads",
        dest="omp_num_threads",
        type=int,
        default=None,
        metavar="N",
        help="Set OMP_NUM_THREADS for OpenMP parallelism (default: 4).",
    )

    arguments = parser.parse_args(args)
    return arguments


def get_all_paths(path_sets: Iterable[Iterable[str]]):
    for paths in path_sets:
        for pattern in paths:
            files = list(glob(pattern, recursive=True))
            if files:
                yield from files
            else:
                logging.warning(f'Path "{pattern}" yielded 0 files.')


@contextmanager
def initialise_runner(arguments: argparse.Namespace, mpi_comm=None):
    with OmniRunner.with_basic_server(
        name=arguments.name,
        address=arguments.address,
        port=arguments.port,
        ssl=initialise_ssl(arguments),
    ) as runner:
        for path in get_all_paths(arguments.recording_entries):
            runner.add_simulation(PlaybackSimulation.from_path(path=path))

        for path in get_all_paths(arguments.openmm_xml_entries):
            simulation = OpenMMSimulation.from_xml_path(path)
            simulation.include_velocities = arguments.include_velocities
            simulation.include_forces = arguments.include_forces
            runner.add_simulation(simulation)

        for path in get_all_paths(arguments.mdanalysis_entries):
            simulation = UniverseSimulation.from_path(path=path)
            runner.add_simulation(simulation)

        for entry in arguments.lammps_entries:
            if LAMMPSSimulation is None:
                print("Skipping --lammps entry: LAMMPS module failed to import (see error above).")
                continue
            # Entry is a list of tokens from nargs="+".
            # Optional trailing integer is the frame interval: --lammps sim.in 20
            *path_tokens, last = entry
            if last.isdigit():
                paths = path_tokens
                frame_interval = int(last)
            else:
                paths = entry
                frame_interval = 1
            for pattern in paths:
                for path in (glob(pattern, recursive=True) or [pattern]):
                    try:
                        simulation = LAMMPSSimulation(
                            input_script=path,
                            mpi_comm=mpi_comm,
                            frame_interval_steps=frame_interval,
                            quiet=arguments.lammps_quiet,
                        )
                        runner.add_simulation(simulation)
                        natoms = int(simulation.lmp.get_natoms())
                        nprocs = mpi_comm.Get_size() if mpi_comm is not None else 1
                        print(
                            f"LAMMPS simulation with {natoms} atoms loaded from {path} "
                            f"(frame interval: {frame_interval}, {nprocs} MPI rank(s))"
                        )
                    except NotImplementedError as e:
                        print(f"LAMMPS simulation not yet implemented: {e}")
                    except Exception as e:
                        print(f"Error initializing LAMMPS simulation from {path}: {e}")

        if arguments.record_to_path is not None:
            stem = arguments.record_to_path
            if stem == "":
                timestamp = time.strftime("%Y-%m-%d-%H%M-%S", time.gmtime())
                stem = f"omni-recording-{timestamp}"

            out_path = f"{stem}.nanover.zip"
            print(f"Recording to {out_path}")

            record_from_runner(runner, out_path)

        if arguments.cloud_discovery_host:
            with DiscoveryClient.advertise_server(
                arguments.cloud_discovery_host, app_server=runner.app_server
            ):
                yield runner
        else:
            yield runner


def initialise_ssl(arguments: argparse.Namespace):
    if arguments.ssl is None:
        return None

    certfile, keyfile, password = arguments.ssl
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_verify_locations(certfile)
    ssl_context.load_cert_chain(certfile, keyfile=keyfile, password=password)

    return ssl_context


def main():
    """
    Entry point for the command line.

    **MPI / parallel LAMMPS**: launch with ``mpirun``/``mpiexec`` and ensure
    ``mpi4py`` is installed.  Rank 0 runs the NanoVer server as normal; all
    other ranks participate in the LAMMPS computation loop::

        mpirun -np 4 nanover-server --lammps simulation.in
    """
    # use the nice logger formatting if available
    try:
        from rich.logging import RichHandler

        logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)])
    except ImportError:
        logging.basicConfig()
    logging.captureWarnings(True)

    arguments = handle_user_arguments()

    if arguments.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(arguments.omp_num_threads)

    # Auto-relaunch under MPI when LAMMPS simulations are requested and we are
    # not already inside an MPI job.  This means a plain `nanover-server --lammps`
    # call automatically uses all available cores without any extra user steps.
    if arguments.lammps_entries and not _is_running_under_mpi():
        n_procs = arguments.n_procs
        if n_procs > 1:
            launcher = _find_mpi_launcher()
            if launcher is None:
                print("Warning: mpiexec/mpirun not found in PATH — running single-process.")
            else:
                try:
                    import mpi4py  # noqa: F401
                except ImportError:
                    print(
                        "Warning: mpi4py not installed — running single-process. "
                        "Install with: pip install mpi4py"
                    )
                else:
                    print(f"Launching LAMMPS with {n_procs} MPI processes via {launcher}.")
                    sys.exit(_relaunch_with_mpi(launcher, n_procs))

    mpi_comm, mpi_rank = _detect_mpi()

    # Worker ranks (rank > 0) participate in LAMMPS collective operations
    # but do not run a NanoVer server.  They stay in lockstep with rank 0
    # because lammps.command() and gather_atoms() are MPI collective calls.
    if mpi_rank > 0:
        if not arguments.lammps_entries:
            return
        entry = arguments.lammps_entries[0]
        *path_tokens, last = entry
        if last.isdigit():
            first_path = next(iter(glob(path_tokens[0], recursive=True) or [path_tokens[0]]), None)
            frame_interval = int(last)
        else:
            first_path = next(iter(glob(entry[0], recursive=True) or [entry[0]]), None)
            frame_interval = 1
        if not first_path:
            return
        try:
            sim = LAMMPSSimulation(input_script=first_path, mpi_comm=mpi_comm, frame_interval_steps=frame_interval, quiet=arguments.lammps_quiet)
            sim.reset()           # participates in rank 0's collective ops
            sim.run_mpi_worker()  # loops in lockstep with rank 0's runner
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"MPI worker rank {mpi_rank} error: {e}", flush=True)
        return

    # Rank 0: normal NanoVer server
    with suppress_keyboard_interrupt_as_cancellation() as cancellation:
        with initialise_runner(arguments, mpi_comm=mpi_comm) as runner:
            if len(runner.simulations) > 0:
                runner.load(0)

            runner.print_basic_info()
            cancellation.wait_cancellation()
            print("Closing due to KeyboardInterrupt.")


def deprecated():
    import warnings

    warnings.warn(
        "use `nanover-server` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    main()
