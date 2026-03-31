"""
Manage NanoVer IMD force injection into a LAMMPS simulation via fix external.
"""

import ctypes
import itertools
from typing import Set

import numpy as np
import numpy.typing as npt

from nanover.imd.imd_force import calculate_imd_force, get_sparse_forces
from nanover.imd import ImdStateWrapper
from nanover.trajectory import FrameData

# Conversion factors per LAMMPS unit style.
# Each entry is (pos_to_nm, force_from_kjmol_per_nm):
#   pos_to_nm               : multiply LAMMPS positions by this to get nm
#   force_from_kjmol_per_nm : multiply kJ/(mol·nm) forces by this to get LAMMPS force units
_UNIT_CONVERSIONS: dict[str, tuple[float, float]] = {
    # real  : positions Å, forces kcal/(mol·Å)
    #   1 kJ/(mol·nm) = 0.1 kJ/(mol·Å) = 0.023901 kcal/(mol·Å)
    "real":   (0.1, 0.023901),
    # metal : positions Å, forces eV/Å
    #   1 kJ/(mol·nm) = 1/(96.485 * 10) eV/Å ≈ 1.03643e-3 eV/Å
    "metal":  (0.1, 1.03643e-3),
    # si    : positions m, forces N (per atom)
    #   1 kJ/(mol·nm) = 1e3/(6.02214076e23 * 1e-9) N ≈ 1.66054e-12 N
    "si":     (1e9, 1.66054e-12),
    # nano  : positions nm (already), forces attogram·nm/ns²
    #   1 kJ/(mol·nm) ≈ 0.069477 ag·nm/ns²
    "nano":   (1.0, 0.069477),
}


def get_unit_conversions(lammps_units: str) -> tuple[float, float]:
    """
    Return ``(pos_to_nm, force_from_kjmol_per_nm)`` for the given LAMMPS unit style.

    :param lammps_units: LAMMPS unit style string (e.g. ``"real"``, ``"metal"``).
    :raises ValueError: If the unit style is not supported.
    """
    try:
        return _UNIT_CONVERSIONS[lammps_units]
    except KeyError:
        raise ValueError(
            f"Unsupported LAMMPS unit style '{lammps_units}'. "
            f"Supported styles: {list(_UNIT_CONVERSIONS)}"
        )


def detect_lammps_units(lmp) -> str:
    """
    Attempt to detect the LAMMPS unit style from the simulation object.
    Falls back to ``"real"`` if detection fails.

    :param lmp: A :class:`lammps.lammps` instance.
    :return: LAMMPS unit style string.
    """
    try:
        units = lmp.extract_global("units")
        if isinstance(units, (bytes, bytearray)):
            units = units.decode()
        if isinstance(units, str) and units in _UNIT_CONVERSIONS:
            return units
    except Exception:
        pass
    return "real"


def _build_particle_interaction_index_set(interactions: dict) -> Set[int]:
    """Return the set of 0-based particle indices covered by *interactions*."""
    indices = (interaction.particles for interaction in interactions.values())
    return set(map(int, itertools.chain(*indices)))


class LammpsImdForceManager:
    FIX_ID = "imd_nanover"

    def __init__(
        self,
        lmp,
        imd_state: ImdStateWrapper | None,
        id_to_index: dict[int, int],
        pbc_vectors: np.ndarray | None = None,
        lammps_units: str | None = None,
    ):
        self.lmp = lmp
        self.imd_state = imd_state
        self._id_to_index = id_to_index  # LAMMPS atom ID → NanoVer 0-based index

        # Precomputed array for O(1) vectorised ID→index lookup in _callback.
        # _id_lookup[lammps_id] = nanover_index, or -1 if not present.
        max_id = max(id_to_index.keys(), default=0)
        self._id_lookup = np.full(max_id + 1, -1, dtype=np.int64)
        for lammps_id, nanover_idx in id_to_index.items():
            self._id_lookup[lammps_id] = nanover_idx

        if lammps_units is None:
            lammps_units = detect_lammps_units(lmp)
        _, self._force_from_kjmol_nm = get_unit_conversions(lammps_units)

        # Pre-compute masses eagerly so _get_masses() is never called lazily
        # during update_interactions() — that would invoke gather_atoms() from
        # rank 0 only, causing an MPI collective deadlock on worker ranks.
        self._masses: np.ndarray = self._get_masses(len(id_to_index))

        # Cached forces applied by the callback each timestep, in LAMMPS units,
        # NanoVer (0-based) index order, shape (natoms, 3).
        self._current_lammps_forces: np.ndarray | None = None

        # Forces in NanoVer units (kJ/mol/nm) for frame broadcasting.
        self.user_forces: np.ndarray = np.empty(0)
        self.total_user_energy: float = 0.0
        self._is_force_dirty: bool = False

        self.periodic_box_lengths: np.ndarray | None = None
        if pbc_vectors is not None:
            pbc = np.asarray(pbc_vectors)
            assert np.all(pbc == np.diagflat(np.diag(pbc))), (
                "The periodic box vectors do not correspond to an orthorhombic cell. "
                "Only orthorhombic PBC is currently supported for LAMMPS IMD."
            )
            self.periodic_box_lengths = np.diag(pbc)

        # Register fix external — callback is invoked every timestep inside run
        lmp.command(f"fix {self.FIX_ID} all external pf/callback 1 1")
        lmp.set_fix_external_callback(self.FIX_ID, self._callback)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def broadcast_forces(self, comm, my_rank: int) -> None:
        natoms = len(self._id_to_index)
        buf = np.zeros((natoms, 3), dtype=np.float64)
        if my_rank == 0 and self._current_lammps_forces is not None:
            buf[:] = self._current_lammps_forces
        comm.Bcast(buf, root=0)
        if my_rank != 0:
            if np.any(buf != 0):
                self._current_lammps_forces = buf
                self._is_force_dirty = True
            else:
                self._current_lammps_forces = None
                self._is_force_dirty = False

    def update_interactions(self, positions_nm: np.ndarray) -> None:
        """
        Compute IMD forces from the current frame's molecule-whole positions and
        cache them for injection by :meth:`_callback`.

        Call this once per frame **after** obtaining the molecule-whole positions
        that will be (or were just) broadcast to clients.  Using the same
        positions ensures that the VR client and the simulation agree on where
        atoms are, so dragging gestures produce forces in the correct direction.

        :param positions_nm: Per-atom positions in nm, shape ``(natoms, 3)``,
            in NanoVer (0-based) index order.
        """
        if self.imd_state is None:
            return  # worker rank — forces are received via broadcast_forces

        natoms = len(positions_nm)
        interactions = self.imd_state.active_interactions

        if not interactions:
            if self._is_force_dirty:
                self._current_lammps_forces = None
                # Leave _is_force_dirty = True so the callback zeros fexternal
                # on the next timestep before clearing the flag itself.
                self.user_forces = np.zeros((natoms, 3), dtype=np.float32)
                self.total_user_energy = 0.0
            return

        energy, forces_kjmol = calculate_imd_force(
            positions_nm,
            self._masses,
            interactions.values(),
            self.periodic_box_lengths,
        )

        # Cache forces converted to LAMMPS units for the callback to apply
        self._current_lammps_forces = np.asarray(forces_kjmol) * self._force_from_kjmol_nm

        self._is_force_dirty = True
        self.total_user_energy = float(energy)
        self.user_forces = np.asarray(forces_kjmol, dtype=np.float32)

    def add_to_frame_data(self, frame_data: FrameData) -> None:
        """Write IMD user energy and forces into *frame_data* for broadcasting."""
        frame_data.user_energy = self.total_user_energy
        if self.user_forces.size > 0:
            sparse_indices, sparse_forces = get_sparse_forces(self.user_forces)
            frame_data.user_forces_sparse = sparse_forces
            frame_data.user_forces_index = sparse_indices

    def unfix(self) -> None:
        """Remove ``fix imd_nanover`` from LAMMPS.  Call before re-initialising."""
        try:
            self.lmp.command(f"unfix {self.FIX_ID}")
        except Exception:
            pass  # LAMMPS may have already removed the fix; safe to ignore

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(self, lmp, ntimestep, nlocal, tag, x, fexternal) -> None:
        """
        Called by LAMMPS every timestep via ``fix external pf/callback``.

        Applies the forces cached by :meth:`update_interactions` (computed from
        molecule-whole nm positions that match what the client sees) into
        *fexternal* (LAMMPS units, local-atom order).

        Parameters follow the LAMMPS Python fix-external callback signature:
        - *nlocal*: number of local atoms (== natoms for single-process LAMMPS)
        - *tag*: 1-D int array of global LAMMPS atom IDs, shape ``(nlocal,)``
        - *x*: 2-D float array of positions, shape ``(nlocal, 3)`` — **not used**
          here; force computation uses molecule-whole positions from the frame loop
        - *fexternal*: 2-D float array to accumulate forces into, shape ``(nlocal, 3)``
        """
        if self._current_lammps_forces is None:
            # No active interactions; zero fexternal once after forces were cleared,
            # then clear the flag so we don't zero it every timestep unnecessarily.
            if self._is_force_dirty:
                fexternal[:] = 0.0
                self._is_force_dirty = False
            return

        # Map local atom IDs to NanoVer indices and scatter cached forces
        tag_arr = np.asarray(tag)
        nanover_indices = self._id_lookup[tag_arr]
        valid = nanover_indices >= 0

        fexternal[:] = 0.0
        fexternal[valid] = self._current_lammps_forces[nanover_indices[valid]]

    def _get_masses(self, natoms: int) -> np.ndarray:
        """
        Return per-atom masses in a.m.u., in NanoVer (0-based) atom order.

        Needed by :func:`~nanover.imd.imd_force.calculate_imd_force` for
        mass-weighted interactions.

        Called eagerly from :meth:`__init__` (which runs on all MPI ranks
        inside ``reset()``) so it is never invoked lazily from rank 0 only
        during :meth:`update_interactions`.
        """
        try:
            lmp_types = np.asarray(self.lmp.gather_atoms("type", 0, 1), dtype=np.int32)
            ntypes = int(lmp_types.max(initial=0))
            mass_ptr = self.lmp.extract_atom("mass", 2)
            if mass_ptr is not None and ntypes > 0:
                masses_by_type = np.ctypeslib.as_array(
                    ctypes.cast(mass_ptr, ctypes.POINTER(ctypes.c_double)),
                    shape=(ntypes + 1,),
                )
                return np.array([float(masses_by_type[int(t)]) for t in lmp_types])
        except Exception:
            pass

        # Fallback: per-atom mass (e.g. sphere/granular atom styles use rmass)
        try:
            rmass_ptr = self.lmp.extract_atom("rmass", 2)
            if rmass_ptr is not None:
                rmass = np.ctypeslib.as_array(
                    ctypes.cast(rmass_ptr, ctypes.POINTER(ctypes.c_double)),
                    shape=(natoms,),
                )
                return rmass.copy()
        except Exception:
            pass

        # Final fallback: unit masses (non-mass-weighted interactions still work)
        return np.ones(natoms, dtype=np.float64)
