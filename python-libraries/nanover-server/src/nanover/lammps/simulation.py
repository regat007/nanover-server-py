from lammps import lammps
from pathlib import Path
import numpy as np
import ctypes

from nanover.lammps.converter import lammps_to_frame_data
from nanover.lammps.imd import LammpsImdForceManager, detect_lammps_units, get_unit_conversions

_ANGSTROM_TO_NM = 0.1


class LAMMPSSimulation:
    """LAMMPS simulation wrapper implementing the Simulation protocol."""

    def __init__(
        self,
        input_script,
        include_velocities=False,
        include_forces=False,
        frame_interval_steps=1,
        type_to_atomic_number: dict[int, int] | None = None,
        lammps_units: str | None = None,
        mpi_comm=None,
        generate_bonds: bool = True,
        quiet: bool = False,
    ):
        self.input_script = input_script
        self.include_velocities = include_velocities
        self.include_forces = include_forces
        self.generate_bonds = generate_bonds
        self.frame_interval = frame_interval_steps
        self.type_to_atomic_number = type_to_atomic_number or {}
        self.name = Path(input_script).stem

        # MPI communicator and rank (None / 0 for single-process runs)
        self._mpi_comm = mpi_comm
        self._mpi_rank: int = mpi_comm.Get_rank() if mpi_comm is not None else 0

        cmdargs = ["-screen", "none"] if quiet else []
        self.lmp = lammps(comm=mpi_comm, cmdargs=cmdargs) if mpi_comm is not None else lammps(cmdargs=cmdargs)
        self.lmp.file(self.input_script)

        # Detect or accept LAMMPS unit style (needed for IMD force conversion)
        self.lammps_units: str = lammps_units or detect_lammps_units(self.lmp)

        self._app_server = None
        self._current_step = 0

        self._id_to_index: dict[int, int] | None = None
        self._bond_pairs: np.ndarray | None = None
        self._bond_orders: np.ndarray | None = None
        self._particle_elements: np.ndarray | None = None

        self._imd_force_manager: LammpsImdForceManager | None = None
        self._needs_pre: bool = True  # True after reset() until first step()

    def step(self, n=1):
        if self._needs_pre:
            self.lmp.command(f"run {int(n)} post no")
            self._needs_pre = False
        else:
            self.lmp.command(f"run {int(n)} pre no post no")

    def load(self):
        pass

    def _build_id_to_index_map(self) -> dict[int, int]:
        ids = np.asarray(self.lmp.gather_atoms("id", 0, 1), dtype=np.int64)
        return {int(aid): i for i, aid in enumerate(ids)}

    def reset(self, app_server=None):
        """
        Reset the simulation for (re)start.

        In single-process runs call as ``reset(app_server)``.

        In MPI runs this method must be called on **all** ranks simultaneously
        because it issues collective LAMMPS commands (``fix``/``unfix``).
        Pass ``app_server`` only on rank 0; worker ranks call ``reset()`` with
        no arguments, then enter :meth:`run_mpi_worker`.
        """
        if app_server is not None:
            self._app_server = app_server

        # unfix is a collective LAMMPS command — all ranks must call it together
        if self._imd_force_manager is not None:
            self._imd_force_manager.unfix()
            self._imd_force_manager = None

        # gather_atoms is allgather — all ranks participate and get the full data
        if self._id_to_index is None:
            self._id_to_index = self._build_id_to_index_map()

        if self._particle_elements is None:
            self._particle_elements = self._build_particle_elements()

        if self._bond_pairs is None or self._bond_orders is None:
            bond_orders, bond_pairs = self.extract_bonds()
            if self.generate_bonds:
                natoms = int(self.lmp.get_natoms())
                # Find atoms that have no explicit LAMMPS bond (e.g. zeolite Si/O
                # framework in a simulation that also has an organic guest molecule
                # with explicit bonds).
                bonded_atoms: set[int] = (
                    set(bond_pairs.ravel().tolist()) if len(bond_pairs) > 0 else set()
                )
                if len(bonded_atoms) < natoms:
                    raw_pos = np.asarray(
                        self.lmp.gather_atoms("x", 1, 3), dtype=float
                    ).reshape((natoms, 3))
                    # Convert positions to Å regardless of LAMMPS unit style
                    # so that the bond-distance thresholds (which are in Å) are correct.
                    pos_to_nm, _ = get_unit_conversions(self.lammps_units)
                    to_angstrom = pos_to_nm * 10.0
                    # Wrap raw positions into the primary periodic image before
                    # inference.  LAMMPS does not guarantee that gather_atoms("x")
                    # returns positions in [xlo, xhi); atoms that crossed a periodic
                    # boundary between remap operations may be stored slightly outside
                    # the box.  Without wrapping, such an atom at x ≈ xlo-ε could be
                    # inferred as bonded to a neighbour at x ≈ xlo+bond_length (their
                    # Cartesian distance is within the covalent-radius threshold), yet
                    # after rendering the first atom wraps to x ≈ xhi-ε, making the
                    # bond appear to span the entire simulation cell.
                    _box = self.lmp.extract_box()
                    _lo = np.array([float(_box[0][0]), float(_box[0][1]), float(_box[0][2])], dtype=float)
                    _L  = np.array([float(_box[1][0]) - float(_box[0][0]),
                                    float(_box[1][1]) - float(_box[0][1]),
                                    float(_box[1][2]) - float(_box[0][2])], dtype=float)
                    raw_pos = (raw_pos - _lo) % _L + _lo
                    # Do NOT pass box_lengths: bonds that straddle a periodic boundary
                    # are real but would be rendered by NanoVer as lines spanning the
                    # entire simulation cell.  Omitting PBC means only bonds between
                    # atoms that are physically close in their wrapped positions are
                    # included; a small number of bonds at the box edges will be absent,
                    # which looks far better than diagonal lines crossing the whole box.
                    extra_orders, extra_pairs = self._generate_bonds_from_positions(
                        raw_pos * to_angstrom, self._particle_elements, box_lengths=None
                    )
                    if len(extra_pairs) > 0:
                        # Keep only inferred bonds that involve at least one atom
                        # that had no explicit LAMMPS bond, then merge.
                        unbonded = np.array(
                            sorted(set(range(natoms)) - bonded_atoms), dtype=np.int32
                        )
                        mask = (
                            np.isin(extra_pairs[:, 0], unbonded)
                            | np.isin(extra_pairs[:, 1], unbonded)
                        )
                        extra_pairs = extra_pairs[mask]
                        extra_orders = extra_orders[mask]
                    if len(extra_pairs) > 0:
                        bond_pairs = (
                            np.vstack([bond_pairs, extra_pairs])
                            if len(bond_pairs) > 0
                            else extra_pairs
                        )
                        bond_orders = (
                            np.concatenate([bond_orders, extra_orders])
                            if len(bond_orders) > 0
                            else extra_orders
                        )
            self._bond_pairs = bond_pairs
            self._bond_orders = bond_orders

        positions, box_bounds = self._get_positions_and_box()
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        min_half_L = min(xhi - xlo, yhi - ylo, zhi - zlo) * 0.5

        # Filter bonds: discard any longer than half the shortest box edge.
        # Positions are wrapped to [0, L), so genuine bonds are always short;
        # only PBC-spanning artefacts are long.
        if self._bond_pairs is not None and len(self._bond_pairs) > 0:
            _delta = positions[self._bond_pairs[:, 0]] - positions[self._bond_pairs[:, 1]]
            _keep = np.linalg.norm(_delta, axis=1) < min_half_L
            self._bond_pairs = self._bond_pairs[_keep]
            self._bond_orders = self._bond_orders[_keep]

        pbc_vectors = np.diag([(xhi - xlo) * _ANGSTROM_TO_NM, (yhi - ylo) * _ANGSTROM_TO_NM, (zhi - zlo) * _ANGSTROM_TO_NM])

        # fix external is collective — all ranks register it simultaneously.
        # On worker ranks imd_state is None; those ranks receive forces via
        # broadcast_forces() rather than computing them.
        imd_state = self._app_server.imd if self._app_server is not None else None
        self._imd_force_manager = LammpsImdForceManager(
            lmp=self.lmp,
            imd_state=imd_state,
            id_to_index=self._id_to_index,
            pbc_vectors=pbc_vectors,
            lammps_units=self.lammps_units,
        )
        # New fix registered — next run must use pre yes to incorporate it.
        self._needs_pre = True

        # Rank 0 only: reset frame stream and send initial topology frame
        if self._mpi_rank == 0 and self._app_server is not None:
            natoms = int(self.lmp.get_natoms())
            topology_frame = lammps_to_frame_data(
                positions_angstrom=positions,
                box_bounds_angstrom=box_bounds,
                particle_count=natoms,
                particle_elements=self._particle_elements,
                bond_pairs=self._bond_pairs,
                bond_orders=self._bond_orders,
                include_positions=True,
            )
            self._app_server.frame_publisher.send_clear()
            self._app_server.frame_publisher.send_frame(topology_frame)

    def advance_by_one_step(self):
        self.advance_to_next_frame()

    def advance_by_seconds(self, dt: float):
        self.advance_to_next_frame()

    def _get_positions_and_box(self):
        natoms = int(self.lmp.get_natoms())

        box = self.lmp.extract_box()
        xlo, ylo, zlo = float(box[0][0]), float(box[0][1]), float(box[0][2])
        xhi, yhi, zhi = float(box[1][0]), float(box[1][1]), float(box[1][2])
        box_bounds = (xlo, xhi, ylo, yhi, zlo, zhi)

        positions = np.asarray(
            self.lmp.gather_atoms("x", 1, 3), dtype=float
        ).reshape((natoms, 3))

        L = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float)
        origin = np.array([xlo, ylo, zlo], dtype=float)
        positions = (positions - origin) % L

        return positions, box_bounds

    def _build_particle_elements(self) -> np.ndarray:
        """
        Build NanoVer/OpenMM-style particle_elements (atomic numbers, uint8)
        from LAMMPS per-atom 'type'.
        """
        natoms = int(self.lmp.get_natoms())
        lmp_types = np.asarray(self.lmp.gather_atoms("type", 0, 1), dtype=np.int32).reshape((natoms,))

        # Build a small reference mass table (amu). Extend as needed.
        # (atomic_number, atomic_weight)
        mass_table: list[tuple[int, float]] = [
            (1, 1.00794),    # H
            (6, 12.0107),    # C
            (7, 14.0067),    # N
            (8, 15.9994),    # O
            (9, 18.9984),    # F
            (11, 22.9898),   # Na
            (12, 24.3050),   # Mg
            (13, 26.9815),   # Al
            (14, 28.0855),   # Si
            (15, 30.9738),   # P
            (16, 32.065),    # S
            (17, 35.453),    # Cl
            (19, 39.0983),   # K
            (20, 40.078),    # Ca
            (26, 55.845),    # Fe
            (29, 63.546),    # Cu
            (30, 65.38),     # Zn
            (35, 79.904),    # Br
            (53, 126.904),   # I
        ]
    
        def closest_z_from_mass(m: float, tol: float = 0.6) -> int | None:
            """Return closest atomic number by mass if within tolerance, else None."""
            best_z: int | None = None
            best_diff = float("inf")
            for z, ref_m in mass_table:
                d = abs(m - ref_m)
                if d < best_diff:
                    best_diff = d
                    best_z = z
            if best_z is None or best_diff > tol:
                return None
            return best_z
    
        # Infer number of types from observed types (safe across LAMMPS builds)
        ntypes = int(lmp_types.max(initial=0))
    
        # Try to extract per-type masses from LAMMPS.
        # LAMMPS uses 1-based indexing for per-type arrays (mass[1..ntypes]).
        masses = None
        try:
            mass_ptr = self.lmp.extract_atom("mass", 2)  # pointer to double array
            if mass_ptr is not None and ntypes > 0:
                masses = np.ctypeslib.as_array(
                    ctypes.cast(mass_ptr, ctypes.POINTER(ctypes.c_double)),
                    shape=(ntypes + 1,),
                )
        except Exception:
            masses = None
    
        # Fill in any missing type->Z mapping using masses.
        if masses is not None:
            for t in range(1, ntypes + 1):
                if int(t) in self.type_to_atomic_number:
                    continue  # explicit override wins
                m = float(masses[t])
                z = closest_z_from_mass(m)
                if z is not None:
                    self.type_to_atomic_number[int(t)] = int(z)
    
        # Build per-atom elements array
        out = np.empty(natoms, dtype=np.uint8)
        for i, t in enumerate(lmp_types):
            z = self.type_to_atomic_number.get(int(t))
            out[i] = np.uint8(0 if z is None else max(0, min(255, int(z))))
        return out

    def extract_bonds(self):
        bonds = self.lmp.numpy.gather_bonds()   # [type, id1, id2]

        if self._id_to_index is None:
            self._id_to_index = self._build_id_to_index_map()

        bond_types = bonds[:, 0].astype(np.int32)
        id1 = bonds[:, 1]
        id2 = bonds[:, 2]

        idx1 = np.array([self._id_to_index[int(x)] for x in id1], dtype=np.int32)
        idx2 = np.array([self._id_to_index[int(x)] for x in id2], dtype=np.int32)

        i = np.minimum(idx1, idx2)
        j = np.maximum(idx1, idx2)
        pairs = np.stack([i, j], axis=1)

        return bond_types, pairs

    @staticmethod
    def _generate_bonds_from_positions(
        positions: np.ndarray,
        elements: np.ndarray,
        box_lengths: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Infer bonds from pairwise distances using covalent radii.

        A bond is placed between atoms i and j when their minimum-image distance
        is less than ``1.15 * (cov_radius_i + cov_radius_j)``.  Using single-bond
        covalent radii (Alvarez, Dalton Trans. 2008, 2832) keeps the threshold
        tight enough to distinguish bonded neighbours from non-bonded contacts
        (e.g. Si–O bond at 1.61 Å vs. next Si–Si at 3.07 Å in a zeolite).

        Called as a fallback from :meth:`reset` when LAMMPS reports no explicit
        bonds (e.g. pair-potential-only simulations such as zeolites).

        :param positions: Per-atom positions **in Å**, shape ``(natoms, 3)``.
        :param elements: Per-atom atomic numbers, shape ``(natoms,)``.
        :param box_lengths: Orthorhombic box edge lengths **in Å** for PBC
            minimum-image, or ``None`` for non-periodic simulations.
        :returns: ``(bond_orders, bond_pairs)`` matching :meth:`extract_bonds`.
        """
        # Atomic-number → single-bond covalent radius (Å).
        # Source: Alvarez, Dalton Trans. 2008, 2832.
        # Bond detected when distance < 1.15 * (r1 + r2).
        # Example thresholds: Si–O 2.04 Å (bond 1.61), Si–Si 2.55 Å (next-nbr 3.07).
        _RADII_BY_Z: dict[int, float] = {
            1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
            11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11,
            15: 1.07, 16: 1.05, 17: 1.02, 19: 2.03,
            20: 1.76, 26: 1.52, 29: 1.32, 30: 1.22,
            35: 1.20, 53: 1.39,
        }
        _DEFAULT_RADIUS = 0.80  # Å fallback for unlisted elements
        _BOND_FACTOR = 1.15

        n = len(positions)
        radii = np.array([_RADII_BY_Z.get(int(z), _DEFAULT_RADIUS) for z in elements])

        bond_pairs: list[tuple[int, int]] = []
        for i in range(n):
            diffs = positions[i + 1:] - positions[i]  # (n-i-1, 3)
            if box_lengths is not None:
                diffs -= np.round(diffs / box_lengths) * box_lengths
            dists = np.linalg.norm(diffs, axis=1)
            cutoffs = _BOND_FACTOR * (radii[i] + radii[i + 1:])
            for k in np.where(dists < cutoffs)[0]:
                bond_pairs.append((i, i + 1 + k))

        if not bond_pairs:
            return np.empty(0, dtype=np.int32), np.empty((0, 2), dtype=np.int32)

        pairs = np.array(bond_pairs, dtype=np.int32)
        orders = np.ones(len(pairs), dtype=np.int32)
        return orders, pairs

    def advance_to_next_frame(self):
        # Collective LAMMPS command — all MPI ranks must call this together
        self.step(self.frame_interval)
        self._current_step += self.frame_interval

        # gather_atoms is allgather — all ranks get the full position data
        positions, box_bounds = self._get_positions_and_box()

        # Rank 0 computes IMD forces; all ranks then sync via broadcast
        if self._imd_force_manager is not None:
            if self._mpi_rank == 0:
                self._imd_force_manager.update_interactions(positions_nm=positions * _ANGSTROM_TO_NM)
            if self._mpi_comm is not None:
                self._imd_force_manager.broadcast_forces(self._mpi_comm, self._mpi_rank)

        # Frame building and sending — rank 0 only
        if self._mpi_rank != 0:
            return

        # Per-frame bond filter: suppress bonds longer than half the shortest box edge.
        # Positions are wrapped to [0, L), so a genuine bond is always short; any bond
        # whose endpoints are > L/2 apart is a PBC artefact from an atom that crossed
        # the boundary since the last frame.
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        min_half_L = min(xhi - xlo, yhi - ylo, zhi - zlo) * 0.5
        vis_pairs = self._bond_pairs
        vis_orders = self._bond_orders
        if self._bond_pairs is not None and len(self._bond_pairs) > 0:
            delta = positions[self._bond_pairs[:, 0]] - positions[self._bond_pairs[:, 1]]
            keep = np.linalg.norm(delta, axis=1) < min_half_L
            vis_pairs = self._bond_pairs[keep]
            vis_orders = self._bond_orders[keep]

        frame = lammps_to_frame_data(
            positions_angstrom=positions,
            box_bounds_angstrom=box_bounds,
            bond_pairs=vis_pairs,
            bond_orders=vis_orders,
            include_positions=True,
        )

        if self._imd_force_manager is not None:
            self._imd_force_manager.add_to_frame_data(frame)

        if self._app_server is not None:
            self._app_server.frame_publisher.send_frame(frame)

    def run_mpi_worker(self):
        """
        Block and run the simulation loop for MPI worker ranks (rank != 0).

        Worker ranks must keep in lockstep with rank 0 because LAMMPS's
        ``run`` command and ``gather_atoms`` are collective MPI operations.
        This method drives that loop; rank 0 is driven by the NanoVer runner.

        Call this only after :meth:`reset` (with no ``app_server`` argument)
        on all non-zero ranks::

            if rank == 0:
                with NanoverImdApplication.basic_server(name="LAMMPS") as app:
                    sim.reset(app)
                    while True:
                        sim.advance_by_one_step()
            else:
                sim.reset()
                sim.run_mpi_worker()
        """
        assert self._mpi_rank != 0, "run_mpi_worker() must only be called on MPI ranks > 0"
        while True:
            self.advance_to_next_frame()

