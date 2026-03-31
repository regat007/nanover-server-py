import numpy as np
from nanover.trajectory import FrameData

_ANGSTROM_TO_NM = 0.1

# Topology frame to be sent once
def add_lammps_topology_to_frame_data(
        data: FrameData,
        *,
        particle_count: int | None = None,
        particle_elements: np.ndarray | None = None,
        bond_pairs: np.ndarray | None = None,
        bond_orders: np.ndarray | None = None,
    ) -> None:

    # Particle count
    if particle_count is not None:
        data.particle_count = int(particle_count)

    # Particle elements
    if particle_elements is not None:
        elems = np.asarray(particle_elements).reshape((-1,))
        data.particle_elements = elems.astype(np.uint8, copy=False)

    # Bonds
    if bond_pairs is not None:
        pairs = np.asarray(bond_pairs)
        pairs = pairs.reshape((-1, 2))
        data.bond_pairs = pairs.astype(np.int32, copy=False)

    if bond_orders is not None:
        orders = np.asarray(bond_orders).reshape((-1,))
        data.bond_orders = orders.astype(np.uint32, copy=False)

# Data to be sent each frame
def add_lammps_data_to_frame_data(
        data: FrameData,
        *,
        positions_angstrom: np.ndarray | None = None,
        box_bounds_angstrom: tuple[float, float, float, float, float, float] | None = None,
        include_positions: bool = True,
    ) -> None:

    # Positions
    if include_positions and positions_angstrom is not None:
        positions_nm = np.asarray(positions_angstrom, dtype=float) * _ANGSTROM_TO_NM
        data.particle_positions = positions_nm.astype(np.float32, copy=False)

    # Box vectors (orthorhombic)
    if box_bounds_angstrom is not None:
        arr = np.asarray(box_bounds_angstrom, dtype=float).flatten()

        xlo, xhi, ylo, yhi, zlo, zhi = arr[:6]

        lx = (xhi - xlo) * _ANGSTROM_TO_NM
        ly = (yhi - ylo) * _ANGSTROM_TO_NM
        lz = (zhi - zlo) * _ANGSTROM_TO_NM

        data.box_vectors = np.array(
            [[lx, 0.0, 0.0],
            [0.0, ly, 0.0],
            [0.0, 0.0, lz]],
            dtype=np.float32,
        )

def lammps_to_frame_data(
        *,
        positions_angstrom: np.ndarray | None = None,
        box_bounds_angstrom: tuple[float, float, float, float, float, float] | None = None,
        particle_count: int | None = None,
        particle_elements: np.ndarray | None = None,
        bond_orders: np.ndarray | None = None,
        bond_pairs: np.ndarray | None = None,
        include_positions: bool = True,
    ) -> FrameData:

    data = FrameData()

    add_lammps_data_to_frame_data(
        data,
        positions_angstrom=positions_angstrom,
        box_bounds_angstrom=box_bounds_angstrom,
        include_positions=include_positions,
    )

    add_lammps_topology_to_frame_data(
        data,
        particle_count=particle_count,
        particle_elements=particle_elements,
        bond_pairs=bond_pairs,
        bond_orders=bond_orders,
    )

    return data
