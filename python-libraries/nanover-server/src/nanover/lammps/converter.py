import numpy as np
from nanover.trajectory import FrameData

def add_lammps_data_to_frame_data(
        data: FrameData,
        *,
        positions_angstrom: np.ndarray | None = None,
        box_bounds_angstrom: tuple[float, float, float, float, float, float] | None = None,
        include_positions: bool = True,
        include_velocities: bool = False,
        include_forces: bool = False,
    ) -> None:
    #Positions
    if include_positions and positions_angstrom is not None:

        # positions_angstrom is expected to be an (N, 3) array in Angstroms
        positioons_nm = np.asarray(positions_angstrom, dtype=float) * 0.1  # Convert to nanometers
        data.particle_positions = positioons_nm.astype(np.float32, copy=False)

    #Box vectors
    if box_bounds_angstrom is not None:
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds_angstrom
        lx = (xhi - xlo) * 0.1  # Convert to nanometers
        ly = (yhi - ylo) * 0.1
        lz = (zhi - zlo) * 0.1
        data.box_vectors = np.array([[lx, 0.0, 0.0],
                                     [0.0, ly, 0.0],
                                     [0.0, 0.0, lz]], dtype=np.float32)
        
def lammps_to_frame_data(
        *,
        positions_angstrom: np.ndarray | None = None,
        box_bounds_angstrom: tuple[float, float, float, float, float, float] | None = None,
        include_positions: bool = True,
        include_velocities: bool = False,
        include_forces: bool = False,
    ) -> FrameData:
    data = FrameData()
    add_lammps_data_to_frame_data(
        data,
        positions_angstrom=positions_angstrom,
        box_bounds_angstrom=box_bounds_angstrom,
        include_positions=include_positions,
        include_velocities=include_velocities,
        include_forces=include_forces,
    )
    return data
        