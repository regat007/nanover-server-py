from lammps import lammps
from pathlib import Path
import numpy as np

from nanover.trajectory import FrameData
from nanover.lammps.converter import lammps_to_frame_data


class LAMMPSSimulation:
    """LAMMPS simulation wrapper implementing the Simulation protocol."""

    def __init__(self, input_script, include_velocities=False, include_forces=False, frame_interval_steps=1):

        self.input_script = input_script
        self.include_velocities = include_velocities
        self.include_forces = include_forces
        self.frame_interval = frame_interval_steps  

        # Set name based on input script filename
        self.name = Path(input_script).stem

        # Initialize LAMMPS instance
        self.lmp = lammps()
        self.lmp.file(self.input_script)

        # Santiy check: get number of atoms
        self._natoms = self.lmp.get_natoms()
        print(f"LAMMPS Simulation initialized with {self._natoms} atoms.")

        self._app_server = None
        self._current_step = 0

    def step(self, n=1):
        self.lmp.command(f"run {int(n)} post no")
        print(f"LAMMPS Simulation stepped by {n} steps.")

    def load(self):
        """Load the simulation. This is called when the runner starts."""
        pass

    def reset(self, app_server):
        """Reset the simulation state and store reference to app_server for frame updates."""
        self._app_server = app_server

    def advance_by_one_step(self):
        """Advance simulation by one frame interval."""
        self.advance_to_next_frame()

    def advance_by_seconds(self, dt: float):
        """Advance simulation to the next frame interval.
        
        Args:
            dt: Time step in seconds (ignored for consistency with OpenMM).
        """
        self.advance_to_next_frame()


    def _get_positions_and_box(self):
        """Retrieve positions and box bounds from LAMMPS."""
        # Get positions
        natoms = self.lmp.get_natoms()
        positions = self.lmp.gather_atoms("x", 1, 3)  # 1 for x, 3 for 3D
        positions_array = np.array(positions).reshape((natoms, 3))

        # Get box bounds
        box = self.lmp.extract_box()
        xlo, xhi = box[0], box[1]
        ylo, yhi = box[2], box[3]
        zlo, zhi = box[4], box[5]
        box_bounds = (xlo, xhi, ylo, yhi, zlo, zhi)

        return positions_array, box_bounds

    def advance_to_next_frame(self):
        """Step the simulation to the next frame output point."""
        try:
            # Run frame_interval steps (advances underlying LAMMPS simulation)
            self.step(self.frame_interval)
            self._current_step += self.frame_interval

            # extract positions and box
            positions, box_bounds = self._get_positions_and_box()

            #create NanoverFrame
            frame = lammps_to_frame_data(
                positions_angstrom=positions,
                box_bounds_angstrom=box_bounds,
                include_positions=True,
                include_velocities=self.include_velocities,
                include_forces=self.include_forces,
            )
            
            # Send a frame (even if minimal) to keep the runner responsive
            if self._app_server is not None:
                self._app_server.frame_publisher.send_frame(frame)
        except Exception as e:
            print(f"Error in advance_to_next_frame: {e}")
            raise

    @classmethod
    def from_data_file(cls, path: str):
        raise NotImplementedError("LAMMPS simulation initialization from data file not yet implemented.")
    