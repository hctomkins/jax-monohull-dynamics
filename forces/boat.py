from forces.foils import Foil
from forces.force_utils import rotate_vector, flow_at_foil, foil_force_on_boat
from forces.sails import MainSail
from forces.hull import HullDragEstimator
import numpy as np

class BoatPhysics:
    def __init__(
        self,
        centreboard_length: float,
        centreboard_chord: float,
        sail_area: float,
        hull_draft: float,
        rudder_length: float,
        rudder_chord: float,
        beam: float,
        lwl: float,
        length: float,
        sail_coe_dist: float,
    ):
        self.board = Foil(centreboard_length, centreboard_chord, "n12")
        self.rudder = Foil(rudder_length, rudder_chord, "n12")
        self.sail = MainSail(sail_area)
        self.hull_drag = HullDragEstimator(hull_draft, beam, lwl)
        self.virtual_damper_foil = Foil(rudder_length, rudder_length, "n12")

        # offsets assuming boat is pointing right and origin is at the centre of the boat
        self.board_offset = (0, 0)
        self.rudder_offset = (-length / 2, 0)
        self.mast_offset = (length / 4, 0)
        self.sail_coe_dist = sail_coe_dist
        self.damper_offset = (length / 2, 0)

    def forces_and_moments(
        self,
        boat_velocity: tuple[float, float],
        wind_velocity: tuple[float, float],
        boat_theta: float,
        boat_theta_dot: float,
        sail_angle: float,
        rudder_angle: float,
    ):
        tide = (0, 0)

        # flow at board
        board_flow = flow_at_foil(
            flow_velocity=tide,
            boat_velocity=boat_velocity,
            foil_offset=self.board_offset,
            foil_theta=0,
            foil_coe=self.board.chord / 2,
            boat_theta=boat_theta,
            boat_theta_dot=boat_theta_dot,
        )
        board_local_force = self.board.foil_frame_resultant(board_flow)
        board_force, board_moment = foil_force_on_boat(
            foil_force=board_local_force,
            foil_offset=self.board_offset,
            foil_theta=0,
            boat_theta=boat_theta,
            foil_coe=0,
        )

        # Rudder
        rudder_flow = flow_at_foil(
            flow_velocity=tide,
            boat_velocity=boat_velocity,
            foil_offset=self.rudder_offset,
            foil_theta=rudder_angle,
            foil_coe=self.rudder.chord / 2,
            boat_theta=boat_theta,
            boat_theta_dot=boat_theta_dot,
        )
        rudder_local_force = self.rudder.foil_frame_resultant(rudder_flow)
        rudder_force, rudder_moment = foil_force_on_boat(
            foil_force=rudder_local_force,
            foil_offset=self.rudder_offset,
            foil_theta=rudder_angle,
            boat_theta=boat_theta,
            foil_coe=0,
        )
        # Damper
        damper_flow = flow_at_foil(
            flow_velocity=tide,
            boat_velocity=boat_velocity,
            foil_offset=self.damper_offset,
            foil_theta=0,
            foil_coe=self.virtual_damper_foil.chord / 2,
            boat_theta=boat_theta,
            boat_theta_dot=boat_theta_dot,
        )
        damper_local_force = self.virtual_damper_foil.foil_frame_resultant(damper_flow)
        _, damper_moment = foil_force_on_boat(
            foil_force=damper_local_force,
            foil_offset=self.damper_offset,
            foil_theta=0,
            boat_theta=boat_theta,
            foil_coe=0,
        )

        # Sail
        sail_flow = flow_at_foil(
            flow_velocity=wind_velocity,
            boat_velocity=boat_velocity,
            foil_offset=self.mast_offset,
            foil_theta=sail_angle,
            foil_coe=self.sail_coe_dist,
            boat_theta=boat_theta,
            boat_theta_dot=boat_theta_dot,
        )
        sail_local_force = self.sail.sail_frame_resultant(sail_flow)
        sail_force, sail_moment = foil_force_on_boat(
            foil_force=sail_local_force,
            foil_offset=self.mast_offset,
            foil_theta=sail_angle,
            boat_theta=boat_theta,
            foil_coe=0,
        )

        boat_speed = np.linalg.norm(boat_velocity)
        hull_drag = self.hull_drag.wave_drag(
            speed=boat_speed
        ) + self.hull_drag.viscous_drag(speed=boat_speed)
        hull_drag = rotate_vector((-hull_drag, 0), boat_theta)

        resultant_force = board_force + rudder_force + sail_force + hull_drag
        resultant_moment = board_moment + rudder_moment + sail_moment
        return resultant_force, resultant_moment


class FireflyPhysics(BoatPhysics):
    def __init__(self):
        super().__init__(
            centreboard_length=1.05,
            centreboard_chord=0.25,
            sail_area=6.3,
            hull_draft=0.25,
            rudder_length=1.5,
            rudder_chord=0.22,
            beam=1.42,
            lwl=3.58,
            length=3.66,
            sail_coe_dist=1,
        )

if __name__ == "__main__":
    boat = FireflyPhysics()
    boat_velocity = (1, 0)
    wind_velocity = (0, 0)
    boat_theta = 0
    boat_theta_dot = 0
    sail_angle = 0
    rudder_angle = 0
    import time
    t0 = time.time()
    N = 10000
    for i in range(N):
        boat.forces_and_moments(boat_velocity, wind_velocity, boat_theta, boat_theta_dot, sail_angle, rudder_angle)
    print(f"Time taken per step: {(time.time()-t0)/N}")