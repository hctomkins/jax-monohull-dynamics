from forces.foils import Foil
from forces.force_utils import rotate_vector, flow_at_foil, foil_force_on_boat
from forces.sails import MainSail
from forces.hull import HullDragEstimator
import numpy as np


class Boat:
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
        self.board = Foil(centreboard_length, centreboard_chord, "polars/n12")
        self.rudder = Foil(rudder_length, rudder_chord, "polars/n12")
        self.sail = MainSail(sail_area)
        self.hull_drag = HullDragEstimator(hull_draft, beam, lwl)
        self.virtual_damper_foil = Foil(rudder_length, rudder_length, "polars/n12")

        # offsets assuming boat is pointing right and origin is at the centre of the boat
        self.board_offset = (0, 0)
        self.rudder_offset = (-length / 2, 0)
        self.mast_offset = (length / 4, 0)
        self.sail_coe_dist = sail_coe_dist
        self.damper_offset = (length / 2, 0)

    def forces(
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


if __name__ == "__main__":
    print("RUDDER")
    coe = 0
    foil_theta = 0  # np.pi / 8 # should produce negative clockwise moment
    flow = flow_at_foil(
        flow_velocity=(0, 0),
        boat_velocity=(1, 0),
        foil_offset=(-1, 0),
        foil_theta=foil_theta,
        foil_coe=coe,
        boat_theta=0,
        boat_theta_dot=0,
    )
    # print(flow)
    rudder = Foil(length=1, chord=0.3, polar_dir="polars/n12")
    force_local = rudder.foil_frame_resultant(flow)
    print(force_local, "local force")
    force_global, moment = foil_force_on_boat(
        foil_force=force_local,
        foil_offset=(-1, 0),
        foil_theta=foil_theta,
        boat_theta=0,
        foil_coe=coe,
    )
    print(force_global, moment, "global rudder force and moment")

    print("Main")
    coe = 1.5
    foil_theta = (
        np.pi / 4
    )  # with wind from above, should produce some positive anticlockwise moemnt
    flow = flow_at_foil(
        flow_velocity=(0, -4),  # 8 knts of wind
        boat_velocity=(1, 0),  # 2 knots forward
        foil_offset=(0.5, 0),  # mast is at 0.5, 0
        foil_theta=foil_theta,
        foil_coe=coe,
        boat_theta=0,
        boat_theta_dot=0,
    )
    print(flow, "flow over sail")
    main = MainSail(area=6.3)
    force_local = main.sail_frame_resultant(flow)
    print(force_local, "force in sail space")
    force_global, moment = foil_force_on_boat(
        foil_force=force_local,
        foil_offset=(0.5, 0),
        foil_theta=foil_theta,
        boat_theta=0,
        foil_coe=coe,
    )
    print(force_global, moment, "main force and moment")
