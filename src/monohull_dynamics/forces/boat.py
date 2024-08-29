from monohull_dynamics.forces.foils import FoilData, foil_frame_resultant
from monohull_dynamics.forces.force_utils import flow_at_foil, foil_force_on_boat
import jax.numpy as jnp
import typing

from monohull_dynamics.forces.hull import (
    HullData,
    wave_drag,
    viscous_drag,
    get_hull_coeffs,
)
from monohull_dynamics.forces.polars.polar import init_polar
from monohull_dynamics.forces.sails import (
    SailData,
    init_sail_data,
    sail_frame_resultant,
)

DUMMY_DEBUG_DATA = {
    "forces": {
        "board": jnp.array([0.0, 0.0]),
        "rudder": jnp.array([0.0, 0.0]),
        "sail": jnp.array([0.0, 0.0]),
        "hull": jnp.array([0.0, 0.0]),
    },
    "moments": {
        "board": jnp.array(0.0),
        "rudder": jnp.array(0.0),
        "sail": jnp.array(0.0),
    },
}


class BoatData(typing.NamedTuple):
    hull_data: HullData
    rudder_data: FoilData
    board_data: FoilData
    sail_data: SailData
    centreboard_length: jnp.ndarray
    centreboard_chord: jnp.ndarray
    sail_area: jnp.ndarray
    hull_draft: jnp.ndarray
    rudder_length: jnp.ndarray
    rudder_chord: jnp.ndarray
    beam: jnp.ndarray
    lwl: jnp.ndarray
    length: jnp.ndarray
    sail_coe_dist: jnp.ndarray
    board_offset: jnp.ndarray
    rudder_offset: jnp.ndarray
    mast_offset: jnp.ndarray
    hull_coeffs: dict[str, jnp.ndarray]  # TODO: should these be top level?


def init_boat(
    centreboard_length: jnp.ndarray,
    centreboard_chord: jnp.ndarray,
    sail_area: jnp.ndarray,
    hull_draft: jnp.ndarray,
    rudder_length: jnp.ndarray,
    rudder_chord: jnp.ndarray,
    beam: jnp.ndarray,
    lwl: jnp.ndarray,
    length: jnp.ndarray,
    sail_coe_dist: jnp.ndarray,
) -> BoatData:
    return BoatData(
        hull_data=HullData(
            lwl=lwl,
            beam=beam,
            hull_draft=hull_draft,
        ),
        rudder_data=FoilData(
            length=rudder_length,
            chord=rudder_chord,
            polar=init_polar("n12"),
        ),
        board_data=FoilData(
            length=centreboard_length,
            chord=centreboard_chord,
            polar=init_polar("n12"),
        ),
        sail_data=init_sail_data(area=sail_area),
        centreboard_length=centreboard_length,
        centreboard_chord=centreboard_chord,
        sail_area=sail_area,
        hull_draft=hull_draft,
        rudder_length=rudder_length,
        rudder_chord=rudder_chord,
        beam=beam,
        lwl=lwl,
        length=length,
        sail_coe_dist=sail_coe_dist,
        board_offset=jnp.array([0.0, 0.0]),
        rudder_offset=jnp.array([-length / 2, 0.0]),
        mast_offset=jnp.array([0.0, 0.0]),  # TODO: Update to actual mast offset
        hull_coeffs=get_hull_coeffs(),
    )


def forces_and_moments(
    boat_data: BoatData,
    boat_velocity: jnp.ndarray,
    wind_velocity: jnp.ndarray,
    boat_theta: jnp.ndarray,
    boat_theta_dot: jnp.ndarray,
    sail_angle: jnp.ndarray,
    rudder_angle: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, dict[str, jnp.ndarray]]]:
    tide = jnp.array([0.0, 0.0])

    # flow at board
    board_flow = flow_at_foil(
        flow_velocity=tide,
        boat_velocity=boat_velocity,
        foil_offset=boat_data.board_offset,
        foil_theta=0.0,
        foil_coe=boat_data.board_data.chord / 2,
        boat_theta=boat_theta,
        boat_theta_dot=boat_theta_dot,
    )
    board_local_force = foil_frame_resultant(boat_data.board_data, board_flow)
    board_force, board_moment = foil_force_on_boat(
        foil_force=board_local_force,
        foil_offset=boat_data.board_offset,
        foil_theta=0.0,
        boat_theta=boat_theta,
        foil_coe=0.0,
    )

    # Rudder
    rudder_flow = flow_at_foil(
        flow_velocity=tide,
        boat_velocity=boat_velocity,
        foil_offset=boat_data.rudder_offset,
        foil_theta=rudder_angle,
        foil_coe=boat_data.rudder_data.chord / 2,
        boat_theta=boat_theta,
        boat_theta_dot=boat_theta_dot,
    )
    rudder_local_force = foil_frame_resultant(boat_data.rudder_data, rudder_flow)
    rudder_force, rudder_moment = foil_force_on_boat(
        foil_force=rudder_local_force,
        foil_offset=boat_data.rudder_offset,
        foil_theta=rudder_angle,
        boat_theta=boat_theta,
        foil_coe=boat_data.rudder_data.chord / 2,
    )

    # Sail
    sail_flow = flow_at_foil(
        flow_velocity=wind_velocity,
        boat_velocity=boat_velocity,
        foil_offset=boat_data.mast_offset,
        foil_theta=sail_angle,
        foil_coe=boat_data.sail_coe_dist,
        boat_theta=boat_theta,
        boat_theta_dot=boat_theta_dot,
    )
    sail_local_force = sail_frame_resultant(boat_data.sail_data, sail_flow)
    sail_force, sail_moment = foil_force_on_boat(
        foil_force=sail_local_force,
        foil_offset=boat_data.mast_offset,
        foil_theta=sail_angle,
        boat_theta=boat_theta,
        foil_coe=boat_data.sail_coe_dist,
    )

    hull_drag = wave_drag(
        hull_data=boat_data.hull_data,
        coeffs=boat_data.hull_coeffs,
        velocity=boat_velocity,
    ) + viscous_drag(hull_data=boat_data.hull_data, velocity=boat_velocity)
    # hull_drag = rotate_vector(hull_drag, boat_theta)
    # print(f"velocity: {boat_velocity}, hull_drag: {hull_drag}")
    resultant_force = board_force + rudder_force + sail_force + hull_drag
    resultant_moment = board_moment + rudder_moment + sail_moment

    debug_data = {
        "forces": {
            "board": board_force,
            "rudder": rudder_force,
            "sail": sail_force,
            "hull": hull_drag,
        },
        "moments": {
            "board": board_moment,
            "rudder": rudder_moment,
            "sail": sail_moment,
        },
    }
    return resultant_force, resultant_moment, debug_data


def init_firefly():
    return init_boat(
        centreboard_length=1.05,
        centreboard_chord=0.25,
        sail_area=6.3,
        hull_draft=0.25,
        rudder_length=1.0,
        rudder_chord=0.22,
        beam=1.42,
        lwl=3.58,
        length=3.66,
        sail_coe_dist=1.0,
    )


if __name__ == "__main__":
    import time
    from jax import jit

    boat = init_firefly()
    boat_velocity = jnp.array([1, 0])
    wind_velocity = jnp.array([0, 0])
    boat_theta = 0
    boat_theta_dot = 0
    sail_angle = 0
    rudder_angle = 0

    f, m, _ = forces_and_moments(
        boat,
        boat_velocity,
        wind_velocity,
        boat_theta,
        boat_theta_dot,
        sail_angle,
        rudder_angle,
    )
    print(f)

    t0 = time.time()
    N = 5
    for i in range(N):
        forces_and_moments(
            boat,
            boat_velocity,
            wind_velocity,
            boat_theta,
            boat_theta_dot,
            sail_angle,
            rudder_angle,
        )
    print(f"Time taken per un-jitted step: {(time.time()-t0)/N}")

    j_forces_and_moments = jit(forces_and_moments)
    j_forces_and_moments(
        boat,
        boat_velocity,
        wind_velocity,
        boat_theta,
        boat_theta_dot,
        sail_angle,
        rudder_angle,
    )
    t0 = time.time()
    N = 5000
    for i in range(N):
        j_forces_and_moments(
            boat,
            boat_velocity,
            wind_velocity,
            boat_theta,
            boat_theta_dot,
            sail_angle,
            rudder_angle,
        )
    print(f"Time taken per inner jit step: {(time.time()-t0)/N}")
