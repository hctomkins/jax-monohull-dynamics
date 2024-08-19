import numpy as np


def moments_about(
    force: tuple[float, float], at: tuple[float, float], about: tuple[float, float]
) -> float:
    """
    Args:
        force: [x, y] force vector
        point: [x, y] point about which to calculate moment
    Returns:
        Anticlockwise moment about point
    """
    r = np.array(at) - np.array(about)
    f = np.array(force)
    return float(np.cross(r, f))


def rotate_vector(vector: tuple[float, float], theta: float) -> tuple[float, float]:
    """
    Args:
        vector: [x, y] vector to rotate
        theta: Anticlockwise rotation angle
    Returns:
        Rotated vector
    """
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    ) @ np.array(vector)


def rotate_vector_about(
    vector: tuple[float, float], theta: float, about: tuple[float, float]
) -> tuple[float, float]:
    """
    Args:
        vector: [x, y] vector to rotate
        theta: Anticlockwise rotation angle
        about: [x, y] point about which to rotate
    Returns:
        Rotated vector
    """
    return rotate_vector(np.array(vector) - np.array(about), theta) + np.array(about)


def coe_offset(
    foil_offset: tuple[float, float], foil_theta: float, coe: float
) -> tuple[float, float]:
    return foil_offset[0] - np.cos(foil_theta) * coe, foil_offset[1] - np.sin(
        foil_theta
    ) * coe


def foil_force_on_boat(
    foil_force: tuple[float, float],
    foil_offset: tuple[float, float],
    foil_theta: float,
    boat_theta: float,
    foil_coe: float,
) -> tuple[tuple[float, float], float]:
    """
    Return force on boat due to foil

    Args:
        foil_force: [x, y] force vector in foil frame
        foil_offset: [x, y] offset of foil from boat origin
        foil_theta: Anticlockwise from boat x axis
        boat_theta: Anticlockwise from +ve x-axis
        foil_coe: Coe length of foil

    Returns: Force on boat, moment about boat origin (measured anticlockwise)

    """
    boat_space_force = rotate_vector(foil_force, foil_theta)
    coe = coe_offset(foil_offset, foil_theta, foil_coe)
    moment = moments_about(boat_space_force, at=coe, about=(0.0, 0.0))
    world_space_force = rotate_vector(boat_space_force, boat_theta)
    return world_space_force, moment


def flow_at_foil(
    flow_velocity: tuple[float, float],
    boat_velocity: tuple[float, float],
    foil_offset: tuple[float, float],
    foil_theta: float,
    foil_coe: float,
    boat_theta: float,
    boat_theta_dot: float,
) -> tuple[float, float]:
    coe = coe_offset(foil_offset, foil_theta, foil_coe)
    global_flow = np.array(flow_velocity) - np.array(boat_velocity)
    boat_space_directional_flow = rotate_vector(global_flow, -boat_theta)
    boat_space_rotational_flow = -np.cross([0, 0, boat_theta_dot], [coe[0], coe[1], 0])[
        0:2
    ]
    boat_space_total_flow = boat_space_directional_flow + boat_space_rotational_flow
    foil_flow = rotate_vector(boat_space_total_flow, -foil_theta)
    return foil_flow


if __name__ == "__main__":
    flow = flow_at_foil(
        flow_velocity=(0, 0),
        boat_velocity=(1, 0),
        foil_offset=(-1, 0),
        foil_theta=np.pi / 2,
        foil_coe=1,
        boat_theta=45,
        boat_theta_dot=1,
    )
    # print(flow)
    # force = Foil
    # print(foil_force_on_boat(
    #     foil_force=(1, 0),
    #     foil_offset=(-1, 0),
    #     foil_theta=np.pi/4,
    #     foil_chord=1
    # ))
    # print(rotate_vector([1, 1], np.pi/2))
