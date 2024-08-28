import jax.numpy as jnp
import numpy as np


def moments_about(
    force: jnp.ndarray, at: jnp.ndarray, about: jnp.ndarray
) -> jnp.ndarray:
    """
    Args:
        force: [x, y] force vector
        about: [x, y] point about which to calculate moment
        at: [x, y] point force is actuing
    Returns:
        Anticlockwise moment about point
    """
    r = at - about
    return jnp.cross(r, force)


def rotate_vector(vector: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        vector: [x, y] vector to rotate
        theta: Anticlockwise rotation angle
    Returns:
        Rotated vector
    """
    return (
        jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
        @ vector
    )


def rotate_vector_about(
    vector: jnp.ndarray, theta: jnp.ndarray, about: jnp.ndarray
) -> jnp.ndarray:
    """
    Args:
        vector: [x, y] vector to rotate
        theta: Anticlockwise rotation angle
        about: [x, y] point about which to rotate
    Returns:
        Rotated vector
    """
    return rotate_vector(vector - about, theta) + about


def coe_offset(
    foil_offset: jnp.ndarray, foil_theta: jnp.ndarray, coe: jnp.ndarray
) -> jnp.ndarray:
    """
    Given the offset of the foil from the boat origin, the angle of the foil from the boat x-axis and
    the coe length of the foil, return the coe offset from the boat origin.

    The foil offset is the boat->foil vector in the boat frame.
    The coe is a positive length, and foil_theta assumes the foil is pointing right, so we SUBTRACT x coe distance
    """
    # foil offset [2] [x,y]
    _coe_offset = jnp.zeros_like(foil_offset, dtype=jnp.float32)
    coe_vector = jnp.ones_like(foil_offset, dtype=jnp.float32) * coe
    coe_vector = coe_vector.at[0].multiply(jnp.cos(foil_theta))
    coe_vector = coe_vector.at[1].multiply(jnp.sin(foil_theta))
    return foil_offset - coe_vector


def foil_force_on_boat(
    foil_force: jnp.ndarray,
    foil_offset: jnp.ndarray,
    foil_theta: jnp.ndarray,
    boat_theta: jnp.ndarray,
    foil_coe: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    moment = moments_about(boat_space_force, at=coe, about=jnp.array([0.0, 0.0]))
    world_space_force = rotate_vector(boat_space_force, boat_theta)
    return world_space_force, moment


def flow_at_foil(
    flow_velocity: jnp.ndarray,
    boat_velocity: jnp.ndarray,
    foil_offset: jnp.ndarray,
    foil_theta: jnp.ndarray,
    foil_coe: jnp.ndarray,
    boat_theta: jnp.ndarray,
    boat_theta_dot: jnp.ndarray,
) -> jnp.ndarray:
    coe = coe_offset(foil_offset, foil_theta, foil_coe)
    flow_world_space = flow_velocity - boat_velocity
    boat_space_directional_flow = rotate_vector(flow_world_space, -boat_theta)
    boat_space_rotational_flow = -jnp.cross(
        jnp.array([0, 0, boat_theta_dot]), jnp.array([coe[0], coe[1], 0])
    )[0:2]
    boat_space_total_flow = boat_space_directional_flow + boat_space_rotational_flow
    foil_flow = rotate_vector(boat_space_total_flow, -foil_theta)
    return foil_flow

