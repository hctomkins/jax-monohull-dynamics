"""
Approximates wind with a sparse set of dynamics on a base velocity vector
"""

import typing

import jax
import jax.numpy as jnp

N_LOCAL_GUSTS = 30


# All wind thetas are measured anticlockwise from +ve x-axis
class WindParams(typing.NamedTuple):
    base_theta_deg: jnp.ndarray  # []
    base_r: jnp.ndarray  # []
    theta_oscillation_amplitude_deg: jnp.ndarray  # []
    theta_oscillation_period_s: jnp.ndarray  # []
    r_oscillation_amplitude: jnp.ndarray  # []
    r_oscillation_period_s: jnp.ndarray  # []
    local_gust_strength_offset_std: jnp.ndarray  # []
    local_gust_theta_deg_offset_std: jnp.ndarray  # []
    local_gust_radius_mean: jnp.ndarray  # []
    local_gust_radius_std: jnp.ndarray  # []
    bbox_lims: jnp.ndarray  # [4] box to wrap gusts, xmin xmax ymin ymax


class WindState(typing.NamedTuple):
    current_theta_phase: jnp.ndarray
    current_r_phase: jnp.ndarray
    current_base_r: jnp.ndarray
    current_base_theta: jnp.ndarray
    local_gust_centers: jnp.ndarray  # [N_LOCAL_GUSTS, 2]
    local_gust_strengths: jnp.ndarray  # [N_LOCAL_GUSTS]
    local_gust_theta_deg: jnp.ndarray  # [N_LOCAL_GUSTS]
    local_gust_effect_radius: jnp.ndarray  # [N_LOCAL_GUSTS]
    local_gust_start_points: jnp.ndarray  # [N_LOCAL_GUSTS, 2]


def wind_spawn(bbox_lims: jnp.ndarray, base_theta_deg, rng):
    # TODO: Does this bias sampling to the center of the box?
    xmin, xmax, ymin, ymax = bbox_lims
    r_circle = jnp.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2) / 2
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    theta_rad = jnp.deg2rad(base_theta_deg) + jnp.pi
    x_on_circle = jnp.cos(theta_rad) * r_circle
    y_on_circle = jnp.sin(theta_rad) * r_circle
    perp_vector = jnp.array([y_on_circle, -x_on_circle])
    perp_vector = perp_vector / jnp.linalg.norm(perp_vector)
    random_perp_offset = jax.random.uniform(rng, minval=-1, maxval=1) * r_circle * perp_vector
    return jnp.array([x_center + x_on_circle, y_center + y_on_circle]) + random_perp_offset


spawn_many = jax.vmap(wind_spawn, in_axes=(None, None, 0))

def default_params(bbox_lims: jnp.ndarray) -> WindParams:
    return WindParams(
        base_theta_deg=jnp.array(-90.0),
        base_r=jnp.array(5.0),  # 10 knots
        theta_oscillation_amplitude_deg=jnp.array(10.0),
        theta_oscillation_period_s=jnp.array(180.0),
        r_oscillation_amplitude=jnp.array(1.0),
        r_oscillation_period_s=jnp.array(60.0*5),
        local_gust_strength_offset_std=jnp.array(0.5),
        local_gust_theta_deg_offset_std=jnp.array(3.0),
        local_gust_radius_mean=jnp.array(10.0),
        local_gust_radius_std=jnp.array(5),
        bbox_lims=bbox_lims
    )

def default_state(params: WindParams, rng) -> WindState:
    xmin, xmax, ymin, ymax = params.bbox_lims
    initial_gust_centers = jax.random.uniform(rng, shape=(N_LOCAL_GUSTS, 2), minval=0.0, maxval=1.0)
    initial_gust_centers = initial_gust_centers * jnp.array([xmax - xmin, ymax - ymin]) + jnp.array([xmin, ymin])

    return WindState(
        current_theta_phase=jax.random.uniform(rng) * 2 * jnp.pi,
        current_r_phase=jax.random.uniform(rng) * 2 * jnp.pi,
        current_base_r=params.base_r,
        current_base_theta=params.base_theta_deg,
        local_gust_centers=initial_gust_centers,
        local_gust_strengths=jnp.maximum(
            params.base_r + params.local_gust_strength_offset_std * jax.random.normal(rng, shape=(N_LOCAL_GUSTS,)),
            params.local_gust_strength_offset_std
        ),
        local_gust_theta_deg=params.base_theta_deg + jax.random.normal(rng, shape=(N_LOCAL_GUSTS,)) * params.local_gust_theta_deg_offset_std,
        local_gust_effect_radius=jnp.maximum(
            params.local_gust_radius_mean + jax.random.normal(rng, shape=(N_LOCAL_GUSTS,)) * params.local_gust_radius_std,
            params.local_gust_radius_std
        ),
        local_gust_start_points=initial_gust_centers
    )


def step_wind_state(state: WindState, rng: jnp.ndarray, dt: jnp.ndarray, params: WindParams) -> tuple[WindState, jnp.ndarray]:
    rng, _ = jax.random.split(rng)
    xmin, xmax, ymin, ymax = params.bbox_lims

    # Step 1: Update the base wind oscillation for direction (theta) and magnitude (r)
    new_theta_phase = (state.current_theta_phase + (2 * jnp.pi / params.theta_oscillation_period_s) * dt) % (2 * jnp.pi)
    new_r_phase = (state.current_r_phase + (2 * jnp.pi / params.r_oscillation_period_s) * dt) % (2 * jnp.pi)

    # Oscillate the base wind direction (theta) and magnitude (r)
    new_base_theta_deg = params.base_theta_deg + params.theta_oscillation_amplitude_deg * jnp.sin(new_theta_phase)
    new_base_r = params.base_r + params.r_oscillation_amplitude * jnp.sin(new_r_phase)

    # Step 2: Update the local gust positions
    base_strength = new_base_r * jnp.array([jnp.cos(jnp.radians(new_base_theta_deg)), jnp.sin(jnp.radians(new_base_theta_deg))])
    gust_dx = (base_strength[None, 0] + state.local_gust_strengths * jnp.cos(jnp.radians(state.local_gust_theta_deg))) * dt
    gust_dy = (base_strength[None, 1] + state.local_gust_strengths * jnp.sin(jnp.radians(state.local_gust_theta_deg))) * dt

    # Update gust positions
    new_gust_centers = state.local_gust_centers + jnp.stack([gust_dx, gust_dy], axis=-1)

    # Step 3: Respawn gusts whose centers were inside the box, and are now outside the box
    distance_traveled = jnp.linalg.norm(new_gust_centers - state.local_gust_start_points, axis=1)
    r_wind_circle = jnp.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    respawn = distance_traveled >= r_wind_circle

    # Respawn gusts upwind but keep their original theta offset
    gust_rngs = jax.random.split(rng, N_LOCAL_GUSTS)
    spawned_gust_centers = spawn_many(params.bbox_lims, new_base_theta_deg, gust_rngs)  # [N_LOCAL_GUSTS, 2]
    spawned_gust_theta_deg = new_base_theta_deg + params.local_gust_theta_deg_offset_std * jax.random.normal(rng, shape=(N_LOCAL_GUSTS,))
    spawned_gust_strengths = jnp.maximum(
        params.local_gust_strength_offset_std * jax.random.normal(rng, shape=(N_LOCAL_GUSTS,)),
        0.0
    )
    spawned_gust_radii = jnp.maximum(
        params.local_gust_radius_mean + params.local_gust_radius_std * jax.random.normal(rng, shape=(N_LOCAL_GUSTS,)),
        params.local_gust_radius_std
    )

    new_gust_centers = jnp.where(respawn[:, None], spawned_gust_centers, new_gust_centers)
    new_gust_strengths = jnp.where(respawn, spawned_gust_strengths, state.local_gust_strengths)
    new_gust_theta_deg = jnp.where(respawn, spawned_gust_theta_deg, state.local_gust_theta_deg)
    new_gust_radius = jnp.where(respawn, spawned_gust_radii, state.local_gust_effect_radius)
    new_start_points = jnp.where(respawn[:, None], spawned_gust_centers, state.local_gust_start_points)

    # Step 4: Return updated wind state
    return WindState(
        current_theta_phase=new_theta_phase,
        current_r_phase=new_r_phase,
        current_base_r=new_base_r,
        current_base_theta=new_base_theta_deg,
        local_gust_centers=new_gust_centers,
        local_gust_strengths=new_gust_strengths,
        local_gust_theta_deg=new_gust_theta_deg,
        local_gust_effect_radius=new_gust_radius,
        local_gust_start_points=new_start_points
    ), rng


def evaluate_gust_effect(
    gust_center: jnp.ndarray, gust_strength: jnp.ndarray, gust_theta_deg: jnp.ndarray, gust_radius: jnp.ndarray, pos: jnp.ndarray
) -> jnp.ndarray:
    """
    Args:
        gust_center: [2]
        gust_strength: []
        gust_theta_deg: []
        gust_radius: []
        pos: [2]

    Returns: [2]
    """
    # Calculate the distance from the position to the gust center
    gust_theta_rad = jnp.deg2rad(gust_theta_deg)
    distance = jnp.linalg.norm(pos - gust_center)

    # Compute the Gaussian effect based on the distance and gust radius
    gaussian_effect = jnp.exp(-0.5 * (distance / gust_radius) ** 2)

    # Calculate the wind effect vector using the gust strength and direction
    gust_direction = jnp.array([jnp.cos(gust_theta_rad), jnp.sin(gust_theta_rad)])
    wind_effect = gust_strength * gaussian_effect * gust_direction

    return wind_effect


evaluate_gusts_effect = jax.vmap(evaluate_gust_effect, in_axes=(0, 0, 0, 0, None))


def evaluate_wind(wind_state: WindState, pos: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        pos: [2]

    Returns: [2]
    """
    # Base wind
    base_theta_rad = jnp.deg2rad(wind_state.current_base_theta)
    base_wind = jnp.array([jnp.cos(base_theta_rad), jnp.sin(base_theta_rad)]) * wind_state.current_base_r  # [2]
    gusts_effect = evaluate_gusts_effect(
        wind_state.local_gust_centers, wind_state.local_gust_strengths, wind_state.local_gust_theta_deg, wind_state.local_gust_effect_radius, pos
    )  # [N_LOCAL_GUSTS, 2]
    return base_wind + jnp.sum(gusts_effect, axis=0)

evaluate_wind_points = jax.vmap(evaluate_wind, in_axes=(None, 0))
evaluate_wind_grid = jax.jit(jax.vmap(evaluate_wind_points, in_axes=(None, 0)))