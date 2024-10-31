from functools import partial

import jax
import jax.numpy as jnp

from monohull_dynamics.dynamics.boat import BoatState, integrate_boats_euler, integrate_boats_jac, integrate_boats_rk4, \
    integrate_boats_hess, integrate_boats_newmark
from monohull_dynamics.dynamics.wind import WindParams, WindState, evaluate_wind_points, step_wind_state
from monohull_dynamics.forces.boat import BoatData, forces_and_moments_many
from monohull_dynamics.forces.polars.polar import rho_air


def cone_effect(cone_u: jnp.ndarray, base_width: jnp.ndarray, end_width: jnp.ndarray, length: jnp.ndarray, at: jnp.ndarray):
    r = jnp.linalg.norm(at)
    cone_u_norm = cone_u / jnp.linalg.norm(cone_u + 1e-6)
    at_norm = at / r
    u_component = jnp.dot(at_norm, cone_u_norm) * r
    v_vector = at - u_component * cone_u_norm
    v_component = jnp.linalg.norm(v_vector)
    cone_width_at_u = base_width + (end_width - base_width) * u_component / length

    # u falls off with gaussian, 4 std being the cone length
    # v falls off with gaussian, 2 std being the width at that u
    factor_u = jnp.exp(-(u_component**2) / (2 * (length / 2) ** 2))
    factor_v = jnp.exp(-(v_component**2) / (2 * (cone_width_at_u) ** 2))

    outside_cone = (u_component < 0) | (u_component > length) | (jnp.abs(v_component) > cone_width_at_u)
    factor = jnp.where(outside_cone, 0.0, factor_v * factor_u)
    factor = jnp.nan_to_num(factor, nan=0.0)
    # trim factor to 0 if at is outside cone (no if statements):
    return factor


@jax.jit
def get_sail_wind_interaction(
    force_on_sail: jnp.ndarray, sail_area: jnp.ndarray, sail_theta: jnp.ndarray, base_wind: jnp.ndarray, at: jnp.ndarray, min_threshold: float = 0.1
) -> jnp.ndarray:
    """
    Compute the effect of the boat on the wind at a position relative to the boat. This is an additive residual
    force_on_sail: [2]
    sail_area: []
    sail_theta: [] -> GLOBAL sail angle, 0 is pointing forwards on +ve x, anticlockwise rotation
    at: [2]
    """
    # f = ma = d/dt (mv)
    # if we can approximate delta t (time force acts on wind) from frontal area
    # then we can compute the change in wind velocity interpolated over the falloff cone
    acting_time = 3.0
    acting_height = 6.0
    acting_length = sail_area / acting_height
    sail_vector = jnp.array([jnp.cos(sail_theta), jnp.sin(sail_theta)])
    wind_norm = base_wind / jnp.linalg.norm(base_wind)

    sailwise_force_ratio = jnp.abs(jnp.dot(wind_norm, sail_vector))
    windwise_force_ratio = 1.0 - sailwise_force_ratio

    # Split dmv into wind and sail aligned cones - TODO: link to boat data
    sail_cone_base_width = 1.0
    sail_cone_end_width = 8.0
    sail_cone_length = 40
    sail_cone_area = (sail_cone_base_width + sail_cone_end_width) / 2 * sail_cone_length

    # Acting cone
    wind_cone_base_width = jnp.cross(wind_norm, sail_vector) * acting_length
    wind_cone_length = 20.0  # All wind effects are 10m long
    wind_cone_end_width = 4.0 + wind_cone_base_width * 3.0
    wind_cone_area = (wind_cone_base_width + wind_cone_end_width) / 2 * wind_cone_length

    dmv = -force_on_sail * acting_time
    sailwise_dmv = dmv * sailwise_force_ratio
    windwise_dmv = dmv * windwise_force_ratio

    # Average velocity change over cone
    windwise_m = rho_air * wind_cone_area * acting_height / 5  # Factor of 5 for non-uniform cone distribution
    sailwise_m = rho_air * sail_cone_area * acting_height / 5  # Factor of 5 for non-uniform cone distribution
    windwise_dv = windwise_dmv / windwise_m
    sailwise_dv = sailwise_dmv / sailwise_m

    # "at" [x,y] -> cone [u,v] u is length v is width relative, cone is in direction of -force_on_sail:
    windwise_factor = cone_effect(cone_u=base_wind, base_width=wind_cone_base_width, end_width=wind_cone_end_width, length=wind_cone_length, at=at)
    # handle flow reversal
    sail_reversed = -jnp.sign(jnp.dot(sail_vector, base_wind))
    sail_u = sail_vector * sail_reversed
    sailwise_factor = cone_effect(cone_u=-sail_u, base_width=sail_cone_base_width, end_width=sail_cone_end_width, length=sail_cone_length, at=at)

    # clip to min threshold radius
    windwise_factor = jnp.where(jnp.linalg.norm(at) < min_threshold, 0.0, windwise_factor)
    sailwise_factor = jnp.where(jnp.linalg.norm(at) < min_threshold, 0.0, sailwise_factor)

    wind_dv = windwise_dv * windwise_factor
    sail_dv = sailwise_dv * sailwise_factor

    return wind_dv + sail_dv, sailwise_factor + windwise_factor  # [2], []


def apply_sail_wind_interaction(base, offset):
    # CBA to vmap vmap, so apply to axis -1
    prev_norm = jnp.linalg.norm(base, axis=-1, keepdims=True)
    new_wind = base + offset
    new_norm = jnp.linalg.norm(new_wind, axis=-1, keepdims=True)
    new_norm_capped = jnp.minimum(prev_norm, new_norm)
    return new_wind * (new_norm_capped / new_norm)


we_grid = jax.vmap(get_sail_wind_interaction, in_axes=(None, None, None, None, 0))
we_grid = jax.vmap(we_grid, in_axes=(None, None, None, None, 0))

sail_wind_interaction_b = jax.vmap(
    get_sail_wind_interaction, in_axes=(0, None, 0, 0, 0)
)  # [B boats, at a list of offsets relevant to each boat ("single position")], return [B, 2]
sail_wind_interaction_b_at_b = jax.vmap(
    sail_wind_interaction_b, in_axes=(0, None, 0, 0, 0)
)  # [BxB effector boats, at a grid of BxB affected offsets] (we're doing an extra broadcast than needed here but it's simpler)


@partial(jax.jit, static_argnames=("integrator",))
def step_wind_and_boats_with_interaction(
    boats_state: BoatState,
    force_model: BoatData,
    wind_state: WindState,
    wind_params: WindParams,
    inner_dt: jnp.ndarray,
    rng: jnp.ndarray,
    integrator="rk4",
) -> tuple[BoatState, WindState, jnp.ndarray]:
    rng, _ = jax.random.split(rng)
    particles = boats_state.particle_state
    wind_state, rng = step_wind_state(wind_state, rng, inner_dt, wind_params)
    wind_velocities = evaluate_wind_points(wind_state, particles.x)  # [B, 2]
    n_boats = particles.x.shape[0]

    for i in range(1):
        f, m, dd = forces_and_moments_many(
            force_model,
            particles.xdot,
            wind_velocities,
            particles.theta,
            particles.thetadot,
            boats_state.sail_angle,
            boats_state.rudder_angle,
        )  # [B, 2]

        # for [B,B] - effector per column, affected per row
        # I.E. from row i to column j / boat i's offset from boat j - fits with vmap strategy
        b_at_b_position_offsets = particles.x[:, None, :] - particles.x[None, :, :]  # [B, B, 2]
        wind_offsets, _ = sail_wind_interaction_b_at_b(
            jax.lax.broadcast(f, (n_boats,)),
            force_model.sail_area,
            jax.lax.broadcast(boats_state.sail_angle, (n_boats,)),
            jax.lax.broadcast(wind_velocities, (n_boats,)),
            b_at_b_position_offsets,
        )  # [B, B, 2]
        wind_offsets = jnp.sum(wind_offsets, axis=1)
        wind_velocities = apply_sail_wind_interaction(wind_velocities, wind_offsets)

    # f, m, dd = forces_and_moments_many(
    #     force_model,
    #     particles.xdot,
    #     wind_velocities,
    #     particles.theta,
    #     particles.thetadot,
    #     boats_state.sail_angle,
    #     boats_state.rudder_angle,
    # )  # [B]
    #
    # boats_state = boats_state._replace(particle_state=integrate_multiple(particles, f, m, inner_dt), debug_data=dd)
    # wind_offsets = wind_velocities
    if integrator == "rk4":
        boats_state = integrate_boats_rk4(boats_state, force_model, wind_velocities, inner_dt)
    elif integrator == "jac":
        boats_state = integrate_boats_jac(boats_state, force_model, wind_velocities, inner_dt)
    elif integrator == "hess":
        boats_state = integrate_boats_hess(boats_state, force_model, wind_velocities, inner_dt)
    elif integrator == "newmark":
        boats_state = integrate_boats_newmark(boats_state, force_model, wind_velocities, inner_dt)
    else:
        boats_state = integrate_boats_euler(boats_state, force_model, wind_velocities, inner_dt)
    return boats_state, wind_state, rng, wind_offsets


@partial(jax.jit, static_argnames=["n", "integrator"])
def step_wind_and_boats_with_interaction_multiple(
    boats_state: BoatState,
    force_model: BoatData,
    wind_state: WindState,
    wind_params: WindParams,
    inner_dt: jnp.ndarray,
    rng: jnp.ndarray,
    n: int,
    integrator="rk4",
) -> BoatState:
    init_rolled_state = (boats_state, wind_state, rng, jnp.array([[0.0, 0.0]]))

    def body_fn(i, rolled_state):
        _boats_state, _wind_state, _rng, _offsets = rolled_state
        return step_wind_and_boats_with_interaction(_boats_state, force_model, _wind_state, wind_params, inner_dt, _rng, integrator)

    retuple = jax.lax.fori_loop(0, n, body_fn, init_rolled_state)
    return retuple
