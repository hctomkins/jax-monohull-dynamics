
import jax
import jax.numpy as jnp

from monohull_dynamics.dynamics.particle import BoatState, integrate
from monohull_dynamics.forces.boat import (
    BoatData,
    forces_and_moments,
)


def step_uncontrolled(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt) -> tuple[BoatState, dict]:
    particle_state = boat_state.particle_state

    f, m, dd = forces_and_moments(
        boat_data=force_model,
        boat_velocity=particle_state.xdot,
        wind_velocity=wind_velocity,
        boat_theta=particle_state.theta,
        boat_theta_dot=particle_state.thetadot,
        sail_angle=boat_state.sail_angle,
        rudder_angle=boat_state.rudder_angle,
    )
    new_boat_state = boat_state._replace(particle_state=integrate(particle_state, f, m, inner_dt))
    # jax.debug.print("force {f}", f=f)
    return new_boat_state, dd


def integrate_many(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt, N) -> BoatState:
    body_fn = lambda i, rolled_state: step_uncontrolled(rolled_state, force_model, wind_velocity, inner_dt)
    return jax.lax.fori_loop(0, N, body_fn, boat_state)


j_step_uncontrolled = jax.jit(step_uncontrolled, static_argnums=(1,))
j_integrate_many = jax.jit(integrate_many, static_argnums=(2))


def integrate_many_debug(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt, N) -> BoatState:
    for i in range(N):
        boat_state = step_uncontrolled(boat_state, force_model, wind_velocity, inner_dt)
    return boat_state
