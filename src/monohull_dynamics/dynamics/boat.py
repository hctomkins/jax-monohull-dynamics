import typing

import jax
import jax.numpy as jnp

from monohull_dynamics.dynamics.integration_utils import gauss_legendre_fourth_order_jax_vector
from monohull_dynamics.dynamics.particle import ParticleState
from monohull_dynamics.forces.boat import (
    BoatData,
    forces_and_moments,
)


class BoatState(typing.NamedTuple):
    particle_state: ParticleState
    rudder_angle: jnp.ndarray
    sail_angle: jnp.ndarray
    debug_data: dict


def integrate_euler(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt) -> BoatState:
    particle_state = boat_state.particle_state

    force, moment, dd = forces_and_moments(
        boat_data=force_model,
        boat_velocity=particle_state.xdot,
        wind_velocity=wind_velocity,
        boat_theta=particle_state.theta,
        boat_theta_dot=particle_state.thetadot,
        sail_angle=boat_state.sail_angle,
        rudder_angle=boat_state.rudder_angle,
    )
    # a
    xdotdot = force / particle_state.m
    thetadotdot = moment / particle_state.I

    # v
    new_xdot = particle_state.xdot + xdotdot * inner_dt
    new_thetadot = particle_state.thetadot + thetadotdot * inner_dt

    # x
    new_x = particle_state.x + new_xdot * inner_dt
    new_theta = particle_state.theta + new_thetadot * inner_dt
    new_boat_state = boat_state._replace(
        particle_state=ParticleState(
            m=particle_state.m,
            I=particle_state.I,
            x=new_x,
            xdot=new_xdot,
            theta=new_theta,
            thetadot=new_thetadot,
        ),
        debug_data=dd,
    )
    return new_boat_state


@jax.jit
def integrate_i4(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, dt) -> BoatState:
    particle_state = boat_state.particle_state

    def a_func(_x, _v):
        x, theta = _x[:2], _x[2]
        v, thetadot = _v[:2], _v[2]
        f, m, dd = forces_and_moments(
            boat_data=force_model,
            boat_velocity=v,
            wind_velocity=wind_velocity,
            boat_theta=theta,
            boat_theta_dot=thetadot,
            sail_angle=boat_state.sail_angle,
            rudder_angle=boat_state.rudder_angle,
        )
        xdotdot = f / particle_state.m
        thetadotdot = m / particle_state.I
        return jnp.concat([xdotdot, thetadotdot[None]], axis=-1), dd

    x0 = jnp.concat([particle_state.x, particle_state.theta[None]], axis=-1)
    v0 = jnp.concat([particle_state.xdot, particle_state.thetadot[None]], axis=-1)

    new_x_, new_v_, dd = gauss_legendre_fourth_order_jax_vector(a_func, x0, v0, dt)
    new_x, new_theta = new_x_[:2], new_x_[2]
    new_xdot, new_thetadot = new_v_[:2], new_v_[2]

    return boat_state._replace(
        particle_state=boat_state.particle_state._replace(x=new_x, xdot=new_xdot, theta=new_theta, thetadot=new_thetadot), debug_data=dd
    )


def integrate_rk4(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, dt) -> BoatState:
    def derivatives_at(xdot, theta, thetadot):
        force, moment, dd = forces_and_moments(
            boat_data=force_model,
            boat_velocity=xdot,
            wind_velocity=wind_velocity,
            boat_theta=theta,
            boat_theta_dot=thetadot,
            sail_angle=boat_state.sail_angle,
            rudder_angle=boat_state.rudder_angle,
        )
        return force / boat_state.particle_state.m, moment / boat_state.particle_state.I, dd

    # k1
    k1_xdotdot, k1_thetadotdot, dd = derivatives_at(
        xdot=boat_state.particle_state.xdot, theta=boat_state.particle_state.theta, thetadot=boat_state.particle_state.thetadot
    )
    k1_xdot = boat_state.particle_state.xdot
    k1_thetadot = boat_state.particle_state.thetadot
    k1_theta = boat_state.particle_state.theta

    k2_xdot = k1_xdot + k1_xdotdot * (dt / 2)
    k2_thetadot = k1_thetadot + k1_thetadotdot * (dt / 2)
    k2_xdotdot, k2_thetadotdot, _ = derivatives_at(
        xdot=k2_xdot,
        theta=k1_theta + k1_thetadot * (dt / 2),
        thetadot=k2_thetadot,
    )
    k3_xdot = k1_xdot + k2_xdotdot * (dt / 2)
    k3_thetadot = k1_thetadot + k2_thetadotdot * (dt / 2)
    k3_xdotdot, k3_thetadotdot, _ = derivatives_at(
        xdot=k3_xdot,
        theta=k1_theta + k2_thetadot * (dt / 2),
        thetadot=k3_thetadot,
    )
    k4_xdot = k1_xdot + k3_xdotdot * dt
    k4_thetadot = k1_thetadot + k3_thetadotdot * dt
    k4_xdotdot, k4_thetadotdot, _ = derivatives_at(
        xdot=k4_xdot,
        theta=k1_theta + k3_thetadot * dt,
        thetadot=k4_thetadot,
    )
    d_x = (1 / 6) * (k1_xdot + 2 * k2_xdot + 2 * k3_xdot + k4_xdot)
    d_theta = (1 / 6) * (k1_thetadot + 2 * k2_thetadot + 2 * k3_thetadot + k4_thetadot)
    d_xdot = (1 / 6) * (k1_xdotdot + 2 * k2_xdotdot + 2 * k3_xdotdot + k4_xdotdot)
    d_thetadot = (1 / 6) * (k1_thetadotdot + 2 * k2_thetadotdot + 2 * k3_thetadotdot + k4_thetadotdot)

    new_thetadot = boat_state.particle_state.thetadot + d_thetadot * dt
    new_x = boat_state.particle_state.x + d_x * dt
    new_xdot = boat_state.particle_state.xdot + d_xdot * dt
    new_theta = boat_state.particle_state.theta + d_theta * dt

    return boat_state._replace(
        particle_state=boat_state.particle_state._replace(x=new_x, xdot=new_xdot, theta=new_theta, thetadot=new_thetadot), debug_data=dd
    )


def integrate(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt, integrator) -> BoatState:
    if integrator == "euler":
        return integrate_euler(boat_state, force_model, wind_velocity, inner_dt)
    elif integrator == "rk4":
        return integrate_rk4(boat_state, force_model, wind_velocity, inner_dt)
    elif integrator == "i4":
        return integrate_i4(boat_state, force_model, wind_velocity, inner_dt)


def integrate_steps(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt, n_steps, integrator) -> BoatState:
    def body_fn(i, boat_state):
        return integrate(boat_state, force_model, wind_velocity, inner_dt, integrator)

    return jax.lax.fori_loop(0, n_steps, body_fn, boat_state)


integrate_boats = jax.vmap(integrate, in_axes=(0, None, 0, None, None))
integrate_boats_steps = jax.vmap(integrate_steps, in_axes=(0, None, 0, None, None, None))
