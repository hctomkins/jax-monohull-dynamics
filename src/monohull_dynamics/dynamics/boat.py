import typing

import jax
import jax.numpy as jnp

from monohull_dynamics.dynamics.particle import ParticleState, integrate
from monohull_dynamics.forces.boat import (
    BoatData,
    forces_and_moments,
)


class BoatState(typing.NamedTuple):
    particle_state: ParticleState
    rudder_angle: jnp.ndarray
    sail_angle: jnp.ndarray
    debug_data: dict


def step_uncontrolled(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt) -> BoatState:
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
    new_boat_state = boat_state._replace(particle_state=integrate(particle_state, f, m, inner_dt), debug_data=dd)
    # jax.debug.print("force {f}", f=f)
    return new_boat_state


def step_uncontrolled_jac(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, dt) -> BoatState:
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

    # Columns are inputs wrt, rows are the r.o.c of that value in the output
    (fd_vel, fd_theta, fd_ang), (mdot_vel, mdot_theta, mdot_ang), _ = jax.jacfwd(forces_and_moments, (1, 3, 4))(
        force_model, particle_state.xdot, wind_velocity, particle_state.theta, particle_state.thetadot, boat_state.sail_angle, boat_state.rudder_angle
    )

    # a
    xdotdot = f / particle_state.m
    thetadotdot = m / particle_state.I

    fdot = fd_vel @ xdotdot + fd_theta * particle_state.thetadot + fd_ang * thetadotdot
    mdot = mdot_vel @ xdotdot + mdot_theta * particle_state.thetadot + mdot_ang * thetadotdot

    new_f = f + fdot * dt
    new_m = m + mdot * dt

    # jax.debug.print("df_0 {df0}, df_1 {df1}", df0=f / particle_state.m, df1=fdot * dt)

    new_xdotdot = new_f / particle_state.m
    new_thetadotdot = new_m / particle_state.I

    # v
    new_xdot = particle_state.xdot + new_xdotdot * dt
    new_thetadot = particle_state.thetadot + new_thetadotdot * dt

    # x
    new_x = particle_state.x + new_xdot * dt
    new_theta = particle_state.theta + new_thetadot * dt

    return boat_state._replace(
        particle_state=boat_state.particle_state._replace(x=new_x, xdot=new_xdot, theta=new_theta, thetadot=new_thetadot), debug_data=dd
    )


def step_uncontrolled_rk4(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, dt) -> BoatState:
    print("USING RK4")

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


def integrate_steps(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt, n) -> BoatState:
    def body_fn(i, rolled_state):
        return step_uncontrolled_rk4(rolled_state, force_model, wind_velocity, inner_dt)

    return jax.lax.fori_loop(0, n, body_fn, boat_state)


integrate_boats_euler = jax.vmap(step_uncontrolled, in_axes=(0, None, 0, None))
integrate_boats_rk4 = jax.vmap(step_uncontrolled_rk4, in_axes=(0, None, 0, None))
integrate_boats_jac = jax.vmap(step_uncontrolled_jac, in_axes=(0, None, 0, None))


j_step_uncontrolled = jax.jit(step_uncontrolled, static_argnums=(3,))
j_integrate_steps = jax.jit(integrate_steps, static_argnums=(4))


def integrate_many_debug(boat_state: BoatState, force_model: BoatData, wind_velocity: jnp.ndarray, inner_dt, n) -> BoatState:
    for i in range(n):
        boat_state = step_uncontrolled(boat_state, force_model, wind_velocity, inner_dt)
    return boat_state
