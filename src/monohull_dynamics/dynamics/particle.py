import typing

import jax
import jax.numpy as jnp


class ParticleState(typing.NamedTuple):
    m: jnp.ndarray
    I: jnp.ndarray
    x: jnp.ndarray
    xdot: jnp.ndarray
    theta: jnp.ndarray
    thetadot: jnp.ndarray


@jax.jit
def integrate(
    particle_state: ParticleState,
    force: jnp.ndarray,
    moment: jnp.ndarray,
    dt: jnp.ndarray,
) -> ParticleState:
    """
    Step dt with semi implicit Euler
    """
    # Force [2], moment []
    # if jnp.any(jnp.isnan(force)):
    #     raise ValueError("Force is nan")

    # a
    xdotdot = force / particle_state.m
    thetadotdot = moment / particle_state.I

    # v
    new_xdot = particle_state.xdot + xdotdot * dt
    new_thetadot = particle_state.thetadot + thetadotdot * dt

    # x
    new_x = particle_state.x + new_xdot * dt
    new_theta = particle_state.theta + new_thetadot * dt

    return ParticleState(
        m=particle_state.m,
        I=particle_state.I,
        x=new_x,
        xdot=new_xdot,
        theta=new_theta,
        thetadot=new_thetadot,
    )

integrate_multiple = jax.vmap(integrate, in_axes=(0, 0, 0, None))