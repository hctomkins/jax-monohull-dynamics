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
