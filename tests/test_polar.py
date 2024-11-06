import jax
import jax.numpy as jnp

from monohull_dynamics.forces.polars.polar import (
    cd0,
    cl,
    init_polar,
    knts_to_ms,
    water_re,
)


def test_n12():
    polar_data = init_polar("n12")
    re = water_re(knts_to_ms(4), 0.2)

    cd0_jit = jax.jit(cd0)
    cl_jit = jax.jit(cl)

    _cd = cd0(polar_data, re)
    _cl = cl(polar_data, re, 10)
    _cd = cd0_jit(polar_data, re)
    _cl = cl_jit(polar_data, re, 10)

    cl_many = jax.vmap(cl, in_axes=(None, 0, 0))
    cl_many_jit = jax.jit(cl_many)

    _cl_many = cl_many(polar_data, jnp.array([re]), jnp.array([10]))
    _cl_many = cl_many_jit(polar_data, jnp.array([re]), jnp.array([10]))
