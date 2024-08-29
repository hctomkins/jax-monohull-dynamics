import jax.numpy as jnp
from jax import jit, vmap
from monohull_dynamics.forces.hull import (
    init_hull,
    get_hull_coeffs,
    viscous_drag,
    wave_drag,
)


def test_hull_drag():
    hull_data = init_hull(hull_draft=0.2, beam=1.4, lwl=3.58)
    coeffs = get_hull_coeffs()
    assert -1500 < float(wave_drag(hull_data, coeffs, jnp.array([5, 0]))[0]) < -500
    assert -200 < float(viscous_drag(hull_data, jnp.array([5, 0]))[0]) < -150

    wd_jit = jit(wave_drag)
    vd_jit = jit(viscous_drag)

    assert -1500 < float(wd_jit(hull_data, coeffs, jnp.array([5, 0]))[0]) < -500
    assert -200 < float(vd_jit(hull_data, jnp.array([5, 0]))[0]) < -150

    wd_many_jit = jit(vmap(wave_drag, in_axes=(0, None, 0)))
    vd_many_jit = jit(vmap(viscous_drag))
    init_hull_many = vmap(init_hull)

    hull_data_many = init_hull_many(
        jnp.array([0.2, 0.3, 0.4]),
        jnp.array([1.4, 1.5, 1.6]),
        jnp.array([3.58, 3.59, 3.60]),
    )
    speeds_many = jnp.array([[5, 0], [6, 0], [7, 0]])

    print(wd_many_jit(hull_data_many, coeffs, speeds_many))

    assert -1500 < float(wd_many_jit(hull_data_many, coeffs, speeds_many)[0, 0]) < -500
    assert -200 < float(vd_many_jit(hull_data_many, speeds_many)[0, 0]) < -150
