import jax.numpy as jnp
from jax import jit, vmap

from monohull_dynamics.forces.foils import FoilData, foil_frame_resultant
from monohull_dynamics.forces.polars.polar import init_polar


def test_foils():
    polar_data = init_polar("n12")
    foil_data = FoilData(
        polar=polar_data,
        length=jnp.array([1.0]),
        chord=jnp.array([0.2]),
    )
    f_sym = foil_frame_resultant(
        foil_data=foil_data,
        foil_frame_flow=jnp.array([-1.0, 0.0]),
    )
    assert f_sym[0] < 0
    assert -0.01 < f_sym[1] < 0.01
    f_lift = foil_frame_resultant(
        foil_data=foil_data,
        foil_frame_flow=jnp.array([-1.0, 0.1]),
    )
    assert f_lift[0] > 1, f_lift[1] > 50

    f_null = foil_frame_resultant(
        foil_data=foil_data,
        foil_frame_flow=jnp.array([-0.0, 0.0]),
    )
    assert jnp.allclose(f_null, jnp.array([0.0, 0.0]), atol=1e-5, rtol=1e-5)

    resultant_many = jit(vmap(foil_frame_resultant, in_axes=(None, 0)))

    many_f = resultant_many(
        foil_data,
        jnp.array([[-1.0, 0.0], [-1.0, 0.1]]),
    )
    assert jnp.allclose(many_f, jnp.stack([f_sym, f_lift]), atol=1e-5, rtol=1e-5)
