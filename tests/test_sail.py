from monohull_dynamics.forces.sails import init_sail_data, sail_frame_resultant
from jax import jit, vmap
import jax.numpy as jnp


def test_sail():
    data = init_sail_data(1.0)
    many_data = vmap(init_sail_data)(jnp.array([1.0, 2.0]))
    many_resultant = vmap(sail_frame_resultant)

    f = sail_frame_resultant(data, jnp.array([1.0, 0.0]))
    f0 = many_resultant(many_data, jnp.array([[1.0, 0.0], [1.0, 0.0]]))

    f = jit(sail_frame_resultant)(data, jnp.array([1.0, 0.0]))
    f0 = jit(many_resultant)(many_data, jnp.array([[1.0, 0.0], [1.0, 0.0]]))

    print(f, f0)
