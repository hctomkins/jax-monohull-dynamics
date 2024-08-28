from monohull_dynamics.dynamics.particle import ParticleState, integrate
import jax.numpy as jnp
import jax


def init_state(x: jnp.ndarray):
    return ParticleState(
        m=jnp.array(1.0),
        I=jnp.array(1.0),
        x=x,
        xdot=jnp.array([0.0, 0.0]),
        theta=jnp.array(0.0),
        thetadot=jnp.array(0.0),
    )


def test_integrate():
    x0 = jnp.array([0.0, 0.0])
    state = init_state(jnp.array(0.0))
    force = jnp.array([1.0, 0.0])
    moment = jnp.array([0.0])
    dt = jnp.array(0.1)
    state = integrate(state, force, moment, dt)
    assert jnp.allclose(state.x[0], 0.01)


def test_integrate_many():
    x1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    force = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    moment = jnp.array([[0.0], [0.0]])

    get_many_state = jax.vmap(init_state, in_axes=0, out_axes=0)
    integrate_many = jax.vmap(integrate, in_axes=(0, 0, 0, None), out_axes=0)

    state = get_many_state(x1)
    dt = jnp.array(0.1)
    state = integrate_many(state, force, moment, dt)
    assert jnp.allclose(state.x, jnp.array([[0.01, 0.0], [1.01, 1.0]]))
