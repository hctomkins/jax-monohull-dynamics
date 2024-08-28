from monohull_dynamics.dynamics.particle import ParticleState, integrate
from monohull_dynamics.forces.boat import init_firefly, forces_and_moments
import jax.numpy as jnp
from jax import jit
import jax
import time


def step_uncontrolled(state, boat, dt) -> ParticleState:
    f, m, _ = forces_and_moments(
        boat,
        state.xdot,
        jnp.array([0.0, 0.0]),
        state.theta,
        state.thetadot,
        jnp.array(0.0),
        jnp.array(0.0),
    )
    return integrate(state, f, m, dt)

def integrate_many(state, boat, dt, N):
    body_fn = lambda i, rolled_state: step_uncontrolled(rolled_state, boat, dt)
    return jax.lax.fori_loop(0, N, body_fn, state)


def test_e2e():
    boat_data = init_firefly()
    state = ParticleState(
        m=jnp.array(1.0),
        I=jnp.array(1.0),
        x=jnp.array([0.0,0.0]),
        xdot=jnp.array([1.0, 0.0]),
        theta=jnp.array(0.0),
        thetadot=jnp.array(0.0),
    )
    dt = jnp.array(0.1)

    N = 10
    t0 = time.time()
    for i in range(N):
        state = step_uncontrolled(state, boat_data, dt)
    print(f"Time taken per un-jitted step: {(time.time()-t0)/N}")

    N = 1000
    j_step_uncontrolled = jit(step_uncontrolled)
    t0 = time.time()
    for i in range(N):
        state = j_step_uncontrolled(state, boat_data, dt)
    print(f"Time taken per jitted step: {(time.time()-t0)/N}")

    N = 1000
    j_many = jit(integrate_many)
    j_many(state, boat_data, dt, N)
    t0 = time.time()
    state = j_many(state, boat_data, dt, N)
    print(f"Time taken per step in jitted loop: {(time.time()-t0)/N}")

