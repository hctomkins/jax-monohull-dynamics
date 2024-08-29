from monohull_dynamics.demo.demo import (
    step_uncontrolled,
    integrate_many,
    init_simulation_state,
)
import jax.numpy as jnp
from jax import jit
import time


def test_e2e():
    state = init_simulation_state()
    dt = jnp.array(0.1)

    N = 10
    t0 = time.time()
    for i in range(N):
        state = step_uncontrolled(state, dt)
    print(f"Time taken per un-jitted step: {(time.time()-t0)/N}")

    N = 1000
    j_step_uncontrolled = jit(step_uncontrolled)
    t0 = time.time()
    for i in range(N):
        state = j_step_uncontrolled(state, dt)
    print(f"Time taken per jitted step: {(time.time()-t0)/N}")

    N = 1000000
    j_many = jit(integrate_many)
    j_many(state, dt, N)
    t0 = time.time()
    state = j_many(state, dt, N)
    print(f"Time taken per step in jitted loop: {(time.time()-t0)/N}")
