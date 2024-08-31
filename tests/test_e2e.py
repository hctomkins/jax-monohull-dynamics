from monohull_dynamics.demo.demo import (
    init_simulation_state,
)
import jax.numpy as jnp
from jax import jit
import time

from monohull_dynamics.dynamics.boat import integrate_many, step_uncontrolled


def test_e2e():
    state = init_simulation_state()
    dt = jnp.array(0.1)

    N = 10
    t0 = time.time()
    for i in range(N):
        new_boat_state = step_uncontrolled(state.boat_state, state.force_model, state.wind_velocity, dt)
    print(f"Time taken per un-jitted step: {(time.time()-t0)/N}")

    N = 1000
    j_step_uncontrolled = jit(step_uncontrolled)
    t0 = time.time()
    for i in range(N):
        new_boat_state = j_step_uncontrolled(state.boat_state, state.force_model, state.wind_velocity, dt)
    print(f"Time taken per jitted step: {(time.time()-t0)/N}")

    N = 1000000
    j_many = jit(integrate_many)
    j_many(state.boat_state, state.force_model, state.wind_velocity, dt, N)
    t0 = time.time()
    _ = j_many(state.boat_state, state.force_model, state.wind_velocity, dt, N)
    print(f"Time taken per step in jitted loop: {(time.time()-t0)/N}")
