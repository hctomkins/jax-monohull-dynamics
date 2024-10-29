from monohull_dynamics.demo.demo import (
    init_simulation_state,
)
from monohull_dynamics.dynamics.boat import j_integrate_many, j_step_uncontrolled, step_uncontrolled
import time
import jax.profiler

from monohull_dynamics.dynamics.wind import evaluate_wind


def test_e2e():
    rng = jax.random.PRNGKey(0)
    sim_state = init_simulation_state(rng=rng)
    boat_state = sim_state.boat_state
    boat_state = jax.tree.map(lambda x: x[0], boat_state)
    wind_velocity = evaluate_wind(sim_state.wind_state, boat_state.particle_state.x)

    dt = 0.1

    N = 10
    t0 = time.time()
    for i in range(N):
        state = step_uncontrolled(boat_state, sim_state.force_model, wind_velocity, dt)
    print(f"Time taken per un-jitted step: {(time.time()-t0)/N}")

    N = 1000
    t0 = time.time()
    j_step_uncontrolled(boat_state, sim_state.force_model, wind_velocity, dt)
    for i in range(N):
        state = j_step_uncontrolled(boat_state, sim_state.force_model, wind_velocity, dt)
    print(f"Time taken per jitted step: {(time.time()-t0)/N}")

    N = 10000
    j_integrate_many(boat_state, sim_state.force_model, wind_velocity, dt, N)
    t0 = time.time()
    state = j_integrate_many(boat_state, sim_state.force_model, wind_velocity, dt, N)
    print(f"Time taken per step in jitted loop: {(time.time()-t0)/N}")

    # print("tracing")
    # with jax.profiler.trace("jax-trace-step-annotate-interp"):
    #     state = j_integrate_many(boat_state, sim_state.force_model, wind_velocity, dt, N)
    #     print("done")


if __name__ == "__main__":
    test_e2e()
