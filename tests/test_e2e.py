from monohull_dynamics.demo.demo import (
    init_simulation_state,
)
import time
import jax.profiler

from monohull_dynamics.dynamics.wind import evaluate_wind
from monohull_dynamics.dynamics.boat import integrate_steps, integrate


integrate_steps_jit = jax.jit(integrate_steps, static_argnames=["integrator","n_steps"])
integrate_jit = jax.jit(integrate, static_argnames=["integrator"])


def test_e2e():
    rng = jax.random.PRNGKey(0)
    sim_state = init_simulation_state(rng=rng)
    boat_state = sim_state.boat_state
    boat_state = jax.tree.map(lambda x: x[0], boat_state)
    wind_velocity = evaluate_wind(sim_state.wind_state, boat_state.particle_state.x)

    dt = 0.1

    N = 10
    t0 = time.time()
    with jax.disable_jit():
        for i in range(N):
            state = integrate(boat_state, sim_state.force_model, wind_velocity, dt, "euler")
    print(f"Time taken per un-jitted step: {(time.time()-t0)/N}")

    N = 1000
    t0 = time.time()
    integrate_jit(boat_state, sim_state.force_model, wind_velocity, dt, "euler")
    for i in range(N):
        state = integrate_jit(boat_state, sim_state.force_model, wind_velocity, dt, "euler")
    print(f"Time taken per jitted step: {(time.time()-t0)/N}")

    N = 10000
    integrate_steps_jit(boat_state, sim_state.force_model, wind_velocity, dt, N, "euler")
    t0 = time.time()
    state = integrate_steps_jit(boat_state, sim_state.force_model, wind_velocity, dt, N, "euler")
    print(f"Time taken per step in jitted loop: {(time.time()-t0)/N}")

    # boats_state = jax.tree.map(lambda x: x[None], boat_state)
    # integrate_wind_and_boats_with_interaction_multiple(boats_state, sim_state.force_model, sim_state.wind_state, sim_state.wind_params, dt, rng, n=1)

    # N = 50
    # print("tracing")
    # with jax.profiler.trace("jax-trace-boatstep-annotate-interp"):
    #     state = j_integrate_steps(boat_state, sim_state.force_model, wind_velocity, dt, N)
    #
    # with jax.profiler.trace("jax-trace-windstep-annotate-interp"):
    #     integrate_wind_and_boats_with_interaction_multiple(boats_state, sim_state.force_model, sim_state.wind_state,
    #                                                        sim_state.wind_params, dt, rng, n=1)
    #     print("done")


if __name__ == "__main__":
    test_e2e()
