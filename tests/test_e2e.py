from monohull_dynamics.demo.demo import (
    init_simulation_state,
)
from monohull_dynamics.dynamics.boat import j_integrate_steps, j_step_uncontrolled, step_uncontrolled, step_uncontrolled_jac
import time
import jax.profiler

from monohull_dynamics.dynamics.boat_wind_interaction import step_wind_and_boats_with_interaction_multiple
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
    j_integrate_steps(boat_state, sim_state.force_model, wind_velocity, dt, N)
    t0 = time.time()
    state = j_integrate_steps(boat_state, sim_state.force_model, wind_velocity, dt, N)
    print(f"Time taken per step in jitted loop: {(time.time()-t0)/N}")

    boats_state = jax.tree.map(lambda x: x[None], boat_state)

    step_wind_and_boats_with_interaction_multiple(boats_state, sim_state.force_model, sim_state.wind_state, sim_state.wind_params, dt, rng, n=1)

    N = 50
    print("tracing")
    with jax.profiler.trace("jax-trace-boatstep-annotate-interp"):
        state = j_integrate_steps(boat_state, sim_state.force_model, wind_velocity, dt, N)

    with jax.profiler.trace("jax-trace-windstep-annotate-interp"):
        step_wind_and_boats_with_interaction_multiple(boats_state, sim_state.force_model, sim_state.wind_state,
                                                      sim_state.wind_params, dt, rng, n=1)
        print("done")


def test_jac():
    rng = jax.random.PRNGKey(0)
    sim_state = init_simulation_state(rng=rng)
    boat_state = sim_state.boat_state
    boat_state = jax.tree.map(lambda x: x[0], boat_state)
    wind_velocity = evaluate_wind(sim_state.wind_state, boat_state.particle_state.x)
    step_uncontrolled_jac(boat_state, sim_state.force_model, wind_velocity, 0.1)


if __name__ == "__main__":
    test_e2e()
