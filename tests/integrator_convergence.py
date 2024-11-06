import pickle

import jax
import numpy as np
from matplotlib import pyplot as plt

from monohull_dynamics.demo.demo import SimulationState
from monohull_dynamics.dynamics.boat_wind_interaction import integrate_wind_and_boats_with_interaction_multiple

_ = SimulationState

with open("../src/monohull_dynamics/demo/state_dump.pkl", "rb") as f:
    states = pickle.load(f)
    # dict(sim_state=sim_state, rng=rng, physics_dt=physics_dt, inner_n=JAX_INNER_N)

#######################################
sim_state = states[-1]["sim_state"]
rng = states[-1]["rng"]
# physics_dt = states[-1]["physics_dt"]
# JAX_INNER_N = states[-1]["inner_n"]
# substeps = 1


# Integrate precisely
boat_state = sim_state.boat_state
wind_state = sim_state.wind_state
substeps = 300
JAX_INNER_N = 1


xs_int = [boat_state.particle_state.x]
thetas_int = [boat_state.particle_state.theta]
t = [0]
dt = 0.001 # physics_dt / 50

for i in range(substeps):
    boat_state, new_wind_state, rng, wind_offsets = integrate_wind_and_boats_with_interaction_multiple(
        boats_state=boat_state,
        force_model=sim_state.force_model,
        wind_state=sim_state.wind_state,
        wind_params=sim_state.wind_params,
        integration_dt=dt,
        rng=rng,
        n_integrations_per_wind_step=1,
        n_wind_equilibrium_steps=1,
        integrator="rk4"
    )
    xs_int.append(boat_state.particle_state.x)
    thetas_int.append(boat_state.particle_state.theta)
    t.append(t[-1] + dt)

xs_int = np.array(xs_int)
thetas_int = np.array(thetas_int)
x_offset = np.min(xs_int, axis=0, keepdims=True)
theta_offset = np.min(thetas_int, axis=0, keepdims=True)

xs_int = xs_int - x_offset
thetas_int = thetas_int - theta_offset
t_int = np.array(t)

# Extrapolate integrator single step
print(boat_state.particle_state.x.shape, "init shape")
for integrator in ["i4", "rk4", "euler"]:
    boat_state = sim_state.boat_state
    wind_state = sim_state.wind_state
    xs_ss = [boat_state.particle_state.x]
    thetas_ss = [boat_state.particle_state.theta]
    for dt in t_int[1:]:
        res_boat_state, new_wind_state, rng, wind_offsets = integrate_wind_and_boats_with_interaction_multiple(
            boats_state=boat_state,
            force_model=sim_state.force_model,
            wind_state=sim_state.wind_state,
            wind_params=sim_state.wind_params,
            integration_dt=dt,
            rng=rng,
            n_integrations_per_wind_step=1,
            n_wind_equilibrium_steps=1,
            integrator=integrator,
        )
        xs_ss.append(res_boat_state.particle_state.x)
        thetas_ss.append(res_boat_state.particle_state.theta)

    xs_ss = np.array(xs_ss)
    thetas_ss = np.array(thetas_ss)

    xs_ss = xs_ss - x_offset
    thetas_ss = thetas_ss - theta_offset

    xs_ss[np.abs(xs_ss) > 10] = np.nan
    thetas_ss[np.abs(thetas_ss) > 10] = np.nan

    plt.plot(t_int, np.abs(xs_ss[:, 0, 0] - xs_int[:,0,0]), label=f"x_{integrator}")#, linestyle="--", marker="o")

import time

for integrator in ["i4", "rk4", "euler"]:
    boat_state = sim_state.boat_state
    wind_state = sim_state.wind_state
    res_boat_state, new_wind_state, rng, wind_offsets = integrate_wind_and_boats_with_interaction_multiple(
        boats_state=boat_state,
        force_model=sim_state.force_model,
        wind_state=sim_state.wind_state,
        wind_params=sim_state.wind_params,
        integration_dt=0.001,
        rng=rng,
        n_integrations_per_wind_step=100,
        n_wind_equilibrium_steps=1,
        integrator=integrator,
    )
    t0 = time.time()
    res_boat_state, new_wind_state, rng, wind_offsets = integrate_wind_and_boats_with_interaction_multiple(
        boats_state=boat_state,
        force_model=sim_state.force_model,
        wind_state=sim_state.wind_state,
        wind_params=sim_state.wind_params,
        integration_dt=0.001,
        rng=rng,
        n_integrations_per_wind_step=100,
        n_wind_equilibrium_steps=1,
        integrator=integrator,
    )
    jax.block_until_ready(res_boat_state)
    print(f"{integrator} took {(time.time() - t0) / 100:.5f} per step")

ax = plt.gca()
ax.set_yscale('log')

plt.xlabel("dt (s)")
plt.ylabel("Error (m)")
plt.title("Integrator single step error vs dt")
plt.legend()

plt.show()
