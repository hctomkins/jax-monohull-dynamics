import pickle
import time
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pyglet

from monohull_dynamics.demo.overlays import BoatDemoOverlays, Boat
from monohull_dynamics.dynamics.boat import BoatState
from monohull_dynamics.dynamics.boat_wind_interaction import integrate_wind_and_boats_with_interaction_multiple, set_boats_foil_angles
from monohull_dynamics.dynamics.particle import ParticleState
from monohull_dynamics.dynamics.wind import WindParams, WindState, default_wind_params, default_wind_state, \
    evaluate_wind, evaluate_wind_points
from monohull_dynamics.forces.boat import (
    DUMMY_DEBUG_DATA,
    BoatData,
    init_firefly,
)

RESOLUTION = 800
SCALE_M = 30

PYTHON_DT = 0.01
JAX_INNER_N = 1
STATE_CACHE = [None, None, None, None, None]

def get_sail_and_rudder(keys, up_key, down_key, left_key, right_key):
    if keys[left_key]:
        rudder_delta = -jnp.deg2rad(2)
    elif keys[right_key]:
        rudder_delta = jnp.deg2rad(2)
    else:
        rudder_delta = 0
    if keys[up_key]:
        new_sail = jnp.pi / 16
    elif keys[down_key]:
        new_sail = jnp.pi / 2
    else:
        new_sail = jnp.pi / 4
    return new_sail, rudder_delta


class SimulationState(typing.NamedTuple):
    force_model: BoatData
    boats_state: BoatState
    wind_params: WindParams
    wind_state: WindState


@dataclass
class MutableSimulationState:
    state: SimulationState
    rng: jnp.ndarray
    start_time: float


def init_simulation_state(rng) -> SimulationState:
    wind_params = default_wind_params(bbox_lims=jnp.array([-100, 100, -100, 100]))
    wind_state = default_wind_state(wind_params, rng)
    boat_state = BoatState(
        particle_state=ParticleState(
            m=jnp.array(100.0),
            I=jnp.array(100.0),
            x=jnp.array([0.0, 0.0]),
            xdot=jnp.array([0.0, 0.0]),
            theta=jnp.array(0.0),
            thetadot=jnp.array(0.0),
        ),
        rudder_angle=jnp.array(0.0),
        sail_angle=jnp.array(0.0),
        debug_data=DUMMY_DEBUG_DATA,
    )
    # Extra dim for multiple boats - 1 boat for now
    boats_state = jax.tree.map(lambda x: x[None], boat_state)
    return SimulationState(force_model=init_firefly(), boats_state=boats_state, wind_params=wind_params, wind_state=wind_state)


def sim_step(measured_dt: float, global_state: MutableSimulationState, keys, overlays: BoatDemoOverlays, boat_sprite: Boat):
    physics_dt = min(0.1, measured_dt)

    # get
    sim_state = global_state.state
    rng = global_state.rng
    rng, _ = jax.random.split(rng)

    boats_state = sim_state.boats_state
    local_wind_velocity = evaluate_wind_points(sim_state.wind_state, boats_state.particle_state.x)  # [B, 2]

    # python update
    new_sail, rudder_delta = get_sail_and_rudder(keys, pyglet.window.key.UP, pyglet.window.key.DOWN, pyglet.window.key.LEFT, pyglet.window.key.RIGHT)
    new_rudder = boats_state.rudder_angle + rudder_delta
    new_sail = jnp.ones_like(new_rudder) * new_sail


    boats_state = set_boats_foil_angles(
        boats_state,
        new_rudder,
        new_sail,
        local_wind_velocity
    )
    sim_state = sim_state._replace(boats_state=boats_state)
    # STATE_CACHE.pop(-1)
    # STATE_CACHE.insert(0, dict(sim_state=sim_state, rng=rng, physics_dt=physics_dt, inner_n=JAX_INNER_N))
    # if jnp.any(jnp.isnan(boat_state.particle_state.x)) or keys[pyglet.window.key.R]:
    #     with open("state_dump.pkl", "wb") as f:
    #         pickle.dump(STATE_CACHE, f)
    #     raise ValueError("NaN in state or exit on dump")

    # JAX update
    new_boats_state, new_wind_state, rng, _ = integrate_wind_and_boats_with_interaction_multiple(
        boats_state=sim_state.boats_state,
        force_model=sim_state.force_model,
        wind_state=sim_state.wind_state,
        wind_params=sim_state.wind_params,
        integration_dt=physics_dt / JAX_INNER_N,
        rng=rng,
        n_integrations_per_wind_step=JAX_INNER_N,
        n_wind_equilibrium_steps=1,
        integrator="i4",
    )
    # boats_state = j_integrate_many(boats_state, sim_state.force_model, sim_state.wind_velocity, dt / JAX_INNER_N, JAX_INNER_N)

    # set
    new_sim_state = sim_state._replace(boats_state=new_boats_state, wind_state=new_wind_state)

    return new_sim_state, rng


def run_demo():
    pyglet.resource.path = ["."]
    pyglet.resource.reindex()
    window = pyglet.window.Window(800, 800)
    keys = pyglet.window.key.KeyStateHandler()
    window.push_handlers(keys)

    rng = jax.random.PRNGKey(0)
    global_state = MutableSimulationState(state=init_simulation_state(rng), rng=rng, start_time=time.time())

    # Debug data
    boat_sprite = Boat(SCALE_M, RESOLUTION)
    overlays = BoatDemoOverlays(RESOLUTION)

    # Time and position text
    def step_physics(dt):
        if keys[pyglet.window.key.SPACE]:
            return

        global_state.state, global_state.rng = sim_step(dt, global_state, keys, overlays, boat_sprite)

    pyglet.clock.schedule_interval(step_physics, PYTHON_DT)

    @window.event
    def on_draw():
        debug_data = global_state.state.boats_state.debug_data
        debug_data = jax.tree.map(lambda x: x[0], debug_data)

        # Overlays
        wind_strength = evaluate_wind(global_state.state.wind_state, global_state.state.boats_state.particle_state.x[0])
        overlays.update_data(debug_data, global_state.state.boats_state.particle_state.x[0], wind_strength, time.time() - global_state.start_time)
        boat_sprite.update_data(
            x=global_state.state.boats_state.particle_state.x[0],
            theta=global_state.state.boats_state.particle_state.theta[0],
            sail_angle=global_state.state.boats_state.sail_angle[0],
            rudder_angle=global_state.state.boats_state.rudder_angle[0],
        )

        window.clear()
        boat_sprite.draw()
        overlays.draw()

    pyglet.app.run()


if __name__ == "__main__":
    run_demo()
