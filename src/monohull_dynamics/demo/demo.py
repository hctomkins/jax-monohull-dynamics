import pickle
import time
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pyglet

from monohull_dynamics.demo.overlays import BoatDemoOverlays, BoatOne
from monohull_dynamics.dynamics.boat import BoatState
from monohull_dynamics.dynamics.boat_wind_interaction import step_wind_and_boats_with_interaction_multiple
from monohull_dynamics.dynamics.particle import ParticleState
from monohull_dynamics.dynamics.wind import WindParams, WindState, default_wind_params, default_wind_state, evaluate_wind
from monohull_dynamics.forces.boat import (
    DUMMY_DEBUG_DATA,
    BoatData,
    init_firefly,
)

RESOLUTION = 800
SCALE_M = 30

PYTHON_DT = 0.01
JAX_INNER_N = 10
STATE_CACHE = [None, None, None, None, None]


class SimulationState(typing.NamedTuple):
    force_model: BoatData
    boat_state: BoatState
    wind_params: WindParams
    wind_state: WindState


@dataclass
class MutableSimulationState:
    state: SimulationState
    rng: jnp.ndarray
    start_time: float


def init_simulation_state(rng):
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
    boat_state = jax.tree.map(lambda x: x[None], boat_state)
    return SimulationState(force_model=init_firefly(), boat_state=boat_state, wind_params=wind_params, wind_state=wind_state)


def sim_step(measured_dt: float, global_state: MutableSimulationState, keys, overlays: BoatDemoOverlays, boat_sprite: BoatOne):
    physics_dt = min(0.1, measured_dt)

    # get
    sim_state = global_state.state
    rng = global_state.rng
    rng, _ = jax.random.split(rng)

    boat_state = sim_state.boat_state
    boat_state_0 = jax.tree.map(lambda x: x[0], boat_state)
    particle_state_0 = boat_state_0.particle_state
    boat_vector = jnp.array(
        [
            jnp.cos(particle_state_0.theta),
            jnp.sin(particle_state_0.theta),
        ]
    )
    local_wind_velocity = evaluate_wind(sim_state.wind_state, particle_state_0.x)  # [B, 2]

    # python update
    sail_sign = -jnp.sign(jnp.cross(boat_vector, local_wind_velocity))

    if keys[pyglet.window.key.LEFT]:
        new_rudder = boat_state.rudder_angle - jnp.deg2rad(2)
    elif keys[pyglet.window.key.RIGHT]:
        new_rudder = boat_state.rudder_angle + jnp.deg2rad(2)
    else:
        new_rudder = boat_state.rudder_angle
    new_rudder = jnp.clip(new_rudder, -jnp.pi / 4, jnp.pi / 4)

    if keys[pyglet.window.key.UP]:
        new_sail = jnp.pi / 16 * sail_sign[None]
    elif keys[pyglet.window.key.DOWN]:
        new_sail = jnp.pi / 2 * sail_sign[None]
    else:
        new_sail = jnp.pi / 4 * sail_sign[None]

    boat_state = boat_state._replace(rudder_angle=new_rudder, sail_angle=new_sail)
    sim_state = sim_state._replace(boat_state=boat_state)
    STATE_CACHE.pop(-1)
    STATE_CACHE.insert(0, dict(sim_state=sim_state, rng=rng, physics_dt=physics_dt, inner_n=JAX_INNER_N))
    if jnp.any(jnp.isnan(boat_state.particle_state.x)) or keys[pyglet.window.key.R]:
        with open("state_dump.pkl", "wb") as f:
            pickle.dump(STATE_CACHE, f)
        raise ValueError("NaN in state or exit on dump")

    # JAX update
    new_boat_state, new_wind_state, rng, _ = step_wind_and_boats_with_interaction_multiple(
        boats_state=sim_state.boat_state,
        force_model=sim_state.force_model,
        wind_state=sim_state.wind_state,
        wind_params=sim_state.wind_params,
        inner_dt=physics_dt / JAX_INNER_N,
        rng=rng,
        n=JAX_INNER_N,
    )
    # boat_state = j_integrate_many(boat_state, sim_state.force_model, sim_state.wind_velocity, dt / JAX_INNER_N, JAX_INNER_N)

    # set
    new_sim_state = sim_state._replace(boat_state=new_boat_state, wind_state=new_wind_state)

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
    boat_sprite = BoatOne(SCALE_M, RESOLUTION)
    overlays = BoatDemoOverlays(RESOLUTION)

    # Time and position text
    def step_physics(dt):
        if keys[pyglet.window.key.SPACE]:
            return

        global_state.state, global_state.rng = sim_step(dt, global_state, keys, overlays, boat_sprite)

    pyglet.clock.schedule_interval(step_physics, PYTHON_DT)

    @window.event
    def on_draw():
        debug_data = global_state.state.boat_state.debug_data
        debug_data = jax.tree.map(lambda x: x[0], debug_data)

        # Overlays
        wind_strength = evaluate_wind(global_state.state.wind_state, global_state.state.boat_state.particle_state.x[0])
        overlays.update_data(debug_data, global_state.state.boat_state.particle_state.x[0], wind_strength, time.time() - global_state.start_time)
        boat_sprite.update_data(
            x=global_state.state.boat_state.particle_state.x[0],
            theta=global_state.state.boat_state.particle_state.theta[0],
            sail_angle=global_state.state.boat_state.sail_angle[0],
            rudder_angle=global_state.state.boat_state.rudder_angle[0],
        )

        window.clear()
        boat_sprite.draw()
        overlays.draw()

    pyglet.app.run()


if __name__ == "__main__":
    # with jax.disable_jit():
    run_demo()
