import typing
from dataclasses import dataclass
import time

import jax.numpy as jnp
import jax
import pyglet

from monohull_dynamics.dynamics.boat_wind_interaction import step_wind_and_boats_with_interaction_multiple
from monohull_dynamics.dynamics.particle import BoatState, ParticleState
from monohull_dynamics.dynamics.wind import default_params, default_state, WindParams, WindState, evaluate_wind_points, \
    evaluate_wind
from monohull_dynamics.forces.boat import (
    DUMMY_DEBUG_DATA,
    BoatData,
    init_firefly,
)

RESOLUTION = 800
SCALE_M = 30

PYTHON_DT = 0.01
JAX_INNER_N = 10


def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


class SimulationState(typing.NamedTuple):
    force_model: BoatData
    boat_state: BoatState
    wind_params: WindParams
    wind_state: WindState


@dataclass
class MutableSimulationState:
    state: SimulationState
    rng: jnp.ndarray


def init_simulation_state(rng):
    wind_params = default_params(bbox_lims=jnp.array([-100, 100, -100, 100]))
    wind_state = default_state(wind_params, rng)
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
    return SimulationState(
        force_model=init_firefly(),
        boat_state=boat_state,
        wind_params=wind_params,
        wind_state=wind_state
    )


def world_to_canvas(position: jnp.ndarray) -> jnp.ndarray:
    return ((position / SCALE_M) + 0.5) * RESOLUTION

# def plot_wind_arrows(wind_state):
#     N = 5  # Define the grid size
#     x = jnp.linspace(-SCALE_M/2, SCALE_M/2, N)
#     y = jnp.linspace(-SCALE_M/2, SCALE_M/2, N)
#     grid_x, grid_y = jnp.meshgrid(x, y)
#     grid_points = jnp.stack([grid_x, grid_y], axis=-1)  # Shape [N, N, 2]
#
#     wind_strengths = evaluate_wind_grid(wind_state, grid_points)  # Shape [N, N, 2]
#
#     arrows = []
#     for i in range(N):
#         for j in range(N):
#             start = world_to_canvas(grid_points[i, j])
#             end = start + wind_strengths[i, j]*3
#             arrow = pyglet.shapes.Line(start[0], start[1], end[0], end[1], width=2, color=(255, 255, 0))
#             arrows.append(arrow)
#
#     return arrows

def run_demo():
    pyglet.resource.path = ["."]
    pyglet.resource.reindex()
    window = pyglet.window.Window(800, 800)
    keys = pyglet.window.key.KeyStateHandler()
    window.push_handlers(keys)

    boat_image = pyglet.resource.image("boat.png")
    center_image(boat_image)
    ego_sprite = pyglet.sprite.Sprite(boat_image)
    h = ego_sprite.height
    h_m = h * SCALE_M / RESOLUTION
    sf = 3.8 / h_m
    ego_sprite.scale = sf
    ego_rudder = pyglet.shapes.Line(0, 0, -20 * sf, 0, width=5, color=(0, 0, 255))
    ego_rudder.anchor_x = 70 * sf
    ego_rudder.anchor_y = 0
    ego_rudder.scale = sf
    ego_sail = pyglet.shapes.Line(0, 0, -80 * sf, 0, width=2, color=(0, 255, 0))
    ego_sail.anchor_x = 0
    ego_sail.anchor_y = 0
    rng = jax.random.PRNGKey(0)
    global_state = MutableSimulationState(state=init_simulation_state(rng), rng=rng)

    # Debug data
    board_moment_widget = pyglet.shapes.Sector(x=30, y=RESOLUTION - 30, radius=20, angle=jnp.deg2rad(0), color=(255, 0, 0))
    rudder_moment_widget = pyglet.shapes.Sector(x=100, y=RESOLUTION - 30, radius=20, angle=jnp.deg2rad(0), color=(0, 0, 255))
    sail_moment_widget = pyglet.shapes.Sector(x=170, y=RESOLUTION - 30, radius=20, angle=jnp.deg2rad(0), color=(0, 255, 0))

    board_force_widget = pyglet.shapes.Line(30, RESOLUTION - 100, 0, 0, width=2, color=(255, 0, 0))
    rudder_force_widget = pyglet.shapes.Line(100, RESOLUTION - 100, 0, 0, width=2, color=(0, 0, 255))
    sail_force_widget = pyglet.shapes.Line(170, RESOLUTION - 100, 0, 0, width=2, color=(0, 255, 0))

    # Time and position text
    time_label = pyglet.text.Label(
        "Time: 0",
        font_name="Times New Roman",
        font_size=12,
        x=10,
        y=10,
        anchor_x="left",
        anchor_y="bottom",
    )
    position_label = pyglet.text.Label(
        "Position: 0",
        font_name="Times New Roman",
        font_size=12,
        x=10,
        y=30,
        anchor_x="left",
        anchor_y="bottom",
    )
    start_time = time.time()

    def step_physics(dt):
        # t0=time.time()
        dt = min(0.1, dt)
        sim_state = global_state.state
        boat_state = sim_state.boat_state
        rng = global_state.rng

        if keys[pyglet.window.key.SPACE]:
            return

        # boat_state = j_integrate_many(boat_state, sim_state.force_model, sim_state.wind_velocity, dt / JAX_INNER_N, JAX_INNER_N)
        boat_state, wind_state, rng = step_wind_and_boats_with_interaction_multiple(
            boats_state=boat_state,
            force_model=sim_state.force_model,
            wind_state=sim_state.wind_state,
            wind_params=sim_state.wind_params,
            inner_dt=dt / JAX_INNER_N,
            rng=rng,
            N=JAX_INNER_N,
        )
        sim_state = sim_state._replace(boat_state=boat_state, wind_state=wind_state)
        debug_data = sim_state.boat_state.debug_data
        debug_data = jax.tree.map(lambda x: x[0], debug_data)
        global_state.state = sim_state
        global_state.rng = rng
        board_moment_widget.angle = jnp.deg2rad(360 * debug_data["moments"]["board"] / 800)
        rudder_moment_widget.angle = jnp.deg2rad(360 * debug_data["moments"]["rudder"] / 800)
        sail_moment_widget.angle = jnp.deg2rad(360 * debug_data["moments"]["sail"] / 800)
        board_force_widget.x2 = board_force_widget.x + debug_data["forces"]["board"][0]
        board_force_widget.y2 = board_force_widget.y + debug_data["forces"]["board"][1]
        board_force_widget.x2 = board_force_widget.x + debug_data["forces"]["hull"][0]
        board_force_widget.y2 = board_force_widget.y + debug_data["forces"]["hull"][1]
        rudder_force_widget.x2 = rudder_force_widget.x + debug_data["forces"]["rudder"][0]
        rudder_force_widget.y2 = rudder_force_widget.y + debug_data["forces"]["rudder"][1]
        sail_force_widget.x2 = sail_force_widget.x + debug_data["forces"]["sail"][0]
        sail_force_widget.y2 = sail_force_widget.y + debug_data["forces"]["sail"][1]

    pyglet.clock.schedule_interval(step_physics, PYTHON_DT)

    @window.event
    def on_draw():
        window.clear()
        sim_state = global_state.state
        boat_state = sim_state.boat_state
        # TODO: Right now we only have one boat, so mix displaying [1] and [] values
        boat_state_0 = jax.tree.map(lambda x: x[0], boat_state)
        particle_state = boat_state_0.particle_state
        boat_vector = jnp.array(
            [
                jnp.cos(particle_state.theta),
                jnp.sin(particle_state.theta),
            ]
        )
        local_wind_velocity = evaluate_wind(sim_state.wind_state, particle_state.x) # [B, 2]
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

        new_boat_state = boat_state._replace(rudder_angle=new_rudder, sail_angle=new_sail)
        sim_state = sim_state._replace(boat_state=new_boat_state)
        global_state.state = sim_state

        sprite_position = world_to_canvas(particle_state.x)
        ego_sprite.x = sprite_position[0]
        ego_sprite.y = sprite_position[1]
        ego_sprite.rotation = 90 - jnp.rad2deg(particle_state.theta)
        ego_sprite.draw()
        rudder_position = world_to_canvas(particle_state.x)
        ego_rudder.x = rudder_position[0]
        ego_rudder.y = rudder_position[1]
        ego_rudder.rotation = -jnp.rad2deg(particle_state.theta + boat_state_0.rudder_angle)
        ego_rudder.draw()
        sail_position = world_to_canvas(particle_state.x)
        ego_sail.x = sail_position[0]
        ego_sail.y = sail_position[1]
        ego_sail.rotation = -jnp.rad2deg(particle_state.theta + boat_state_0.sail_angle)
        ego_sail.draw()

        time_label.text = f"Time: {time.time() - start_time:.2f}"
        position_label.text = f"Position: {particle_state.x[0]:.2f}, {particle_state.x[1]:.2f}"
        time_label.draw()
        position_label.draw()

        board_moment_widget.draw()
        rudder_moment_widget.draw()
        sail_moment_widget.draw()
        board_force_widget.draw()
        rudder_force_widget.draw()
        sail_force_widget.draw()

    pyglet.app.run()


if __name__ == "__main__":
    # with jax.disable_jit():
    run_demo()
