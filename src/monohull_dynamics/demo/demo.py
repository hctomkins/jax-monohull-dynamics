from dataclasses import dataclass

import pyglet
import typing
import jax.numpy as jnp
import jax
jax.config.update('jax_numpy_dtype_promotion', 'strict')

from monohull_dynamics.dynamics.particle import ParticleState, integrate
from monohull_dynamics.forces.boat import BoatData, init_firefly, forces_and_moments, DUMMY_DEBUG_DATA

RESOLUTION = 800
SCALE_M = 30

PYTHON_DT = 0.01
JAX_INNER_N = 10000


def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

class SimulationState(typing.NamedTuple):
    force_model: BoatData
    particle_state: ParticleState
    wind_velocity: jnp.ndarray
    rudder_angle: float
    sail_angle: float
    sail_sign: int
    debug_data: dict
    
@dataclass
class MutableSimulationState:
    state: SimulationState

def init_simulation_state():
    return SimulationState(
        force_model=init_firefly(),
        particle_state=ParticleState(
            m=jnp.array(1.0),
            I=jnp.array(1.0),
            x=jnp.array([0.0,0.0]),
            xdot=jnp.array([0.0, 0.0]),
            theta=jnp.array(0.0),
            thetadot=jnp.array(0.0),
        ),
        rudder_angle=0.0,
        sail_angle=0.0,
        sail_sign=1.0,
        wind_velocity=jnp.array([0.0, -4.0]),
        debug_data=DUMMY_DEBUG_DATA
    )

def step_uncontrolled(simulation_state: SimulationState, inner_dt) -> SimulationState:
    particle_state = simulation_state.particle_state
    f, m, dd = forces_and_moments(
        boat_data=simulation_state.force_model,
        boat_velocity=particle_state.xdot,
        wind_velocity=simulation_state.wind_velocity,
        boat_theta=particle_state.theta,
        boat_theta_dot=particle_state.thetadot,
        sail_angle=simulation_state.sail_angle,
        rudder_angle=simulation_state.rudder_angle,
    )
    # jax.debug.print("force {f}", f=f)
    return simulation_state._replace(particle_state=integrate(particle_state, f, m, inner_dt), debug_data=dd)

j_step_uncontrolled = jax.jit(step_uncontrolled, static_argnums=(1,))

def integrate_many(simulation_state: SimulationState, inner_dt, N) -> SimulationState:
    body_fn = lambda i, rolled_state: step_uncontrolled(rolled_state, inner_dt)
    return jax.lax.fori_loop(0, N, body_fn, simulation_state)

j_integrate_many = jax.jit(integrate_many, static_argnums=(2))

def integrate_many_debug(simulation_state: SimulationState, inner_dt, N) -> SimulationState:
    # print("Integrating")
    for i in range(N):
        simulation_state = j_step_uncontrolled(simulation_state, inner_dt)
        # print(simulation_state.particle_state.x)
    return simulation_state


def world_to_canvas(position: jnp.ndarray) -> jnp.ndarray:
    return ((position / SCALE_M) + 0.5) * RESOLUTION


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
    global_state = MutableSimulationState(state=init_simulation_state())

    # Debug data
    board_moment_widget = pyglet.shapes.Sector(
        x=30, y=RESOLUTION - 30, radius=20, angle=jnp.deg2rad(0), color=(255, 0, 0)
    )
    rudder_moment_widget = pyglet.shapes.Sector(
        x=100, y=RESOLUTION - 30, radius=20, angle=jnp.deg2rad(0), color=(0, 0, 255)
    )
    sail_moment_widget = pyglet.shapes.Sector(
        x=170, y=RESOLUTION - 30, radius=20, angle=jnp.deg2rad(0), color=(0, 255, 0)
    )

    board_force_widget = pyglet.shapes.Line(
        30, RESOLUTION - 100, 0, 0, width=2, color=(255, 0, 0)
    )
    rudder_force_widget = pyglet.shapes.Line(
        100, RESOLUTION - 100, 0, 0, width=2, color=(0, 0, 255)
    )
    sail_force_widget = pyglet.shapes.Line(
        170, RESOLUTION - 100, 0, 0, width=2, color=(0, 255, 0)
    )


    def step_physics(dt):
        # t0=time.time()
        print(dt)
        sim_state = global_state.state

        if keys[pyglet.window.key.SPACE]:
            return

        boat_vector = jnp.array([
            jnp.cos(sim_state.particle_state.theta),
            jnp.sin(sim_state.particle_state.theta),
        ])
        sim_state = sim_state._replace(sail_sign=-jnp.sign(jnp.cross(boat_vector, sim_state.wind_velocity)))
        sim_state = j_integrate_many(sim_state, dt / JAX_INNER_N, JAX_INNER_N)
        debug_data = sim_state.debug_data
        global_state.state = sim_state
        board_moment_widget.angle = jnp.deg2rad(
            360 * debug_data["moments"]["board"] / 800
        )
        rudder_moment_widget.angle = jnp.deg2rad(
            360 * debug_data["moments"]["rudder"] / 800
        )
        sail_moment_widget.angle = jnp.deg2rad(360 * debug_data["moments"]["sail"] / 800)
        board_force_widget.x2 = board_force_widget.x + debug_data["forces"]["board"][0]
        board_force_widget.y2 = board_force_widget.y + debug_data["forces"]["board"][1]
        board_force_widget.x2 = board_force_widget.x + debug_data["forces"]["hull"][0]
        board_force_widget.y2 = board_force_widget.y + debug_data["forces"]["hull"][1]
        rudder_force_widget.x2 = (
            rudder_force_widget.x + debug_data["forces"]["rudder"][0]
        )
        rudder_force_widget.y2 = (
            rudder_force_widget.y + debug_data["forces"]["rudder"][1]
        )
        sail_force_widget.x2 = sail_force_widget.x + debug_data["forces"]["sail"][0]
        sail_force_widget.y2 = sail_force_widget.y + debug_data["forces"]["sail"][1]

    pyglet.clock.schedule_interval(step_physics, PYTHON_DT)

    @window.event
    def on_draw():
        window.clear()
        sim_state = global_state.state
        if keys[pyglet.window.key.LEFT]:
            new_rudder = sim_state.rudder_angle - jnp.deg2rad(2)
            new_rudder = max(-jnp.pi / 4, new_rudder)
        elif keys[pyglet.window.key.RIGHT]:
            new_rudder = sim_state.rudder_angle + jnp.deg2rad(2)
            new_rudder = min(jnp.pi / 4, new_rudder)
        else:
            new_rudder = sim_state.rudder_angle

        if keys[pyglet.window.key.UP]:
            new_sail = jnp.pi / 16 * sim_state.sail_sign
        elif keys[pyglet.window.key.DOWN]:
            new_sail = jnp.pi / 2 * sim_state.sail_sign
        else:
            new_sail = jnp.pi / 4 * sim_state.sail_sign

        sim_state = sim_state._replace(rudder_angle=new_rudder, sail_angle=new_sail)
        global_state.state = sim_state

        sprite_position = world_to_canvas(sim_state.particle_state.x)
        ego_sprite.x = sprite_position[0]
        ego_sprite.y = sprite_position[1]
        ego_sprite.rotation = 90 - jnp.rad2deg(sim_state.particle_state.theta)
        ego_sprite.draw()
        rudder_position = world_to_canvas(sim_state.particle_state.x)
        ego_rudder.x = rudder_position[0]
        ego_rudder.y = rudder_position[1]
        ego_rudder.rotation = -jnp.rad2deg(
            sim_state.particle_state.theta + sim_state.rudder_angle
        )
        ego_rudder.draw()
        sail_position = world_to_canvas(sim_state.particle_state.x)
        ego_sail.x = sail_position[0]
        ego_sail.y = sail_position[1]
        ego_sail.rotation = -jnp.rad2deg(
            sim_state.particle_state.theta + sim_state.sail_angle
        )
        ego_sail.draw()
        board_moment_widget.draw()
        rudder_moment_widget.draw()
        sail_moment_widget.draw()
        board_force_widget.draw()
        rudder_force_widget.draw()
        sail_force_widget.draw()

    pyglet.app.run()


if __name__ == "__main__":
    # jax.disable_jit()
    run_demo()
