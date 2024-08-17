import pyglet

from forces.boat import FireflyPhysics
from dynamics.particle import Particle2D
import numpy as np
import time
RESOLUTION = 800
SCALE_M = 20


def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


# Replace with your own simulation state holder
class BoatState:
    def __init__(self):
        self.force_model = FireflyPhysics()
        self.particle_state = Particle2D(m=100, I = 100, x=(0.0, 0.0), xdot=(0.0, 0.0), theta=0.0, thetadot=0.0)
        self.rudder_angle = 0 #np.pi/8
        self.sail_angle = np.pi/4

def world_to_canvas(position: tuple[float, float]) -> tuple[float, float]:
    return ((np.array(position) / SCALE_M) + 0.5) * RESOLUTION


if __name__ == "__main__":
    pyglet.resource.path = ['.']
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
    ego_rudder = pyglet.shapes.Line(0, 0, -20*sf, 0, width=5)
    ego_rudder.anchor_x = 70*sf
    ego_rudder.anchor_y = 0
    ego_rudder.scale = sf
    ego_sail = pyglet.shapes.Line(0, 0, -80*sf, 0, width=2)
    ego_sail.anchor_x = 0
    ego_sail.anchor_y = 0
    boat_state = BoatState()

    def step_physics(dt):
        t0=time.time()
        force, moment = boat_state.force_model.forces_and_moments(
            boat_velocity=boat_state.particle_state.xdot,
            wind_velocity=(0, -4),
            boat_theta=boat_state.particle_state.theta,
            boat_theta_dot=boat_state.particle_state.thetadot,
            sail_angle=boat_state.sail_angle,
            rudder_angle=boat_state.rudder_angle,
        )
        print(f"Time taken: {time.time()-t0}")
        boat_state.particle_state.step(force, moment, dt=dt)

    pyglet.clock.schedule_interval(step_physics, 0.001)


    @window.event
    def on_draw():
        window.clear()
        if keys[pyglet.window.key.LEFT]:
            boat_state.rudder_angle = - np.pi/3
        elif keys[pyglet.window.key.RIGHT]:
            boat_state.rudder_angle = np.pi/3
        else:
            boat_state.rudder_angle = 0

        if keys[pyglet.window.key.UP]:
            boat_state.sail_angle = np.pi/16
        elif keys[pyglet.window.key.DOWN]:
            boat_state.sail_angle = np.pi/2
        else:
            boat_state.sail_angle = np.pi/4

        sprite_position = world_to_canvas(boat_state.particle_state.x)
        ego_sprite.x = sprite_position[0]
        ego_sprite.y = sprite_position[1]
        ego_sprite.rotation = 90 - np.rad2deg(boat_state.particle_state.theta)
        ego_sprite.draw()
        rudder_position = world_to_canvas(boat_state.particle_state.x)
        ego_rudder.x = rudder_position[0]
        ego_rudder.y = rudder_position[1]
        ego_rudder.rotation = - np.rad2deg(boat_state.particle_state.theta + boat_state.rudder_angle)
        ego_rudder.draw()
        sail_position = world_to_canvas(boat_state.particle_state.x)
        ego_sail.x = sail_position[0]
        ego_sail.y = sail_position[1]
        ego_sail.rotation = - np.rad2deg(boat_state.particle_state.theta + boat_state.sail_angle)
        ego_sail.draw()




    pyglet.app.run()