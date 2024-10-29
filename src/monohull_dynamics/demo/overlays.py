import jax.numpy as jnp
import pyglet


def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


def world_to_canvas(position: jnp.ndarray, scale, resolution) -> jnp.ndarray:
    return ((position / scale) + 0.5) * resolution


class BoatOne:
    def __init__(self, scale_m, resolution):
        self.draw_batch = pyglet.graphics.Batch()
        self.scale = scale_m
        self.resolution = resolution

        boat_image = pyglet.resource.image("boat.png")
        center_image(boat_image)
        self.ego_sprite = pyglet.sprite.Sprite(boat_image, batch=self.draw_batch)
        h = self.ego_sprite.height
        h_m = h * scale_m / resolution
        sf = 3.8 / h_m
        self.ego_sprite.scale = sf
        self.ego_rudder = pyglet.shapes.Line(0, 0, -20 * sf, 0, width=5, color=(0, 0, 255), batch=self.draw_batch)
        self.ego_rudder.anchor_x = 70 * sf
        self.ego_rudder.anchor_y = 0
        self.ego_rudder.scale = sf
        self.ego_sail = pyglet.shapes.Line(0, 0, -80 * sf, 0, width=2, color=(0, 255, 0), batch=self.draw_batch)
        self.ego_sail.anchor_x = 0
        self.ego_sail.anchor_y = 0

    def update_data(self, x, theta, sail_angle, rudder_angle):
        sprite_position = world_to_canvas(x, self.scale, self.resolution)
        self.ego_sprite.x = sprite_position[0]
        self.ego_sprite.y = sprite_position[1]
        self.ego_sprite.rotation = 90 - jnp.rad2deg(theta)
        self.ego_sprite.draw()
        rudder_position = world_to_canvas(x, self.scale, self.resolution)
        self.ego_rudder.x = rudder_position[0]
        self.ego_rudder.y = rudder_position[1]
        self.ego_rudder.rotation = -jnp.rad2deg(theta + rudder_angle)
        self.ego_rudder.draw()
        sail_position = world_to_canvas(x, self.scale, self.resolution)
        self.ego_sail.x = sail_position[0]
        self.ego_sail.y = sail_position[1]
        self.ego_sail.rotation = -jnp.rad2deg(theta + sail_angle)
        self.ego_sail.draw()

    def draw(self):
        self.draw_batch.draw()


class BoatDemoOverlays:
    def __init__(self, resolution):
        self.draw_batch = pyglet.graphics.Batch()

        self.board_moment_widget = pyglet.shapes.Sector(x=30, y=resolution - 30, radius=20, angle=0, color=(255, 0, 0), batch=self.draw_batch)
        self.rudder_moment_widget = pyglet.shapes.Sector(x=100, y=resolution - 30, radius=20, angle=0, color=(0, 0, 255), batch=self.draw_batch)
        self.sail_moment_widget = pyglet.shapes.Sector(x=170, y=resolution - 30, radius=20, angle=0, color=(0, 255, 0), batch=self.draw_batch)

        self.board_force_widget = pyglet.shapes.Line(30, resolution - 100, 0, 0, width=2, color=(255, 0, 0), batch=self.draw_batch)
        self.rudder_force_widget = pyglet.shapes.Line(100, resolution - 100, 0, 0, width=2, color=(0, 0, 255), batch=self.draw_batch)
        self.sail_force_widget = pyglet.shapes.Line(170, resolution - 100, 0, 0, width=2, color=(0, 255, 0), batch=self.draw_batch)
        self.wind_widget = pyglet.shapes.Line(170 + 70, resolution - 100, 0, 0, width=2, color=(255, 255, 0), batch=self.draw_batch)
        self.time_label = pyglet.text.Label(
            "Time: 0", font_name="Times New Roman", font_size=12, x=10, y=10, anchor_x="left", anchor_y="bottom", batch=self.draw_batch
        )
        self.position_label = pyglet.text.Label(
            "Position: 0", font_name="Times New Roman", font_size=12, x=10, y=30, anchor_x="left", anchor_y="bottom", batch=self.draw_batch
        )

    def update_data(self, debug_data: dict, position: jnp.ndarray, wind_velocity: jnp.ndarray, runtime: float):
        self.board_moment_widget.angle = jnp.deg2rad(360 * debug_data["moments"]["board"] / 800)
        self.rudder_moment_widget.angle = jnp.deg2rad(360 * debug_data["moments"]["rudder"] / 800)
        self.sail_moment_widget.angle = jnp.deg2rad(360 * debug_data["moments"]["sail"] / 800)
        self.board_force_widget.x2 = self.board_force_widget.x + debug_data["forces"]["board"][0]
        self.board_force_widget.y2 = self.board_force_widget.y + debug_data["forces"]["board"][1]
        self.board_force_widget.x2 = self.board_force_widget.x + debug_data["forces"]["hull"][0]
        self.board_force_widget.y2 = self.board_force_widget.y + debug_data["forces"]["hull"][1]
        self.rudder_force_widget.x2 = self.rudder_force_widget.x + debug_data["forces"]["rudder"][0]
        self.rudder_force_widget.y2 = self.rudder_force_widget.y + debug_data["forces"]["rudder"][1]
        self.sail_force_widget.x2 = self.sail_force_widget.x + debug_data["forces"]["sail"][0]
        self.sail_force_widget.y2 = self.sail_force_widget.y + debug_data["forces"]["sail"][1]
        self.wind_widget.x2 = self.wind_widget.x + wind_velocity[0]
        self.wind_widget.y2 = self.wind_widget.y + wind_velocity[1]

        self.time_label.text = f"Time: {runtime:.2f}"
        self.position_label.text = f"Position: {position[0]:.2f}, {position[1]:.2f}"

    def draw(self):
        self.draw_batch.draw()


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
