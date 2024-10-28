from monohull_dynamics.dynamics.boat_wind_interaction import we_grid
from monohull_dynamics.dynamics.wind import wind_spawn, WindParams, N_LOCAL_GUSTS, WindState, step_wind_state, \
    evaluate_wind_grid, evaluate_wind, default_wind_state
import jax.numpy as jnp
import jax

from monohull_dynamics.forces.boat import forces_and_moments, init_firefly


def demo_wind_spawn():
    import numpy as np
    from matplotlib import pyplot as plt

    rng = jax.random.split(jax.random.PRNGKey(30), 10)
    spawn_many = jax.jit(jax.vmap(wind_spawn, in_axes=(None, 0, 0)))


    # Arrow length
    arrow_length = 3
    angle_degrees = -90
    angle_radians = jnp.deg2rad(angle_degrees)

    spawn_points = spawn_many(
        jnp.array([-15,15,-15,15]),
        jnp.ones(10) * angle_degrees,
        rng
    )  # [10, 2]
    spawn_points = np.array(spawn_points)

    # Calculate the direction of the arrow (using the angle)
    arrow_dx = arrow_length * np.cos(angle_radians)
    arrow_dy = arrow_length * np.sin(angle_radians)

    # Plotting
    fig, ax = plt.subplots()

    # Plot the bounding box
    box = plt.Polygon([[-15, 15], [15, 15], [15, -15], [-15, -15]], closed=True, fill=None, edgecolor='blue')
    ax.add_patch(box)

    # Plot the spawn points
    ax.scatter(spawn_points[:, 0], spawn_points[:, 1], color='red', label='Spawn Points')

    # Plot the arrow indicating the direction of the angle
    ax.arrow(0, 0, arrow_dx, arrow_dy, head_width=0.3, head_length=0.5, fc='green', ec='green',
             label=f'Angle: {angle_degrees}°')

    # Formatting the plot
    ax.set_aspect('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()

    plt.title(f'Bounding Box, Spawn Points, and Arrow at {angle_degrees}°')
    plt.show()

def animate_wind_grid():
    import cv2
    import numpy as np
    # Initialize wind parameters
    R_m = 15
    params = WindParams(
        base_theta_deg=jnp.array(-90),
        base_r=jnp.array(5.0),  # 10 knots
        theta_oscillation_amplitude_deg=jnp.array(0),#10.0),
        theta_oscillation_period_s=jnp.array(60.0),
        r_oscillation_amplitude=jnp.array(0),#2.0),
        r_oscillation_period_s=jnp.array(60.0),
        local_gust_strength_offset_std=jnp.array(0),#1.0),
        local_gust_theta_deg_offset_std=jnp.array(0),#5.0),
        local_gust_radius_mean=jnp.array(10.0),
        local_gust_radius_std=jnp.array(5),
        bbox_lims=jnp.array([-R_m, R_m, -R_m, R_m])
    )

    # Initialize wind state
    rng = jax.random.PRNGKey(0)
    # spawn first gust centers uniformly in the box
    initial_state = default_wind_state(params, rng)
    XRES = 1000
    YRES = 1000
    XNUM = 20
    YNUM = 20

    # Create a grid of sample points
    x = jnp.linspace(-R_m, R_m, XNUM)
    y = jnp.linspace(-R_m, R_m, YNUM)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([xx, yy], axis=-1)

    # Initialize OpenCV window
    cv2.namedWindow('Wind Field', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    # Animation loop
    firefly = init_firefly()
    state = initial_state
    dt = jnp.array(0.1)
    for _ in range(1000):
        # Step the wind state
        for i in range(10):
            state, rng = step_wind_state(state, rng, dt, params)

        # Evaluate wind at grid points
        wind_vectors = evaluate_wind_grid(state, grid_points)
        boat_wind = evaluate_wind(state, jnp.array([0, 0]))
        f, m, dd = forces_and_moments(
            boat_data=firefly,
            boat_velocity=jnp.array([0, 0]),
            wind_velocity=boat_wind,
            boat_theta=jnp.array(0),
            boat_theta_dot=jnp.array(0),
            sail_angle=jnp.array(0),
            rudder_angle=jnp.array(0),
        )
        wind_offset, _ = we_grid(
            f,
            jnp.array(8.0),
            jnp.array(0),
            boat_wind,
            grid_points
        )
        # from matplotlib import pyplot as plt
        # plt.imshow(jnp.linalg.norm(wind_offset, axis=-1))
        wind_vectors = wind_vectors + wind_offset
        # plt.imshow(wind_vectors[:, :, 1])
        # plt.colorbar()
        # plt.show()

        # Create an image to draw the vector field
        img = np.zeros((XRES, YRES, 3), dtype=np.uint8)

        # Scale and draw the vectors
        # imshow coordinate systems - +ve Y is down the page
        for i in range(XNUM):
            for j in range(YNUM):
                # Arrow for field
                start_point = (
                    XRES // 4 + int((xx[i, j] + R_m) * XRES / (R_m * 4)),
                    YRES-(YRES // 4 + int((yy[i, j] + R_m) * YRES / (R_m * 4)))
                )
                end_point = (
                    start_point[0] + int(wind_vectors[i, j, 0] * 10),
                    start_point[1] - int(wind_vectors[i, j, 1] * 10)
                )
                # print(wind_vectors[i, j, 1])
                cv2.arrowedLine(img, start_point, end_point, (0, 255, 0), 1, tipLength=0.1)
                wind_strength = jnp.linalg.norm(wind_vectors[i, j])
                text_position = (end_point[0] + 5, end_point[1] - 5)
                cv2.putText(img, f'{wind_strength:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


        # Cross and circle at gust center and radius:
        for i in range(N_LOCAL_GUSTS):
            center = (
                XRES // 4 + int((state.local_gust_centers[i, 0] + R_m) * XRES / (R_m * 4)),
                YRES-(YRES // 4 + int((state.local_gust_centers[i, 1] + R_m) * YRES / (R_m * 4)))
            )
            cv2.circle(img, center, int(state.local_gust_effect_radius[i] * 10), (255, 0, 0), 1)
            cv2.drawMarker(img, center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

            # red arrow in gust direction and strength
            gust_end_point = (
                int(center[0] + state.local_gust_strengths[i] * jnp.cos(
                    jnp.radians(state.local_gust_theta_deg[i])) * 10),
                int(center[1] - state.local_gust_strengths[i] * jnp.sin(
                    jnp.radians(state.local_gust_theta_deg[i])) * 10),
            )
            cv2.arrowedLine(img, center, gust_end_point, (0, 0, 255), 1, tipLength=0.1)

        # Display the image
        cv2.imshow('Wind Field', img)
        if cv2.waitKey(30) & 0xFF == 27:  # Exit on ESC key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # demo_wind_spawn()
    animate_wind_grid()