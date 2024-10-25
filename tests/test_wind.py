from monohull_dynamics.dynamics.wind import wind_spawn, WindParams, N_LOCAL_GUSTS, WindState, step_wind_state, \
    evaluate_wind_grid
import jax.numpy as jnp
import jax

def demo_wind_spawn():
    import numpy as np
    from matplotlib import pyplot as plt

    rng = jax.random.split(jax.random.PRNGKey(30), 10)
    spawn_many = jax.jit(jax.vmap(wind_spawn, in_axes=(None, None, None, None, 0, 0)))


    # Arrow length
    arrow_length = 3
    angle_degrees = -90
    angle_radians = jnp.deg2rad(angle_degrees)

    spawn_points = spawn_many(
        jnp.array(-15),
        jnp.array(15),
        jnp.array(-15),
        jnp.array(15),
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
    R_m = 150
    params = WindParams(
        base_theta_deg=jnp.array(45.0),
        base_r=jnp.array(5.0),  # 10 knots
        theta_oscillation_amplitude_deg=jnp.array(10.0),
        theta_oscillation_period_s=jnp.array(60.0),
        r_oscillation_amplitude=jnp.array(2.0),
        r_oscillation_period_s=jnp.array(60.0),
        local_gust_strength_offset_std=jnp.array(1.0),
        local_gust_theta_deg_offset_std=jnp.array(5.0),
        local_gust_radius_mean=jnp.array(10.0),
        local_gust_radius_std=jnp.array(5),
        bbox_lims=jnp.array([-R_m, R_m, -R_m, R_m])
    )

    # Initialize wind state
    rng = jax.random.PRNGKey(0)
    gust_rng = jax.random.split(rng, N_LOCAL_GUSTS)
    # spawn first gust centers uniformly in the box
    xmin, xmax, ymin, ymax = params.bbox_lims
    initial_gust_centers = jax.random.uniform(rng, shape=(N_LOCAL_GUSTS, 2), minval=0.0, maxval=1.0)
    initial_gust_centers = initial_gust_centers * jnp.array([xmax - xmin, ymax - ymin]) + jnp.array([xmin, ymin])

    initial_state = WindState(
        current_theta_phase=jnp.array(0.0),
        current_r_phase=jnp.array(0.0),
        current_base_r=params.base_r,
        current_base_theta=params.base_theta_deg,
        local_gust_centers=initial_gust_centers,
        local_gust_strengths=params.base_r + jax.random.normal(rng, shape=(
        N_LOCAL_GUSTS,)) * params.local_gust_strength_offset_std,
        local_gust_theta_deg=params.base_theta_deg + jax.random.normal(rng, shape=(
        N_LOCAL_GUSTS,)) * params.local_gust_theta_deg_offset_std,
        local_gust_effect_radius=jnp.maximum(params.local_gust_radius_mean + jax.random.normal(rng, shape=(
            N_LOCAL_GUSTS,)) * params.local_gust_radius_std, params.local_gust_radius_std),
        local_gust_start_points=initial_gust_centers
    )
    XRES = 1000
    YRES = 1000
    XNUM = 50
    YNUM = 50

    # Create a grid of sample points
    x = jnp.linspace(-R_m, R_m, XNUM)
    y = jnp.linspace(-R_m, R_m, YNUM)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([xx, yy], axis=-1)

    # Initialize OpenCV window
    cv2.namedWindow('Wind Field', cv2.WINDOW_NORMAL)

    # Animation loop
    state = initial_state
    dt = jnp.array(1)
    for _ in range(1000):
        # Step the wind state
        for i in range(100):
            state, rng = step_wind_state(state, rng, dt, params)

        # Evaluate wind at grid points
        wind_vectors = evaluate_wind_grid(state, grid_points)

        # Create an image to draw the vector field
        img = np.zeros((XRES, YRES, 3), dtype=np.uint8)

        # Scale and draw the vectors
        for i in range(XNUM):
            for j in range(YNUM):
                # Arrow for field
                start_point = (
                    XRES // 4 + int((xx[i, j] + R_m) * XRES / (R_m * 4)),
                    YRES // 4 + int((yy[i, j] + R_m) * YRES / (R_m * 4))
                )
                end_point = (
                    int(start_point[0] + wind_vectors[i, j, 0] * 10),
                    int(start_point[1] + wind_vectors[i, j, 1] * 10)
                )
                cv2.arrowedLine(img, start_point, end_point, (0, 255, 0), 1, tipLength=0.1)

        # Cross and circle at gust center and radius:
        for i in range(N_LOCAL_GUSTS):
            center = (
                XRES // 4 + int((state.local_gust_centers[i, 0] + R_m) * XRES / (R_m * 4)),
                YRES // 4 + int((state.local_gust_centers[i, 1] + R_m) * YRES / (R_m * 4))
            )
            cv2.circle(img, center, int(state.local_gust_effect_radius[i] * 10), (255, 0, 0), 1)
            cv2.drawMarker(img, center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

            # red arrow in gust direction and strength
            gust_end_point = (
                int(center[0] + state.local_gust_strengths[i] * jnp.cos(
                    jnp.radians(state.local_gust_theta_deg[i])) * 10),
                int(center[1] + state.local_gust_strengths[i] * jnp.sin(
                    jnp.radians(state.local_gust_theta_deg[i])) * 10),
            )
            cv2.arrowedLine(img, center, gust_end_point, (0, 0, 255), 1, tipLength=0.1)

        # Display the image
        cv2.imshow('Wind Field', img)
        if cv2.waitKey(30) & 0xFF == 27:  # Exit on ESC key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_wind_spawn()
    animate_wind_grid()