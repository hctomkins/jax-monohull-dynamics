import jax
import jax.numpy as jnp

from monohull_dynamics.dynamics.boat_wind_interaction import we_grid
from monohull_dynamics.dynamics.particle import BoatState, integrate, ParticleState
from monohull_dynamics.forces.boat import (
    forces_and_moments, DUMMY_DEBUG_DATA, init_firefly,
)
import matplotlib.pyplot as plt


def test_demo_plots():
    # Define the grid over a 15m x 15m box
    res = 50
    lim = 20
    X, Y = jnp.meshgrid(jnp.linspace(-lim, lim, res), jnp.linspace(-lim, lim, res))
    at = jnp.stack([X, Y], axis=-1)

    thetas = [0, jnp.pi / 4, jnp.pi / 2, -jnp.pi / 4]  # , jnp.pi / 4, -jnp.pi / 2, 0]

    # Create a figure with subplots
    fig, axes = plt.subplots(len(thetas), 2, figsize=(20, 20))

    for i, theta in enumerate(thetas):
        # Update the boat state with the current theta
        particle_state = ParticleState(
            m=jnp.array(100.0),
            I=jnp.array(100.0),
            x=jnp.array([0.0, 0.0]),
            xdot=jnp.array([0.0, 0.0]),
            theta=jnp.array(0.0),
            thetadot=jnp.array(0.0),
        )
        boat_state = BoatState(
            particle_state=particle_state,
            rudder_angle=0.0,
            sail_angle=theta,
            debug_data=DUMMY_DEBUG_DATA,
        )
        force_model = init_firefly()
        wind_velocity = jnp.array([0.0, -4.0])

        # Wind effect
        with jax.disable_jit():
            f, m, dd = forces_and_moments(
                boat_data=force_model,
                boat_velocity=particle_state.xdot,
                wind_velocity=wind_velocity,
                boat_theta=particle_state.theta,
                boat_theta_dot=particle_state.thetadot,
                sail_angle=boat_state.sail_angle,
                rudder_angle=boat_state.rudder_angle,
            )
            print(f)

        frontal_area = 8
        with jax.disable_jit():
            wind_effects, factor = we_grid(f, frontal_area, particle_state.theta + boat_state.sail_angle,
                                           wind_velocity, at)
        wind_effects = wind_effects

        # Plot the factor
        ax_factor = axes[i, 0]
        ax_factor.imshow(factor, origin='lower', extent=(-lim, lim, -lim, lim))
        # Add force arrow
        ax_factor.arrow(0, 0, f[0] / 10, f[1] / 10, color='r', head_width=0.5, head_length=1.0,
                        label='Force on Sail')

        ax_factor.set_xlabel('X Position (m)')
        ax_factor.set_ylabel('Y Position (m)')
        ax_factor.set_title(f'Wind Effect Factor (theta={jnp.rad2deg(theta)})')

        # Plot the quiver
        ax_quiver = axes[i, 1]
        ax_quiver.quiver(X, Y, wind_effects[..., 0], wind_effects[..., 1], scale=0.5, scale_units='xy')
        sail_direction = particle_state.theta + boat_state.sail_angle + jnp.pi
        sail_x = jnp.cos(sail_direction)
        sail_y = jnp.sin(sail_direction)

        # Plot the sail direction arrow
        ax_quiver.arrow(0, 0, sail_x, sail_y, color='r', head_width=0.5, head_length=1.0, label='Sail Direction')
        ax_quiver.set_xlabel('X Position (m)')
        ax_quiver.set_ylabel('Y Position (m)')
        ax_quiver.set_title(f'Wind Effect Grid (theta={jnp.rad2deg(theta)})')
        ax_quiver.grid(True)
        # equal aspect
        ax_quiver.set_aspect("equal")
