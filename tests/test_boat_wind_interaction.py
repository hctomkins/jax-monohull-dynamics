import jax
import jax.numpy as jnp

from monohull_dynamics.dynamics.boat_wind_interaction import we_grid, integrate_wind_and_boats_with_interaction_multiple, \
    get_sail_wind_interaction
from monohull_dynamics.dynamics.particle import ParticleState
from monohull_dynamics.dynamics.boat import BoatState
from monohull_dynamics.dynamics.wind import default_wind_state, default_wind_params, evaluate_wind, step_wind_state
from monohull_dynamics.forces.boat import (
    forces_and_moments, DUMMY_DEBUG_DATA, init_firefly,
)
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def _init_default_boat(x):
    return BoatState(
        particle_state=ParticleState(
            m=jnp.array(100.0),
            I=jnp.array(100.0),
            x=x,
            xdot=jnp.array([0.0, 0.0]),
            theta=jnp.array(0.0),
            thetadot=jnp.array(0.0),
        ),
        rudder_angle=jnp.array(0.0),
        sail_angle=jnp.array(0.0),
        debug_data=DUMMY_DEBUG_DATA,
    )

_init_default_boats = jax.vmap(_init_default_boat, in_axes=0, out_axes=0)

def get_boat_from_tree(boat_state, i):
    return jax.tree.map(lambda x: x[i], boat_state)


def test_demo_plots(plot: bool = False):
    # Define the grid over a 15m x 15m box
    res = 50
    lim = 20
    X, Y = jnp.meshgrid(jnp.linspace(-lim, lim, res), jnp.linspace(-lim, lim, res))
    at = jnp.stack([X, Y], axis=-1)

    thetas = [0, jnp.pi / 4, jnp.pi / 2, -jnp.pi / 4]  # , jnp.pi / 4, -jnp.pi / 2, 0]

    # Create a figure with subplots
    if plot:
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

        if plot:
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

    if plot:
        plt.show()


def test_boat_wind_interaction(plot: bool = False):
    force_model = init_firefly()
    N = 4
    rng = jax.random.PRNGKey(0)
    inner_dt = jnp.array(0.01)
    boat_positions = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ]
    )
    boat_states = _init_default_boats(boat_positions)
    wind_params = default_wind_params(bbox_lims=jnp.array([-100, 100, -100, 100]))
    wind_state = default_wind_state(wind_params, rng)

    bxb_wind = np.zeros((N, N, 2))
    f_y = np.zeros(N)
    wind_state_loop, _ = step_wind_state(wind_state, rng, inner_dt, wind_params)

    # Debug loop to test VMAP
    for effector in range(N):
        for affected in range(N):
            effector_boat = get_boat_from_tree(boat_states, effector)
            affected_boat = get_boat_from_tree(boat_states, affected)
            pos_offset = boat_positions[affected] - boat_positions[effector]
            f, m, dd = forces_and_moments(
                force_model,
                effector_boat.particle_state.xdot,
                evaluate_wind(wind_state_loop, effector_boat.particle_state.x),
                effector_boat.particle_state.theta,
                effector_boat.particle_state.thetadot,
                effector_boat.sail_angle,
                effector_boat.rudder_angle,
            )
            f_y[affected] = f[1]
            wind_offsets, _ = get_sail_wind_interaction(
                force_on_sail=f,
                sail_area=force_model.sail_area,
                sail_theta=effector_boat.sail_angle,
                base_wind=evaluate_wind(wind_state_loop, affected_boat.particle_state.x),
                at = pos_offset
            )
            bxb_wind[affected, effector, :] = np.array(wind_offsets)
    b_wind = bxb_wind.sum(axis=1)


    _, _, _, wind_offsets = integrate_wind_and_boats_with_interaction_multiple(
        boats_state=boat_states,
        force_model=force_model,
        wind_state=wind_state,
        wind_params=wind_params,
        integration_dt=jnp.array(0.01),
        n_wind_equilibrium_steps=1,
        n_integrations_per_wind_step=1,
        rng=jax.random.PRNGKey(0),
        integrator="euler"
    )

    print(wind_offsets) # [B, 2]
    assert jnp.allclose(wind_offsets, b_wind, atol=1e-2)

    # Check per boat
    assert jnp.abs(jnp.sum(wind_offsets, axis=1))[1] < 1e-6
    assert jnp.abs(jnp.sum(wind_offsets, axis=1))[3] < 0.15
    assert jnp.abs(jnp.sum(wind_offsets, axis=1))[0] > 1.8
    assert jnp.abs(jnp.sum(wind_offsets, axis=1))[2] > 1.8

    # Check x not affected
    assert jnp.abs(jnp.sum(wind_offsets, axis=0))[0] < 0.5

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(N):
            boat_pos = boat_positions[i]
            wind_offset = wind_offsets[i]
            circle = plt.Circle(boat_pos, 1, color='b', fill=False)
            ax.add_patch(circle)
            ax.arrow(boat_pos[0], boat_pos[1], wind_offset[0], wind_offset[1], fc='r',
                     ec='r')
        ax.set_xlim(-5, 15)
        ax.set_ylim(-5, 15)
        ax.set_aspect('equal')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Boat Wind Interaction')
        plt.grid(True)
        plt.show()