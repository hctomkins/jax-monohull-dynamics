**Note that this package is published under aGPLv3 - restrictions apply to commercial use.**

# JAX Monohull Sailing Simulation
A package intended to provide per-element simulation of sailing monohulls. Implemented in pure JAX and aimed at 
reinforcement learning for sailboat control. 

I made this package as existing packages mainly account for optimal states
(E.G. assuming the main is trimmed appropriately upwind), so this includes reasonable approximations for more unusual states
like backing of the mainsail etc. The estimator currently supports variable foils, a single sail with finn geometry and variable area, and approximations for wave drag using the delft series. Simulations are limited to 2D, neglecting heel and pitch. Leeway is naturally produced through the force model.

## Installation

```bash
git@github.com:hctomkins/jax-monohull-dynamics.git
pip install -e jax-monohull-dynamics
```

## Usage
### Demo GUI
```bash
python -m monohull_dynamics.demo.demo
```

### Module
The module entrypoint is through defining a statics object to resolve forces:
```python
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
particle_state = boat_state.particle_state
force_model = init_boat(
        centreboard_length=1.05,
        centreboard_chord=0.25,
        sail_area=6.3,
        hull_draft=0.25,
        rudder_length=1.0,
        rudder_chord=0.22,
        beam=1.42,
        lwl=3.58,
        length=3.66,
        sail_coe_dist=1.0,
    )

# TODO: Example for storing boat velocity (see demo for now)
while True:
    f, m, dd = forces_and_moments(
        boat_data=force_model,
        boat_velocity=particle_state.xdot,
        wind_velocity=wind_velocity,
        boat_theta=particle_state.theta,
        boat_theta_dot=particle_state.thetadot,
        sail_angle=boat_state.sail_angle,
        rudder_angle=boat_state.rudder_angle,
    )
    new_boat_state = boat_state._replace(particle_state=integrate(particle_state, f, m, inner_dt), debug_data=dd)
```