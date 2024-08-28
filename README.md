**Note that this package is published under aGPLv3 - restrictions apply to commercial use.**

# Python Monohull Sail Dynamics
A package intended to provide per-element simulation of sailing monohulls. Implemented in pure JAX and aimed at 
reinforcement learning for sailboat control. 

I made this package as existing packages mainly account for optimal states
(E.G. assuming the main is trimmed appropriately upwind), so this includes reasonable approximations for more unusual states
like backing of the mainsail etc. The estimator currently supports variable foils, a single sail with finn geometry and variable area, and approximations for wave drag using the delft series. Simulations are limited to 2D, neglecting heel and pitch. Leeway is naturally produced through the force model.

## Installation

```bash
git clone git@github.com:hctomkins/monohull-dynamics.git
pip install -e monohull-dynamics
```

## Usage
### Demo GUI
```bash
python -m monohull_dynamics.demo.demo
```

### Module
The module entrypoint is through defining a statics object to resolve forces:
```python
import monohull_dynamics as md
boat_forces = md.forces.boat.BoatPhysics(
    centreboard_length=1.05,
    centreboard_chord=0.25,
    sail_area=6.3,
    hull_draft=0.25,
    rudder_length=1,
    rudder_chord=0.22,
    beam=1.42,
    lwl=3.58,
    length=3.66,
    sail_coe_dist=1,
)
integrator = md.dynamics.particle.Particle2D()

# TODO: Example for storing boat velocity (see demo for now)
while True:
    force, moment, _ = boat_forces.forces_and_moments(
        boat_velocity=# see demo,
        wind_velocity=(0, -4),
        boat_theta=integrator.theta,
        boat_theta_dot=integrator.thetadot,
        sail_angle=# see demo,
        rudder_angle=# see demo,
    )
    integrator.step(force, moment, dt=0.001)

```