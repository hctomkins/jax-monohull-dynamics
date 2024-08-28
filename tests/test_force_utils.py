import pytest
from monohull_dynamics.forces.force_utils import (
    moments_about,
    rotate_vector_about,
    rotate_vector,
    coe_offset,
    foil_force_on_boat, flow_at_foil
)
import jax.numpy as jnp
from jax import jit, vmap


@pytest.mark.parametrize(
    "force, at, about, min, max",
    [
        (jnp.array([1, 0]), jnp.array([1, 1]), jnp.array([0, 0]), -10, 0),
        (jnp.array([0, 1]), jnp.array([-1, 0]), jnp.array([0, 0]), -10, 0),
        (jnp.array([-1, -1]), jnp.array([-1, 0]), jnp.array([0, 0]), 0, 10),
        (jnp.array([0, -1]), jnp.array([-1, 0]), jnp.array([0, 0]), 0, 10),
    ],
)
def test_moments_about(force, at, about, min, max):
    assert min < moments_about(force=force, at=at, about=about) < max


def test_rotate_vector():
    assert jnp.allclose(
        rotate_vector(jnp.array([1, 0]), jnp.pi / 2),
        jnp.array([0, 1]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert jnp.allclose(
        rotate_vector(jnp.array([-1, 0]), jnp.pi / 2),
        jnp.array([0, -1]),
        atol=1e-5,
        rtol=1e-5,
    )

    rotate_many = jit(vmap(rotate_vector, in_axes=(0, None)))
    assert jnp.allclose(
        rotate_many(jnp.array([[1, 0], [-1, 0]]), jnp.pi / 2),
        jnp.array([[0, 1], [0, -1]]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_rotate_vector_about():
    assert jnp.allclose(
        rotate_vector_about(jnp.array([1, 0]), jnp.pi / 2, jnp.array([0, 2])),
        jnp.array([2, 3]),
        atol=1e-5,
        rtol=1e-5,
    )

    rotate_many = jit(vmap(rotate_vector_about, in_axes=(0, None, 0)))

    assert jnp.allclose(
        rotate_many(
            jnp.array([[1, 0], [-1, 0]]),
            jnp.pi / 2,
            jnp.array([[0, 2], [0, 2]]),
        ),
        jnp.array([[2, 3], [2, 1]]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_coe_offset():
    foil_offset = jnp.array([-5, 0])
    coe = jnp.sqrt(2)
    foil_theta = jnp.pi / 4
    assert jnp.allclose(
        coe_offset(foil_offset, foil_theta, coe),
        jnp.array([-6, -1]),
        atol=1e-5,
        rtol=1e-5,
    )
    coe_many = jit(vmap(coe_offset, in_axes=(0, 0, None)))
    assert jnp.allclose(
        coe_many(
            jnp.array([[-5, 0], [-5, 0]]),
            jnp.array([jnp.pi / 4, jnp.pi / 2]),
            jnp.sqrt(2),
        ),
        jnp.array([[-6, -1], [-5, -jnp.sqrt(2)]]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_foil_force_on_boat():
    foil_force = jnp.array([0, 1])
    foil_offset = jnp.array([-1, 0])
    foil_theta = jnp.pi / 2
    boat_theta = jnp.pi / 2
    foil_coe = 1.0
    f, m = foil_force_on_boat(foil_force, foil_offset, foil_theta, boat_theta, foil_coe)
    print(m)
    assert jnp.allclose(
        f,
        jnp.array([0, -1]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert jnp.allclose(m, -1, atol=1e-5, rtol=1e-5)
    foil_force_many = jit(vmap(foil_force_on_boat, in_axes=(0, None, None, None, None)))
    f, m = foil_force_many(
        jnp.array([[0, 1], [0, -1]]), foil_offset, foil_theta, boat_theta, foil_coe
    )
    assert jnp.allclose(
        f,
        jnp.array([[0, -1], [0, 1]]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert jnp.allclose(m, jnp.array([-1, 1]), atol=1e-5, rtol=1e-5)

def test_flow_at_foil():

    # Boat moving sideways, flow up into foil
    flow = flow_at_foil(
        flow_velocity=jnp.array([0, 0]),
        boat_velocity=jnp.array([5, 0]),
        foil_offset=jnp.array([-1, 0]),
        foil_theta=0,
        foil_coe=0, #jnp.sqrt(2),
        boat_theta=jnp.pi / 2,
        boat_theta_dot=0,
    )
    assert jnp.allclose(flow, jnp.array([0, 5]), atol=1e-5, rtol=1e-5)

    # foil 45 degrees anticlockwise, flow now up and to the right
    flow = flow_at_foil(
        flow_velocity=jnp.array([0, 0]),
        boat_velocity=jnp.array([jnp.sqrt(2), 0]),
        foil_offset=jnp.array([-1, 0]),
        foil_theta=jnp.pi/4,
        foil_coe=0, #jnp.sqrt(2),
        boat_theta=jnp.pi / 2,
        boat_theta_dot=0,
    )
    assert jnp.allclose(flow, jnp.array([1, 1]), atol=1e-5, rtol=1e-5)

    # boat rotating anticlockwise
    flow = flow_at_foil(
        flow_velocity=jnp.array([0, 0]),
        boat_velocity=jnp.array([jnp.sqrt(2), 0]),
        foil_offset=jnp.array([-1, 0]),
        foil_theta=jnp.pi/4,
        foil_coe=0, #jnp.sqrt(2),
        boat_theta=jnp.pi / 2,
        boat_theta_dot=jnp.sqrt(2),
    )
    assert jnp.allclose(flow, jnp.array([2, 2]), atol=1e-5, rtol=1e-5)

    #with coe
    flow = flow_at_foil(
        flow_velocity=jnp.array([0, 0]),
        boat_velocity=jnp.array([jnp.sqrt(2), 0]),
        foil_offset=jnp.array([-1, 0]),
        foil_theta=jnp.pi/4,
        foil_coe=jnp.sqrt(2),
        boat_theta=jnp.pi / 2,
        boat_theta_dot=jnp.sqrt(2),
    )
    assert jnp.allclose(flow, jnp.array([2, 4]), atol=1e-5, rtol=1e-5)

    flow_many = jit(vmap(flow_at_foil, in_axes=(None, 0, None, None, None, None, None)))
    flows = flow_many(
        jnp.array([0, 0]),
        jnp.array([[jnp.sqrt(2), 0],[2*jnp.sqrt(2), 0]]),
        jnp.array([-1, 0]),
        jnp.pi/4,
        jnp.sqrt(2),
        jnp.pi / 2,
        jnp.sqrt(2),
    )
    assert jnp.allclose(flows, jnp.array([[2, 4],[3,5]]), atol=1e-5, rtol=1e-5)

