from monohull_dynamics.forces.polars.polar import PolarData, cl, cd0
import jax.numpy as jnp
from monohull_dynamics.forces.polars.polar import rho_water, water_re
import typing
from jax import lax
import jax.debug

class FoilData(typing.NamedTuple):
    length: jnp.ndarray
    chord: jnp.ndarray
    polar: PolarData

def aspect_ratio(foil_data: FoilData):
    return foil_data.length / foil_data.chord

def area(foil_data: FoilData):
    return foil_data.length * foil_data.chord

def foil_frame_resultant(foil_data: FoilData, foil_frame_flow: jnp.ndarray) -> jnp.ndarray:
    mag = jnp.linalg.norm(foil_frame_flow)
    return lax.cond(
        mag < 1e-6,
        lambda _: jnp.zeros(2),
        lambda _: _foil_frame_resultant_unsafe(foil_data, foil_frame_flow),
        None,
    )

def _foil_frame_resultant_unsafe(foil_data: FoilData, foil_frame_flow: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        foil_frame_flow: [x, y] velocity of flow at foil where foil is at origin and pointing right,
        so horizontal flow is [-1,0], positive foil_alpha is from below, so flow is [-1,1]
    Returns:
        Resultant force vector in foil reference frame (foil pointing left)
    """

    # arctan2 is anticlockwise from +ve x, alpha defined clockwise from -ve x axis in terms of flow
    alpha = jnp.arctan2(
        foil_frame_flow[1], -foil_frame_flow[0]
    )
    alpha_deg = jnp.rad2deg(alpha)
    re = water_re(jnp.linalg.norm(foil_frame_flow), foil_data.chord)
    _cl = cl(foil_data.polar, re, alpha_deg)
    _cd = cd0(foil_data.polar, re)
    cd_tot = _cd + _cl**2 / (jnp.pi * aspect_ratio(foil_data))
    lift_dir = jnp.array([jnp.sin(alpha), jnp.cos(alpha)])
    drag_dir = foil_frame_flow / jnp.linalg.norm(foil_frame_flow)
    lift = (
        lift_dir
        * 0.5
        * _cl
        * rho_water
        * foil_data.chord
        * jnp.linalg.norm(foil_frame_flow) ** 2
    )
    drag = (
        drag_dir
        * 0.5
        * cd_tot
        * rho_water
        * foil_data.chord
        * jnp.linalg.norm(foil_frame_flow) ** 2
    )
    # jax.debug.print("lift: {lift}", lift=lift)
    # jax.debug.print("drag: {drag}", drag=drag)
    # print(
    #     f"foil alpha: {alpha_deg}, fl: {np.linalg.norm(lift)}, fd: {np.linalg.norm(drag)}"
    # )

    return lift + drag
