import typing

import jax.numpy as jnp

from monohull_dynamics.forces.polars.polar import fast_interp, rho_air

REVERSAL_FACTOR = -0.3
FINN_CL_CD = {
    0: (0, 0.097),
    5: (0.386, 0.097),
    7.5: (0.516, 0.108),
    10: (0.649, 0.13),
    15: (0.78, 0.154),
    17.5: (1.043, 0.195),
    20: (1.133, 0.235),
    25: (1.21, 0.278),
    27.5: (1.26, 0.33),
    30: (1.305, 0.386),
    32.5: (1.277, 0.454),
    35: (1.234, 0.519),
    37: (1.2, 0.627),
    40: (1.185, 0.72),
    50: (0.95, 0.86),
    60: (0.75, 1),
    70: (0.53, 1.17),
    80: (0.32, 1.2),
    90: (0, 1.24),
}

# Flow reversal regime up to 180 - hack for now - mirror cd, and mirror cl with a reversal factor
FINN_CL_CD = {**FINN_CL_CD, **{180 - k: (v[0] * REVERSAL_FACTOR, v[1]) for k, v in list(FINN_CL_CD.items())[::-1] if k != 90}}


FINN_ALPHAS = list(FINN_CL_CD.keys())
FINN_CL = [x[0] for x in FINN_CL_CD.values()]
FINN_CD = [x[1] for x in FINN_CL_CD.values()]


class SailData(typing.NamedTuple):
    area: jnp.ndarray
    alphas: jnp.ndarray
    cl: jnp.ndarray
    cd: jnp.ndarray


def init_sail_data(area: jnp.ndarray):
    return SailData(
        area=area,
        alphas=jnp.array(FINN_ALPHAS),
        cl=jnp.array(FINN_CL),
        cd=jnp.array(FINN_CD),
    )


def sail_frame_resultant(sail_data: SailData, flow: jnp.array):
    """
    Args:
        flow: [x, y] velocity of flow at sail where sail is at origin and pointing right,
        so horizontal flow is [-1,0], positive sail alpha is from below, so flow is [-1,1]. Shape [2]
    Returns:
        Resultant force vector in foil reference frame (foil pointing left)
    """
    alpha = jnp.arctan2(flow[1], -flow[0])
    alpha_sign = jnp.sign(alpha)
    alpha_abs_deg = jnp.abs(jnp.rad2deg(alpha))
    cl = fast_interp(alpha_abs_deg, sail_data.alphas, sail_data.cl, left=sail_data.cl[0], right=sail_data.cl[-1])
    cd = fast_interp(alpha_abs_deg, sail_data.alphas, sail_data.cd, left=sail_data.cd[0], right=sail_data.cd[-1])
    lift_dir = jnp.stack([jnp.sin(alpha), jnp.cos(alpha)]) * alpha_sign
    drag_dir = jnp.array(flow) / jnp.linalg.norm(flow)
    lift = lift_dir * 0.5 * cl * rho_air * sail_data.area * jnp.linalg.norm(flow) ** 2
    drag = drag_dir * 0.5 * cd * rho_air * sail_data.area * jnp.linalg.norm(flow) ** 2
    # print(
    #     f"sail alpha: {np.rad2deg(alpha)} degrees, fl: {np.linalg.norm(lift)}, fd: {np.linalg.norm(drag)}"
    # )
    return lift + drag
