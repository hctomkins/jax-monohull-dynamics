import jax.numpy as jnp
import typing
from monohull_dynamics.forces.polars.polar import rho_water


# import matplotlib
# from scipy.optimize import curve_fit
# from functools import partial
# from scipy.interpolate import interp1d
# matplotlib.use("WebAgg")
# import matplotlib.pyplot as plt

coeffs = {
    "Fn": [
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    ],
    "a0": [
        -0.0005,
        -0.0003,
        -0.0002,
        -0.0009,
        -0.0026,
        -0.0064,
        -0.0218,
        -0.0388,
        -0.0347,
        -0.0361,
        0.0008,
        0.0108,
        0.1023,
    ],
    "a1": [
        0.0023,
        0.0059,
        -0.0156,
        0.0016,
        -0.0567,
        -0.4034,
        -0.5261,
        -0.5986,
        -0.4764,
        -0.0037,
        0.3728,
        -0.1238,
        0.7726,
    ],
    "a2": [
        -0.0086,
        -0.0064,
        0.0031,
        0.0337,
        0.0446,
        -0.1250,
        -0.2945,
        -0.3038,
        -0.2361,
        -0.2960,
        -0.3667,
        -0.2026,
        0.5040,
    ],
    "a3": [
        0.0015,
        0.0070,
        -0.0021,
        -0.0285,
        -0.1091,
        0.0273,
        0.2485,
        0.6033,
        0.8726,
        0.9661,
        1.3957,
        1.1282,
        1.7867,
    ],
    "a4": [
        0.0061,
        0.0014,
        -0.0070,
        -0.0367,
        -0.0707,
        -0.1341,
        -0.2428,
        -0.0430,
        0.4219,
        0.6123,
        1.0343,
        1.1836,
        2.1934,
    ],
    "a5": [
        0.0010,
        0.0013,
        0.0148,
        0.0218,
        0.0914,
        0.3578,
        0.6293,
        0.8332,
        0.8990,
        0.7534,
        0.3230,
        0.4973,
        -1.5479,
    ],
    "a6": [
        0.0001,
        0.0005,
        0.0010,
        0.0053,
        0.0021,
        0.0045,
        0.0081,
        0.0106,
        0.0106,
        0.0100,
        0.0072,
        0.0038,
        -0.0115,
    ],
    "a7": [
        0.0052,
        -0.0020,
        -0.0043,
        -0.0172,
        -0.0078,
        0.1115,
        0.2086,
        0.1336,
        -0.2272,
        -0.3352,
        -0.4632,
        -0.4477,
        -0.0977,
    ],
}


def get_hull_coeffs():
    return {k: jnp.array(v) for k, v in coeffs.items()}


def a0(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a0"])


def a1(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a1"])


def a2(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a2"])


def a3(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a3"])


def a4(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a4"])


def a5(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a5"])


def a6(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a6"])


def a7(coeffs: dict[str, jnp.ndarray], Fn: jnp.ndarray):
    return jnp.interp(Fn, coeffs["Fn"], coeffs["a7"])


class HullData(typing.NamedTuple):
    hull_draft: jnp.ndarray
    beam: jnp.ndarray
    lwl: jnp.ndarray
    prismatic_coefficient: jnp.ndarray = 0.5
    midship_section_coefficient: jnp.ndarray = 0.7


def init_hull(hull_draft: jnp.ndarray, beam: jnp.ndarray, lwl: jnp.ndarray):
    return HullData(
        hull_draft=jnp.array(hull_draft),
        beam=jnp.array(beam),
        lwl=jnp.array(lwl),
        prismatic_coefficient=jnp.array(0.5),
        midship_section_coefficient=jnp.array(0.7),
    )


def displacement_wave_drag(
    hull_data: HullData, coeffs: dict[str, jnp.ndarray], fn: jnp.ndarray
) -> jnp.ndarray:
    _lcb = lcb(hull_data.lwl)
    _lcf = lcf(hull_data.lwl)
    _bwl = bwl(hull_data.beam)
    _awp = awp(hull_data.lwl, _bwl)
    _volume_of_displacement = volume_of_displacement(_awp, hull_data.hull_draft)

    dimensionless_drag = (
        a0(coeffs, fn)
        + (_volume_of_displacement ** (1 / 3) / hull_data.lwl)
        * (
            a1(coeffs, fn) * _lcb / _lcf
            + a2(coeffs, fn) * hull_data.prismatic_coefficient
            + a3(coeffs, fn) * _volume_of_displacement ** (2 / 3) / _awp
            + a4(coeffs, fn) * _bwl / hull_data.lwl
        )
        + (_volume_of_displacement ** (1 / 3) / hull_data.lwl)
        * (
            a5(coeffs, fn) * _lcb / _lcf
            + a6(coeffs, fn) * _bwl / hull_data.hull_draft
            + a7(coeffs, fn) * hull_data.midship_section_coefficient
        )
    )
    return jnp.clip(
        _volume_of_displacement * rho_water * 9.81 * dimensionless_drag, min=0, max=None
    )


def wave_drag(
    hull_data: HullData, coeffs: dict[str, jnp.ndarray], velocity: jnp.ndarray
) -> jnp.ndarray:
    speed = jnp.linalg.norm(velocity)
    fn = speed / jnp.sqrt(9.81 * hull_data.lwl)
    drag_magnitude = displacement_wave_drag(hull_data, coeffs, fn)
    drag_direction = -velocity / (speed + 1e-3)
    return drag_magnitude * drag_direction


def viscous_drag(hull_data: HullData, velocity: jnp.ndarray) -> jnp.ndarray:
    _bwl = bwl(hull_data.beam)
    _awp = awp(hull_data.lwl, _bwl)
    speed = jnp.linalg.norm(velocity)
    drag_magnitude = (
        1
        / 2
        * rho_water
        * speed**2
        * _awp
        * skin_friction_coefficient(hull_data.lwl, speed)
    )
    drag_direction = -velocity / (speed + 1e-3)
    return drag_magnitude * drag_direction


def skin_friction_coefficient(lwl: jnp.ndarray, u: jnp.ndarray):
    return 0.075 / (jnp.log10(588000 * lwl * u) - 2) ** 2


def lcb(lwl: jnp.ndarray):
    return 0.5 * lwl


def lcf(lwl: jnp.ndarray):
    return 0.5 * lwl


def bwl(
    beam: jnp.ndarray,
):
    return beam * 0.95


def awp(lwl: jnp.ndarray, bwl: jnp.ndarray):
    return lwl * bwl


def volume_of_displacement(awp: jnp.ndarray, hull_draft: jnp.ndarray):
    return awp * hull_draft * 0.6


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from jax import jit, vmap

    speeds = jnp.array([[0, s] for s in jnp.linspace(0, 15, 100)])
    hull_data = init_hull(hull_draft=0.2, beam=1.4, lwl=3.58)
    coeffs = get_hull_coeffs()

    wd_jit = jit(vmap(wave_drag, in_axes=(None, None, 0)))
    vd_jit = jit(vmap(viscous_drag, in_axes=(None, 0)))

    wave_drags = wd_jit(hull_data, coeffs, speeds)
    viscous_drags = vd_jit(hull_data, speeds)
    ms_to_knots = 1.94384

    x_axis = speeds[:, 1] * ms_to_knots

    plt.figure()
    plt.plot(x_axis, -wave_drags[:, 1], label="Wave drag")
    plt.plot(x_axis, -viscous_drags[:, 1], label="Viscous drag")
    plt.plot(x_axis, -(wave_drags + viscous_drags)[:, 1], label="Total drag")
    plt.xlabel("Speed [knots]")
    plt.legend()
    plt.show()
