from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from forces.polars.polar import rho_water
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

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


def a0(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a0"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a0"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


def a1(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a1"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a1"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


def a2(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a2"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a2"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


def a3(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a3"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a3"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


def a4(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a4"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a4"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


def a5(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a5"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a5"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


def a6(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a6"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a6"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


def a7(Fn):
    return np.interp(Fn, coeffs["Fn"], coeffs["a7"])
    # interpolator = interp1d(coeffs["Fn"], coeffs["a7"], kind="linear", fill_value="extrapolate")
    # return interpolator(Fn)


class HullDragEstimator:
    def __init__(self, hull_draft: float, beam: float, lwl: float):
        self.hull_draft = hull_draft
        self.beam = beam
        self.lwl = lwl
        self.prismatic_coefficient = 0.5
        self.midship_section_coefficient = 0.7
        # self.wave_drag_estimator = self.setup_fit()

    def displacement_wave_drag(self, fn: float) -> float:
        dimensionless_drag = (
            a0(fn)
            + (self.volume_of_displacement ** (1 / 3) / self.lwl)
            * (
                a1(fn) * self.lcb / self.lcf
                + a2(fn) * self.prismatic_coefficient
                + a3(fn) * self.volume_of_displacement ** (2 / 3) / self.awp
                + a4(fn) * self.bwl / self.lwl
            )
            + (self.volume_of_displacement ** (1 / 3) / self.lwl)
            * (
                a5(fn) * self.lcb / self.lcf
                + a6(fn) * self.bwl / self.hull_draft
                + a7(fn) * self.midship_section_coefficient
            )
        )
        return max(
            self.volume_of_displacement * rho_water * 9.81 * dimensionless_drag, 0
        )

    def wave_drag(self, velocity: tuple[float, float]):
        speed = np.linalg.norm(velocity)
        fn = speed / np.sqrt(9.81 * self.lwl)
        drag_magnitude = self.displacement_wave_drag(fn)
        drag_direction = -velocity / (speed + 1e-3)
        return drag_magnitude * drag_direction

    def viscous_drag(self, velocity: tuple[float, float]):
        speed = np.linalg.norm(velocity)
        drag_magnitude = (
            1
            / 2
            * rho_water
            * speed**2
            * self.awp
            * self.skin_friction_coefficient(speed)
        )
        drag_direction = -velocity / (speed + 1e-3)
        return drag_magnitude * drag_direction

    def skin_friction_coefficient(self, u: float):
        return 0.075 / (np.log10(588000 * self.lwl * u) - 2) ** 2

    # def setup_fit(self, plot=True):
    #     fns = np.linspace(0.15, 0.75, 1000)
    #     drag = np.array([self.displacement_wave_drag(fn) for fn in fns])
    #     max_drag = np.max(drag)
    #
    #     def f(fn, b, s):
    #         a = -0.7
    #         return max_drag * s * fn * (1/(b**2 * (fn + a)**2 + 1)**2) # + (b * (fn + a))**2 / 20)
    #
    #     popt, _ = curve_fit(f=f, xdata=fns, ydata=drag, p0=[-3, 1.5])
    #
    #     if plot:
    #         plt.figure()
    #         plt.plot(fns, drag, label="Data")
    #         plt.plot(fns, [f(fn, b=popt[0], s=popt[1]) for fn in fns], label="Fit")
    #         plt.legend()
    #
    #     return partial(f, b=popt[0], s=popt[1])

    @property
    def lcb(self):
        return 0.5 * self.lwl

    @property
    def lcf(self):
        return 0.5 * self.lwl

    @property
    def bwl(self):
        return self.beam * 0.95

    @property
    def awp(self):
        return self.lwl * self.bwl

    @property
    def volume_of_displacement(self):
        return self.awp * self.hull_draft * 0.6


if __name__ == "__main__":
    awd = HullDragEstimator(lwl=3.58, beam=1.4, hull_draft=0.2)
    speeds = np.linspace(0, 15, 100)
    fn = speeds / np.sqrt(9.81 * awd.lwl)
    wave_drag = [awd.wave_drag(speed) for speed in speeds]
    viscous_drag = [awd.viscous_drag(speed) for speed in speeds]
    ms_to_knots = 1.94384

    x_axis = speeds * ms_to_knots
    # x_axis = fn

    plt.figure()
    plt.plot(x_axis, wave_drag, label="Wave drag")
    plt.plot(x_axis, viscous_drag, label="Viscous drag")
    plt.plot(x_axis, np.array(wave_drag) + np.array(viscous_drag), label="Total drag")
    plt.xlabel("Speed [knots]")
    plt.legend()
    plt.show()
