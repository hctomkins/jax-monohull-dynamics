import glob
import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def knts_to_ms(knts):
    return knts * 0.514444


def water_re(v, l):
    nu = 1.26e-6
    return v * l / nu


def air_re(v, l):
    nu = 1.45e-5
    return v * l / nu


rho_air = 1.27
rho_water = 1000


class Polar:
    def __init__(self, dir: str | None):
        if dir is None:
            self.alpha_min = 0
            self.alpha_max = 0
        else:
            dfs = []
            for f in glob.glob(os.path.join(dir, "*.csv")):
                stem = Path(f).stem
                (
                    _xf,
                    _naca,
                    _il,
                    re,
                    _ncsv,
                ) = stem.split("-")
                data = pd.read_csv(f, header=9)
                data["re"] = float(re)
                dfs.append(data)
            self.data = pd.concat(dfs, axis=0)
            self.alpha_min = self.data["Alpha"].min()
            self.alpha_max = self.data["Alpha"].max()

    def cd0(self, re: int):
        return self.cd(re, 0)

    def cd(self, re: int, alpha_deg):
        if alpha_deg <= self.alpha_min or alpha_deg >= self.alpha_max:
            # TODO: should this terminate at 2 or 1.24? Flat plate in 2d or 3d?
            return 1 - np.cos(np.deg2rad(alpha_deg) * 2)

        re = self.nearest_re(re)
        at_re = self.data[self.data["re"] == re]
        return np.interp(alpha_deg, at_re["Alpha"], at_re["Cd"])

    def cl(self, re: int, alpha_deg):
        if alpha_deg <= self.alpha_min or alpha_deg >= self.alpha_max:
            return np.sin(np.deg2rad(alpha_deg) * 2)

        re = self.nearest_re(re)
        at_re = self.data[self.data["re"] == re]
        return np.interp(alpha_deg, at_re["Alpha"], at_re["Cl"])

    def nearest_re(self, re: int):
        all_re = self.data["re"].unique()
        return all_re[np.argmin(np.abs(all_re - re))]


if __name__ == "__main__":
    p = Polar("n12")
    re = water_re(knts_to_ms(4), 0.2)
    print(re)
    print(p.cd0(re))
    print(p.cl(re, 10))

    alphas = np.linspace(-90, 90, 100)
    plt.plot(alphas, [p.cd(re, a) for a in alphas])
    plt.show()
