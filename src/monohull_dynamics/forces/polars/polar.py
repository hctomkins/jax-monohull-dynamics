import glob
import pandas as pd
import os
from pathlib import Path
import jax.numpy as jnp
import typing

POLAR_ROOT = Path(__file__).parent


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


class PolarData(typing.NamedTuple):
    all_re: jnp.ndarray  # [n_re]
    alphas_by_re: jnp.ndarray  # [n_re, n_alpha]
    cd_by_re: jnp.ndarray  # [n_re, n_alpha]
    cl_by_re: jnp.ndarray  # [n_re, n_alpha]


def init_polar(dir: str | None):
    assert dir is not None
    cd_arrays = []
    cl_arrays = []
    alphas = []
    all_re: list[int] = []
    # Return interpolated array of:
    # alphas [n_re, n_alpha]
    # cd [n_re, n_alpha]
    # cl [n_re, n_alpha]
    # all_re [n_re]
    # Where we pad the end of each array with duplicate data

    for f in glob.glob(os.path.join(POLAR_ROOT, dir, "*.csv")):
        stem = Path(f).stem
        (
            _xf,
            _naca,
            _il,
            re,
            _ncsv,
        ) = stem.split("-")
        re = int(re)
        all_re.append(re)
        data = pd.read_csv(f, header=9)
        alphas.append(jnp.array(data["Alpha"]).astype(jnp.float32))
        cd_arrays.append(jnp.array(data["Cd"]).astype(jnp.float32))
        cl_arrays.append(jnp.array(data["Cl"]).astype(jnp.float32))

    max_alphas = max(len(a) for a in alphas)
    for i, re in enumerate(all_re):
        padding = max_alphas - len(alphas[i])
        alphas[i] = jnp.pad(alphas[i], (0, padding), mode="edge")
        cd_arrays[i] = jnp.pad(cd_arrays[i], (0, padding), mode="edge")
        cl_arrays[i] = jnp.pad(cl_arrays[i], (0, padding), mode="edge")

    return PolarData(
        all_re=jnp.array(all_re),
        alphas_by_re=jnp.stack(alphas),
        cd_by_re=jnp.stack(cd_arrays),
        cl_by_re=jnp.stack(cl_arrays),
    )


def cd0(polar_data: PolarData, re):
    return cd(polar_data, re, 0)


def cd(polar_data: PolarData, re, alpha_deg):
    re = re_index(polar_data, re)
    left_right_val = (1 - jnp.cos(jnp.deg2rad(alpha_deg) * 2)) * 0.6
    return jnp.interp(
        alpha_deg,
        polar_data.alphas_by_re[re],
        polar_data.cd_by_re[re],
        left=left_right_val,
        right=left_right_val,
    )


def cl(polar_data: PolarData, re, alpha_deg):
    re = re_index(polar_data, re)
    left_right_val = jnp.sin(jnp.deg2rad(alpha_deg) * 2)

    return jnp.interp(
        alpha_deg,
        polar_data.alphas_by_re[re],
        polar_data.cl_by_re[re],
        left=left_right_val,
        right=left_right_val,
    )


def re_index(polar_data: PolarData, re: int):
    return jnp.argmin(jnp.abs(polar_data.all_re - re))


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    polar_data = init_polar("n12")
    re = water_re(knts_to_ms(4), 0.2)

    print(cd0(polar_data, re))
    print(cl(polar_data, re, 10))

    alphas = jnp.linspace(-90, 90, 100)
    for re in [
        water_re(knts_to_ms(4), 0.2),
        water_re(knts_to_ms(40), 0.2),
        water_re(knts_to_ms(0.4), 0.2),
    ]:
        plt.plot(alphas, [cd(polar_data, re, a) for a in alphas], label=f"Re={re}")

    plt.legend()
    plt.show()
