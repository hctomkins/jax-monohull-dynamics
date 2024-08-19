from forces.polars.polar import air_re, rho_air
import numpy as np
import pandas as pd

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
FINN_ALPHAS = list(FINN_CL_CD.keys())
FINN_CL = [x[0] for x in FINN_CL_CD.values()]
FINN_CD = [x[1] for x in FINN_CL_CD.values()]


class MainSail:
    def __init__(self, area: float):
        self.area = area

    def sail_frame_resultant(self, flow: tuple[float, float]):
        """
        Args:
            flow: [x, y] velocity of flow at sail where sail is at origin and pointing right,
            so horizontal flow is [-1,0], positive sail alpha is from below, so flow is [-1,1]
        Returns:
            Resultant force vector in foil reference frame (foil pointing left)
        """
        alpha = np.arctan2(flow[1], -flow[0])
        alpha_sign = np.sign(alpha)
        alpha_abs_deg = np.abs(np.rad2deg(alpha))
        cl = np.interp(alpha_abs_deg, FINN_ALPHAS, FINN_CL)
        cd = np.interp(alpha_abs_deg, FINN_ALPHAS, FINN_CD)
        lift_dir = np.array([np.sin(alpha), np.cos(alpha)]) * alpha_sign
        drag_dir = np.array(flow) / np.linalg.norm(flow)
        lift = lift_dir * 0.5 * cl * rho_air * self.area * np.linalg.norm(flow) ** 2
        drag = drag_dir * 0.5 * cd * rho_air * self.area * np.linalg.norm(flow) ** 2
        # print(
        #     f"sail alpha: {np.rad2deg(alpha)} degrees, fl: {np.linalg.norm(lift)}, fd: {np.linalg.norm(drag)}"
        # )
        return lift + drag
