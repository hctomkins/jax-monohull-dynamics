from forces.polars.polar import Polar
import numpy as np
from forces.polars.polar import rho_water, water_re


class Foil:
    def __init__(self, length: float, chord: float, polar_dir: str | None):
        self.polar = Polar(polar_dir)

        self.length = length
        self.chord = chord

    @property
    def aspect_ratio(self):
        return self.length / self.chord

    @property
    def area(self):
        return self.length * self.chord

    def foil_frame_resultant(self, foil_frame_flow: tuple[float, float]):
        """
        Args:
            foil_frame_flow: [x, y] velocity of flow at foil where foil is at origin and pointing right,
            so horizontal flow is [-1,0], positive foil_alpha is from below, so flow is [-1,1]
        Returns:
            Resultant force vector in foil reference frame (foil pointing left)
        """
        if np.linalg.norm(foil_frame_flow) < 1e-3:
            return np.array([0, 0])

        alpha = np.arctan2(
            foil_frame_flow[1], -foil_frame_flow[0]
        )  # clockwise from -ve x axis
        alpha_deg = np.rad2deg(alpha)
        re = water_re(np.linalg.norm(foil_frame_flow), self.chord)
        cl = self.polar.cl(re, alpha_deg)
        cd = self.polar.cd0(re)
        cd_tot = cd + cl**2 / (np.pi * self.aspect_ratio)
        lift_dir = np.array([np.sin(alpha), np.cos(alpha)])
        drag_dir = np.array(foil_frame_flow) / np.linalg.norm(foil_frame_flow)
        lift = (
            lift_dir
            * 0.5
            * cl
            * rho_water
            * self.chord
            * np.linalg.norm(foil_frame_flow) ** 2
        )
        drag = (
            drag_dir
            * 0.5
            * cd_tot
            * rho_water
            * self.chord
            * np.linalg.norm(foil_frame_flow) ** 2
        )
        # print(
        #     f"foil alpha: {alpha_deg}, fl: {np.linalg.norm(lift)}, fd: {np.linalg.norm(drag)}"
        # )

        return lift + drag
