import numpy as np
import time

class Particle2D:
    def __init__(
        self,
        m,
        I,
        x: tuple[float, float],
        xdot: tuple[float, float],
        theta: float,
        thetadot: float,
    ):
        # Theta anticlockwise from x axis
        self.m = m
        self.I = I
        self.x = np.array(x)
        self.xdot = np.array(xdot)
        self.theta = theta
        self.thetadot = thetadot

    def step(self, force: tuple[float, float], moment: float, dt):
        """
        Step dt with semi implicit Euler
        """
        if np.any(np.isnan(force)):
            raise ValueError("Force is nan")

        # a
        xdotdot = np.array(force) / self.m
        thetadotdot = moment / self.I

        # v
        self.xdot += xdotdot * dt
        self.thetadot += thetadotdot * dt

        # x
        self.x += self.xdot * dt
        self.theta += self.thetadot * dt

