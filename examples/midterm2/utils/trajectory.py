import numpy as np


class TargetTrajectory:
    def __init__(self, delta=0.5, tf=9.0, W=1.2, H=1.3):
        """Task 3: Mathematical Trajectory Generator."""
        self.delta = delta
        self.tf = tf
        self.W = W
        self.H = H

    def get_pose(self, t):
        # Apply min(t, tf) to ensure target stays at final position [cite: 163]
        t_c = np.minimum(t, self.tf)

        # a0 maps time to normalized range [-1, 1] [cite: 154]
        z = (2 * t_c / self.tf) - 1

        # a1 shapes the trajectory in the e1 direction [cite: 154]
        a1_z = (np.sin(np.pi / 2 * z) ** 2) ** (2 - self.delta) * np.tanh(10 * z)

        # a2 shapes the trajectory in the e2 direction [cite: 156]
        a2_val = (np.sin(np.pi / 2 * a1_z) ** 2) ** (
            0.1 * np.exp(-self.delta)
        ) * np.tanh(10 * a1_z)

        # Apply functions to define target x1 and x2 components [cite: 163]
        x1_star = self.W * a1_z
        x2_star = self.H * a2_val

        return np.array([x1_star, x2_star])
