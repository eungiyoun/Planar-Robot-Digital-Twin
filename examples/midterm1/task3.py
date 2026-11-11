import sys
from pathlib import Path
from tqdm import tqdm

src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt

from rse.system import Robot
from rse.environment import Environment
from rse.integrator import PositionVerletIntegrator
from rse.simulator import Simulator
from rse.action import ActionProtocol


class Task3Action(ActionProtocol):
    """Piecewise control sequence mimicking Figure 5, WITH SATURATION CLIPPING."""

    def __init__(self, track_width: float, u_min: float = -1.0, u_max: float = 1.0) -> None:
        self.w = track_width
        self.u_min = u_min
        self.u_max = u_max

    def apply(self, robot: Robot, current_time: float) -> None:
        t = current_time

        # Left Wheel Command (Solid Grey Line)
        if 0.0 <= t < 3.0:
            ul = 1.5 - 0.5 * t
        elif 3.0 <= t < 5.0:
            ul = 0.0
        elif 5.0 <= t < 6.0:
            ul = -0.5 - 1.0 * (t - 5.0)
        else:
            ul = 0.0

        # Right Wheel Command (Dashed Black Line)
        if 0.0 <= t < 1.0:
            ur = 1.5 * t
        elif 1.0 <= t < 2.0:
            ur = 1.5 - 2.0 * (t - 1.0)
        elif 2.0 <= t < 5.0:
            ur = 0.0
        elif 5.0 <= t < 6.0:
            ur = 0.5 + 1.0 * (t - 5.0)
        else:
            ur = 0.0

        ul = np.clip(ul, self.u_min, self.u_max)
        ur = np.clip(ur, self.u_min, self.u_max)

        force_x = ul + ur
        torque = (ur - ul) * (self.w / 2.0)
        robot.apply_force_body(np.array([force_x, 0.0]))
        robot.apply_torque(torque)


def main() -> None:
    print("Running Task 3 Simulation...")

    # --- System setup ---
    mass = 2.0
    J = 0.05
    robot = Robot(mass=mass, moment_of_inertia=J, radius=0.2, track_width=0.15)
    robot.position = np.array([-1.2, -1.3])

    environment = Environment()
    integrator = PositionVerletIntegrator()
    action = Task3Action(track_width=robot.track_width)

    dt = 0.001
    simulator = Simulator(robot, environment, integrator, dt)
    time_horizon = 10.0
    
    num_steps = int(time_horizon / dt)

    # --- Pre-allocate data arrays ---
    pos_x     = np.zeros(num_steps)
    pos_y     = np.zeros(num_steps)
    heading_x = np.zeros(num_steps)
    heading_y = np.zeros(num_steps)
    l_vals    = np.zeros(num_steps)
    eta_norm  = np.zeros(num_steps)
    energy    = np.zeros(num_steps)
    times     = np.zeros(num_steps)

    # --- Run simulation ---
    for i in tqdm(range(num_steps), desc="Simulating"):
        pos_x[i] = robot.position[0]
        pos_y[i] = robot.position[1]
        h = robot.orientation[:, 0]
        heading_x[i] = float(h[0])
        heading_y[i] = float(h[1])

        eta = robot.linear_momentum
        l   = robot.angular_momentum
        eta_norm[i] = float(np.linalg.norm(eta))
        l_vals[i]   = float(l)
        energy[i]   = float((eta[0]**2 + eta[1]**2) / (2 * mass) + l**2 / (2 * J))
        times[i]    = simulator.time_current

        simulator.step(action)

    print("Simulation done. Plotting...")

    # --- Build static figure (matches task3_animation.py panel layout) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")

    # ============================================================
    # PANEL 1: Robot Trajectory
    # ============================================================
    verts = [(-1.5, -2), (1.5, -2), (1.5, -1), (0.9, -1), (0.9, 1),
             (1.5, 1), (1.5, 2), (-1.5, 2), (-1.5, 1), (-0.9, 1),
             (-0.9, -1), (-1.5, -1), (-1.5, -2)]
    ax1.plot(*zip(*verts), "k-", linewidth=3)

    ax1.fill([-0.75, 1.5, 1.5, -1.5], [1, 1, 2, 2],
             color="lightgray", alpha=0.5, label=r"$\mu_f=1$")
    ax1.fill([-0.75, 0.9, 0.9, 0.75], [1, 1, -1, -1],
             color="lightgray", alpha=0.5)
    ax1.fill([0.75, 1.5, 1.5], [-1, -1, -2],
             color="lightgray", alpha=0.5)

    # Full trajectory path
    ax1.plot(pos_x, pos_y, "k--", linewidth=1.5)

    # Robot at final position: circle + centre dot + heading arrow
    i_final = num_steps - 1
    robot_circle = plt.Circle(
        (pos_x[i_final], pos_y[i_final]), 0.2,
        fill=True, facecolor="silver", edgecolor="black", linewidth=2, zorder=3
    )
    ax1.add_patch(robot_circle)
    ax1.plot([pos_x[i_final]], [pos_y[i_final]], "ko", markersize=4, zorder=5)
    ax1.quiver(
        pos_x[i_final], pos_y[i_final],
        heading_x[i_final] * 0.40, heading_y[i_final] * 0.40,
        color="black", scale=1, scale_units="xy", angles="xy",
        width=0.008, headwidth=4, headlength=5, zorder=4
    )

    ax1.set_xlim(-1.7, 1.7)
    ax1.set_ylim(-2.2, 2.2)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Robot Trajectory")
    ax1.legend(loc="upper left", fontsize=8)

    # ============================================================
    # PANEL 2: Momenta
    # ============================================================
    ax2.plot(times, l_vals, color="gray", linewidth=2, label="$l_t$")
    ax2.set_ylabel("angular momentum")
    ax2.set_ylim(-0.15, 0.35)
    ax2.set_xlabel("$t$ time (s)")
    ax2.set_xticks(range(0, 11))
    ax2.grid(True)
    ax2.set_title("Momenta")

    ax2_r = ax2.twinx()
    ax2_r.plot(times, eta_norm, "k:", linewidth=2, label=r"$||\eta_t||$")
    ax2_r.set_ylabel("linear momentum")
    ax2_r.set_ylim(-1.5, 3.5)
    ax2_r.grid(False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # ============================================================
    # PANEL 3: Kinetic Energy
    # ============================================================
    ax3.plot(times, energy, "k-", linewidth=2)
    ax3.set_title("Kinetic Energy $H_t$")
    ax3.set_xlabel("$t$ time (s)")
    ax3.set_xticks(range(0, 11))
    ax3.set_ylabel("Energy (J)")
    ax3.grid(True)

    plt.tight_layout()

    out_path = Path(__file__).parent.parent / "results" / "task3_results.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
