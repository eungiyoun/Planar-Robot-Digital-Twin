"""Task 2 — Estimation validation script (Midterm II, Section 2.2).

Runs the 5-stage input sequence with a noisy simulated sensor and the
low-pass / complementary filter estimator.  Produces:
    1. Position plot: x_hat vs ground truth vs raw GPS (x1 and x2)
    2. Heading plot: theta_hat vs ground truth vs compass vs integrated gyro
    3. RMSE table comparing raw measurement error vs filtered error

Outputs saved to ``examples/outputs/``.

NOTE: This script uses NoisySensorSim (a test helper) as a stand-in for
the real sensor module.  Once feat/sensor is merged, replace it with:
    from rse.sensor import GPSCompass, Gyro
    from rse.estimation import SensorAdapter
    sensor = SensorAdapter(GPSCompass(...), Gyro(...), dt=dt)
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rse.system import Robot
from rse.integrator import PositionVerletIntegrator
from rse.simulator import Simulator
from rse.action import ActionProtocol
from rse.estimation import Estimator
from rse.sensor._gpscompass import GPSCompass
from rse.sensor._gyro import Gyro
from rse.estimation._sensor_adapter import SensorAdapter


# ── 5-Stage Action ────────────────────────────────────────────────────
class UnboundedEnvironment:
    """An empty environment to satisfy the 'unbounded plane' requirement."""

    def detect_collision(self, robot: Robot) -> None:
        pass

    def compute_loads(self, robot: Robot, time_step: float) -> None:
        pass


class Task2Action(ActionProtocol):
    """Piecewise control for the 5-stage input sequence.

    Stage 1  [0, 2)   stationary
    Stage 2  [2, 3)   forward  u_l = u_r = 1 N
    Stage 3  [3, 6)   coast
    Stage 4  [6, 7)   rotate   u_l = 1, u_r = -1 N
    Stage 5  [7, 10)  coast
    """

    def __init__(self, track_width: float) -> None:
        self.w = track_width

    def apply(self, robot, current_time: float) -> None:
        t = current_time
        if 2.0 <= t < 3.0:
            ul, ur = 1.0, 1.0  # forward
        elif 6.0 <= t < 7.0:
            ul, ur = 1.0, -1.0  # in-place rotation
        else:
            ul, ur = 0.0, 0.0  # stationary / coast

        force_x = ul + ur
        torque = (ur - ul) * (self.w / 2.0)
        robot.apply_force_body(np.array([force_x, 0.0]))
        robot.apply_torque(torque)


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    print("Task 2 — Estimation Validation (Midterm II)")
    print("=" * 60)

    # --- System setup (same parameters as task3.py) ---
    dt = 0.001
    robot = Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)
    robot.position = np.array([0, 0])

    environment = UnboundedEnvironment()
    integrator = PositionVerletIntegrator()
    action = Task2Action(track_width=robot.track_width)
    simulator = Simulator(robot, environment, integrator, dt)

    # --- Sensor (NoisySensorSim stand-in; replace with SensorAdapter later) ---
    gps = GPSCompass(gps_variance=0.01, compass_variance=0.25)
    gyro = Gyro(gyro_variance=0.04, bias_initial=0.1, bias_noise_intensity=0.01)

    sensor = SensorAdapter(gps, gyro, dt=dt)
    estimator = Estimator(sensor, alpha_x=0.05, alpha_theta=0.99, dt=dt)

    time_horizon = 10.0
    num_steps = int(time_horizon / dt)

    # --- Pre-allocate recording arrays ---
    times = np.zeros(num_steps)
    true_pos = np.zeros((num_steps, 2))
    true_theta = np.zeros(num_steps)
    gps_pos = np.zeros((num_steps, 2))
    est_pos = np.zeros((num_steps, 2))
    compass_theta = np.zeros(num_steps)
    gyro_integrated = np.zeros(num_steps)
    est_theta = np.zeros(num_steps)

    # --- Simulation loop ---
    for i in tqdm(range(num_steps), desc="Simulating"):
        # Record ground truth BEFORE stepping
        Q_true = robot.orientation
        true_pos[i] = robot.position.copy()
        true_theta[i] = np.arctan2(Q_true[1, 0], Q_true[0, 0])
        times[i] = simulator.time_current

        # Physics step
        simulator.step(action)

        # Sensor measurement
        sensor.measure(robot)

        # Estimator update
        estimator.update()

        # Record measurements and estimates
        gps_pos[i] = sensor.y_gps.copy()
        compass_theta[i] = sensor.y_compass
        est_pos[i] = estimator.x_hat.copy()
        est_theta[i] = estimator.theta_hat

        # Integrated gyro (for drift comparison)
        if i == 0:
            gyro_integrated[i] = wrap_angle(sensor.y_compass)
        else:
            gyro_integrated[i] = wrap_angle(gyro_integrated[i - 1] + sensor.y_gyro * dt)

    print("Simulation complete.\n")

    # --- RMSE computation (skip first 500 steps for transient) ---
    skip = 500

    rmse_gps_pos = np.sqrt(
        np.mean(np.sum((gps_pos[skip:] - true_pos[skip:]) ** 2, axis=1))
    )
    rmse_est_pos = np.sqrt(
        np.mean(np.sum((est_pos[skip:] - true_pos[skip:]) ** 2, axis=1))
    )

    rmse_compass = np.sqrt(
        np.mean(angle_diff(compass_theta[skip:], true_theta[skip:]) ** 2)
    )
    rmse_gyro = np.sqrt(
        np.mean(angle_diff(gyro_integrated[skip:], true_theta[skip:]) ** 2)
    )
    rmse_est_theta = np.sqrt(
        np.mean(angle_diff(est_theta[skip:], true_theta[skip:]) ** 2)
    )

    pos_improvement = (1.0 - rmse_est_pos / rmse_gps_pos) * 100
    heading_vs_compass = (1.0 - rmse_est_theta / rmse_compass) * 100
    heading_vs_gyro = (1.0 - rmse_est_theta / rmse_gyro) * 100

    print("RMSE Comparison")
    print("-" * 70)
    print(f"  {'Metric':<30} {'Raw':>10} {'Filtered':>10} {'Improv.':>10}")
    print("-" * 70)
    print(
        f"  {'Position (GPS baseline)':<30} {rmse_gps_pos:>10.3f} {rmse_est_pos:>10.3f} {pos_improvement:>9.1f}%"
    )
    print(
        f"  {'Heading (compass baseline)':<30} {rmse_compass:>10.3f} {rmse_est_theta:>10.3f} {heading_vs_compass:>9.1f}%"
    )
    print(
        f"  {'Heading (gyro-only baseline)':<30} {rmse_gyro:>10.3f} {rmse_est_theta:>10.3f} {heading_vs_gyro:>9.1f}%"
    )
    print("-" * 70)

    # --- Output directory ---
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # ── Figure 1: Position Estimation via Low-Pass Filter ──
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig1.suptitle("Position Estimation via Low-Pass Filter", fontsize=14)

    ax1.plot(times, true_pos[:, 0], "k-", linewidth=2, label="Ground truth $x_1$")
    ax1.plot(
        times,
        gps_pos[:, 0],
        ".",
        color="lightcoral",
        markersize=0.5,
        alpha=0.5,
        label="Raw GPS",
    )
    ax1.plot(times, est_pos[:, 0], "b--", linewidth=1.5, label="Filtered $\\hat{x}_1$")
    ax1.set_ylabel("$x_1$ (m)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.5)

    ax2.plot(times, true_pos[:, 1], "k-", linewidth=2, label="Ground truth $x_2$")
    ax2.plot(
        times,
        gps_pos[:, 1],
        ".",
        color="lightcoral",
        markersize=0.5,
        alpha=0.5,
        label="Raw GPS",
    )
    ax2.plot(times, est_pos[:, 1], "b--", linewidth=1.5, label="Filtered $\\hat{x}_2$")
    ax2.set_ylabel("$x_2$ (m)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.5)

    fig1.savefig(str(out_dir / "task2_position.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_dir / 'task2_position.png'}")

    # ── Figure 2: Orientation Estimation via Complementary Filter ──
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    fig2.suptitle("Orientation Estimation via Complementary Filter", fontsize=14)

    ax3.plot(times, true_theta, "k-", linewidth=2, label="Ground truth $\\theta$")
    ax3.plot(
        times,
        compass_theta,
        ".",
        color="lightcoral",
        markersize=0.5,
        alpha=0.5,
        label="Raw compass",
    )
    # ax3.plot(times, gyro_integrated, "--", color="orange", linewidth=1,
    #          label="Integrated gyro (drift)")
    ax3.plot(
        times,
        est_theta,
        "b--",
        linewidth=1.5,
        label="Complementary filter $\\hat{\\theta}$",
    )
    ax3.set_ylabel("Heading (rad)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.5)

    fig2.savefig(str(out_dir / "task2_heading.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'task2_heading.png'}")

    # Compute RMSEs
    rmse_gps_x1 = np.sqrt(np.mean((gps_pos[skip:, 0] - true_pos[skip:, 0]) ** 2))
    rmse_gps_x2 = np.sqrt(np.mean((gps_pos[skip:, 1] - true_pos[skip:, 1]) ** 2))
    rmse_hat_x1 = np.sqrt(np.mean((est_pos[skip:, 0] - true_pos[skip:, 0]) ** 2))
    rmse_hat_x2 = np.sqrt(np.mean((est_pos[skip:, 1] - true_pos[skip:, 1]) ** 2))

    # Compute improvements
    imp_x1 = (1 - rmse_hat_x1 / rmse_gps_x1) * 100
    imp_x2 = (1 - rmse_hat_x2 / rmse_gps_x2) * 100
    heading_vs_compass = (1 - rmse_est_theta / rmse_compass) * 100

    # Create figure
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))
    fig3.suptitle("Task 2: RMSE Analysis", fontsize=16, y=1.05)

    # ── Position RMSE ─────────────────────────
    pos_labels = [
        "GPS $x_1$",
        "Filter $\\hat{x}_1$",
        "GPS $x_2$",
        "Filter $\\hat{x}_2$",
    ]
    pos_values = [rmse_gps_x1, rmse_hat_x1, rmse_gps_x2, rmse_hat_x2]
    pos_colors = ["lightgrey", "lightblue", "lightgrey", "lightblue"]

    bars1 = axes3[0].bar(
        pos_labels, pos_values, color=pos_colors, edgecolor="black", width=0.5
    )

    axes3[0].set_title(
        f"Position RMSE (x₁: {imp_x1:.0f}% ↓, x₂: {imp_x2:.0f}% ↓)", fontsize=14
    )
    axes3[0].set_ylabel("RMSE (m)", fontsize=12)
    axes3[0].grid(axis="y", alpha=0.4)

    # Annotate values
    for bar in bars1:
        yval = bar.get_height()
        axes3[0].text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(pos_values) * 0.02),
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # ── Heading RMSE ─────────────────────────
    head_labels = ["Raw Compass", "Integrated\nGyro", "Complementary\nFilter"]
    head_values = [rmse_compass, rmse_gyro, rmse_est_theta]
    head_colors = ["lightgrey", "lightyellow", "lightblue"]

    bars2 = axes3[1].bar(
        head_labels, head_values, color=head_colors, edgecolor="black", width=0.5
    )

    axes3[1].set_title(
        f"Heading RMSE ({heading_vs_compass:.0f}% ↓ vs compass)", fontsize=14
    )
    axes3[1].set_ylabel("RMSE (rad)", fontsize=12)
    axes3[1].grid(axis="y", alpha=0.4)
    axes3[1].set_ylim(0, max(head_values) * 1.1)
    axes3[0].set_ylim(0, max(pos_values) * 1.1)
    # Annotate values
    for bar in bars2:
        yval = bar.get_height()
        axes3[1].text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(head_values) * 0.02),
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Save and show
    fig3.savefig(str(out_dir / "task2_rmse_bars.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'task2_rmse_bars.png'}")
    plt.show()


if __name__ == "__main__":
    main()
