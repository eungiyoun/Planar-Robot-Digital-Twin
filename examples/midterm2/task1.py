import sys
import numpy as np
import matplotlib.pyplot as plt

from rse.system._robot import Robot
from rse.sensor._gpscompass import GPSCompass
from rse.sensor._gyro import Gyro
from rse.integrator._position_verlet import PositionVerletIntegrator
from rse.simulator._simulator import Simulator
from rse.action._action import ActionProtocol


def wrap_angle(angle: float) -> float:
    """Wraps an angle to the [-pi, pi] range."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# --- 1. Unbounded Environment ---
class UnboundedEnvironment:
    """An empty environment to satisfy the 'unbounded plane' requirement."""

    def detect_collision(self, robot: Robot) -> None:
        pass

    def compute_loads(self, robot: Robot, time_step: float) -> None:
        pass


# --- 2. The Midterm II Task 1 Action Sequence ---
class Task1Mid2Action(ActionProtocol):
    """5-stage input sequence for Midterm II Task 1."""

    def __init__(self, track_width: float):
        self.w = track_width

    def apply(self, robot: Robot, current_time: float) -> None:
        t = current_time
        # 2s stationary, 1s forward, 3s coast, 1s rotate, 3s coast
        if 0.0 <= t < 2.0:
            ul, ur = 0.0, 0.0
        elif 2.0 <= t < 3.0:
            ul, ur = 1.0, 1.0
        elif 3.0 <= t < 6.0:
            ul, ur = 0.0, 0.0
        elif 6.0 <= t < 7.0:
            ul, ur = 1.0, -1.0  # Left=1, Right=-1 (In-place rotation)
        else:
            ul, ur = 0.0, 0.0

        # Convert to body forces
        force_x = ul + ur
        torque = (ur - ul) * (self.w / 2.0)

        robot.apply_force_body(np.array([force_x, 0.0]))
        robot.apply_torque(torque)


def main():
    # --- Initialization ---
    robot = Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)
    environment = UnboundedEnvironment()
    integrator = PositionVerletIntegrator()
    action = Task1Mid2Action(track_width=robot.track_width)

    # Initialize Sensors with exact Table 2 parameters
    gps_compass = GPSCompass(gps_variance=0.01, compass_variance=0.25, seed=42)
    gyro = Gyro(
        gyro_variance=0.04, bias_initial=0.1, bias_noise_intensity=0.01, seed=42
    )

    dt = 0.001
    simulator = Simulator(robot, environment, integrator, time_step=dt)

    time_horizon = 10.0
    num_steps = int(time_horizon / dt)

    # --- Data Logging Arrays ---
    times = []

    # Ground Truth
    true_x, true_y, true_theta, true_omega = [], [], [], []

    # Measurements
    meas_gps_x, meas_gps_y, meas_compass, meas_gyro = [], [], [], []

    # Integrated Gyro Heading tracker
    current_gyro_heading = 0.0
    int_gyro_headings = []

    # --- Simulation Loop ---
    for k in range(num_steps):
        t = simulator.time_current

        # 1. Extract True State
        tx, ty = robot.position
        ttheta = np.arctan2(robot.orientation[1, 0], robot.orientation[0, 0])
        tomega = robot.angular_momentum / robot.moment_of_inertia

        # 2. Take Noisy Measurements
        gps_reading = gps_compass.measure(robot)
        gyro_reading = gyro.measure(robot, dt)

        # 3. Integrate Gyroscope
        current_gyro_heading += gyro_reading.omega * dt

        # Log Data
        times.append(t)
        true_x.append(tx)
        true_y.append(ty)
        true_theta.append(ttheta)
        true_omega.append(tomega)

        meas_gps_x.append(gps_reading.gps[0])
        meas_gps_y.append(gps_reading.gps[1])
        meas_compass.append(gps_reading.compass)
        meas_gyro.append(gyro_reading.omega)
        int_gyro_headings.append(current_gyro_heading)

        # 4. Step Physics Engine
        simulator.step(action)

    # Convert to numpy arrays for easy math
    true_x, true_y, true_theta, true_omega = map(
        np.array, [true_x, true_y, true_theta, true_omega]
    )
    meas_gps_x, meas_gps_y, meas_compass, meas_gyro = map(
        np.array, [meas_gps_x, meas_gps_y, meas_compass, meas_gyro]
    )
    int_gyro_headings = np.array(int_gyro_headings)

    # --- Statistical Analysis (Residuals) ---
    res_x = meas_gps_x - true_x
    res_y = meas_gps_y - true_y
    res_compass = wrap_angle(
        meas_compass - true_theta
    )  # Wrap angle to prevent -pi to pi jumps
    res_gyro = meas_gyro - true_omega

    print("\n--- Empirical Noise Validation ---")
    print(f"GPS X Variance:       {np.var(res_x):.4f} (Target: 0.0100)")
    print(f"GPS Y Variance:       {np.var(res_y):.4f} (Target: 0.0100)")
    print(f"Compass Variance:     {np.var(res_compass):.4f} (Target: 0.2500)")
    # Note: Gyro residual variance will be slightly higher than 0.04 because the residual includes the drifting bias!
    print(f"Gyro Total Variance:  {np.var(res_gyro):.4f} (Target: >0.0400 due to bias)")

    # --- Plotting ---

    # Plot 1: Time Series Overlays
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 10))
    axs1[0].plot(
        times, meas_gps_x, "b.", markersize=1, alpha=0.3, label="Measured GPS X"
    )
    axs1[0].plot(times, true_x, linewidth=2, label="True X")
    axs1[0].plot(
        times, meas_gps_y, "r.", markersize=1, alpha=0.3, label="Measured GPS Y"
    )
    axs1[0].plot(times, true_y, linewidth=2, label="True Y")
    axs1[0].set_ylabel("Position (m)")
    axs1[0].legend()
    axs1[0].grid(True)
    axs1[0].set_title("GPS measurements vs Ground Truth Position")

    axs1[1].plot(
        times, meas_compass, "b.", markersize=1, alpha=0.5, label="Measured Compass"
    )
    axs1[1].plot(times, true_theta, "k-", linewidth=2, label="True Heading")
    axs1[1].set_ylabel("Heading (rad)")
    axs1[1].legend()
    axs1[1].grid(True)
    axs1[1].set_title("Compass measurements vs Ground Truth Heading")

    axs1[2].plot(times, meas_gyro, "g.", markersize=1, alpha=0.5, label="Measured Gyro")
    axs1[2].plot(times, true_omega, "k-", linewidth=2, label="True Angular Velocity")
    axs1[2].set_xlabel("Time (s)")
    axs1[2].set_ylabel("Angular Velocity (rad/s)")
    axs1[2].legend()
    axs1[2].grid(True)
    axs1[2].set_title("Gyro measurements vs Ground Truth Angular Velocity")

    # Plot 2: Residual Histograms
    fig2, axs2 = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    axs = axs2.flatten()

    # X-axis range for drawing the smooth math curves
    x_val = np.linspace(-1.5, 1.5, 300)

    def plot_hist(ax, data, spec_var, title):
        """Helper to plot a clean histogram with a theoretical Gaussian curve."""
        # 1. Simple, uniform histogram bars
        ax.hist(data, bins=50, density=True, color="steelblue", alpha=0.8)

        # 2. The fitted theoretical curve (N(0, sigma^2))
        pdf = (1.0 / np.sqrt(2 * np.pi * spec_var)) * np.exp(
            -0.5 * (x_val**2) / spec_var
        )
        ax.plot(
            x_val,
            pdf,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Theory $\\sigma^2={spec_var}$",
        )

        # 3. Simple text and formatting
        ax.set_title(f"{title}\nEmpirical Var: {np.var(data):.4f}")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right")

    # Top Left: GPS X
    plot_hist(axs[0], res_x, 0.01, "GPS X Residuals")
    axs[0].set_ylabel("Density")

    # Top Right: GPS Y
    plot_hist(axs[1], res_y, 0.01, "GPS Y Residuals")

    # Bottom Left: Compass
    plot_hist(axs[2], res_compass, 0.25, "Compass Residuals")
    axs[2].set_xlabel("Residual Error")
    axs[2].set_ylabel("Density")

    # Bottom Right: Gyroscope
    plot_hist(axs[3], res_gyro, 0.04, "Gyroscope Residuals (With Bias)")
    axs[3].set_xlabel("Residual Error")

    # Lock the global axes limits
    axs[0].set_xlim(-1.5, 1.5)
    axs[0].set_ylim(0, 5)

    plt.tight_layout()

    # Plot 3: Gyroscopic Drift Analysis
    plt.figure(figsize=(10, 5))
    plt.plot(times, true_theta, "k-", linewidth=3, label="True Heading")
    plt.plot(times, meas_compass, "b.", markersize=1, alpha=0.5, label="Noisy Compass")
    plt.plot(
        times,
        wrap_angle(int_gyro_headings),
        "r--",
        linewidth=2,
        label="Integrated Gyroscope (Drifting)",
    )
    plt.title("Drift Analysis: Integrated Gyroscope vs True Heading")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (rad)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
