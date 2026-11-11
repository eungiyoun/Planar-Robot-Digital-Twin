import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from rse.system import Robot
from rse.sensor import GPSCompass, Gyro


def rotation_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def main() -> None:
    # Create robot
    robot = Robot(
        mass=2.0,
        moment_of_inertia=0.05,
        radius=0.2,
        track_width=0.15,
    )

    # Assign a known true state
    robot.position = np.array([1.0, -0.5])
    robot.orientation = rotation_matrix(np.pi / 6)  # 30 deg
    robot.linear_momentum = np.array([0.4, 0.0])
    robot.angular_momentum = 0.1

    # Create sensors
    gps_compass = GPSCompass(
        gps_variance=0.01,
        compass_variance=0.25,
        seed=42,
    )

    gyro = Gyro(
        gyro_variance=0.04,
        bias_initial=0.1,
        bias_noise_intensity=0.05,
        seed=42,
    )

    dt = 0.001
    n_steps = 2000

    # True values
    theta_true = np.arctan2(robot.orientation[1, 0], robot.orientation[0, 0])
    omega_true = robot.angular_momentum / robot.moment_of_inertia

    # Storage
    gps_x = []
    gps_y = []
    compass_vals = []
    gyro_vals = []
    bias_vals = []

    for _ in range(n_steps):
        gps_measurement = gps_compass.measure(robot)
        gyro_measurement = gyro.measure(robot, time_step=dt)

        gps_x.append(gps_measurement.gps[0])
        gps_y.append(gps_measurement.gps[1])
        compass_vals.append(gps_measurement.compass)
        gyro_vals.append(gyro_measurement.omega)
        bias_vals.append(gyro_measurement.bias)

    gps_x = np.array(gps_x)
    gps_y = np.array(gps_y)
    compass_vals = np.array(compass_vals)
    gyro_vals = np.array(gyro_vals)
    bias_vals = np.array(bias_vals)

    # Annotated terminal output
    print("\n================ TRUE STATE ================")
    print(f"True position x: {robot.position[0]:.4f}")
    print(f"True position y: {robot.position[1]:.4f}")
    print(
        f"True heading theta: {theta_true:.4f} rad ({np.degrees(theta_true):.2f} deg)"
    )
    print(f"True angular velocity omega: {omega_true:.4f} rad/s")

    print("\n=========== SAMPLE NOISY MEASUREMENTS ===========")
    for i in range(5):
        print(f"Sample {i + 1}:")
        print(f"  GPS measurement        = [{gps_x[i]:.4f}, {gps_y[i]:.4f}]")
        print(f"  Compass measurement    = {compass_vals[i]:.4f} rad")
        print(f"  Gyro measurement       = {gyro_vals[i]:.4f} rad/s")
        print(f"  Internal gyro bias     = {bias_vals[i]:.4f}")
        print()

    print("=============== SUMMARY STATISTICS ===============")
    print(f"GPS x mean: {np.mean(gps_x):.4f}, std: {np.std(gps_x):.4f}")
    print(f"GPS y mean: {np.mean(gps_y):.4f}, std: {np.std(gps_y):.4f}")
    print(f"Compass mean: {np.mean(compass_vals):.4f}, std: {np.std(compass_vals):.4f}")
    print(f"Gyro mean: {np.mean(gyro_vals):.4f}, std: {np.std(gyro_vals):.4f}")
    print(f"Bias start: {bias_vals[0]:.4f}, Bias end: {bias_vals[-1]:.4f}")

    # Graphical output
    t = np.arange(n_steps) * dt

    plt.figure(figsize=(6, 6))
    plt.scatter(gps_x, gps_y, s=18, alpha=0.7, label="Noisy GPS samples")
    plt.scatter(
        robot.position[0], robot.position[1], marker="x", s=120, label="True position"
    )
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("GPS Sensor Output")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    plt.figure(figsize=(8, 4))
    plt.plot(t, compass_vals, label="Noisy compass")
    plt.axhline(theta_true, linestyle="--", label="True heading")
    plt.xlabel("time (s)")
    plt.ylabel("heading (rad)")
    plt.title("Compass Output")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(8, 4))
    plt.plot(t, gyro_vals, label="Noisy gyro")
    plt.axhline(omega_true, linestyle="--", label="True angular velocity")
    plt.xlabel("time (s)")
    plt.ylabel("angular velocity (rad/s)")
    plt.title("Gyro Output")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(8, 4))
    plt.plot(t, bias_vals, label="Gyro bias")
    plt.xlabel("time (s)")
    plt.ylabel("bias")
    plt.title("Gyro Bias Drift")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
