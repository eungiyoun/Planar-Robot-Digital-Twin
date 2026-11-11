import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch

# --- Project Imports ---
from rse.system import Robot
from rse.controller import Controller
from rse.estimation import Estimator
from rse.environment import Environment
from rse.sensor import Sensor
from utils import TargetTrajectory


def run_simulation(delta_sharpness: float = 0.5):
    dt = 0.001
    t_steps = np.arange(0, 10.0, dt)

    target = TargetTrajectory(delta=delta_sharpness)
    environment = Environment()
    robot = Robot()
    sensor = Sensor()

    robot.position = np.array([-1.2, -1.3], dtype=float)

    estimator = Estimator(sensor)

    heading_controller = Controller(estimator=estimator, Kp=0.13, Ki=0.2, decay_factor=0.0)
    lateral_controller = Controller(estimator=estimator, Kp=1.3, Ki=1.0, decay_factor=0.0)

    log = {
        "t": t_steps,
        "actual": np.zeros((len(t_steps), 2)),
        "est": np.zeros((len(t_steps), 2)),
        "target": np.zeros((len(t_steps), 2)),
        "error": np.zeros((len(t_steps), 2)),
        "control": np.zeros((len(t_steps), 2)),
        "speed": np.zeros(len(t_steps)),
        "heading": np.zeros((len(t_steps), 2)),
        "u_l": np.zeros(len(t_steps)),
        "u_r": np.zeros(len(t_steps)),
        "u_heading": np.zeros(len(t_steps)),
        "u_lateral": np.zeros(len(t_steps)),
    }

    for i, t in enumerate(t_steps):
        x_star = np.asarray(target.get_pose(t))

        sensor.measure(robot, dt)
        estimator.update()

        x_hat = estimator.x_hat
        Q_hat = estimator.Q_hat
        e_k = Q_hat.T @ (x_star - x_hat)
        e_heading, e_lateral = e_k[0], e_k[1]

        u_heading = heading_controller.control(e_heading)
        u_lateral = lateral_controller.control(e_lateral)

        rot_mat = np.array([[1, -1], [1, 1]])
        u_k = rot_mat @ np.array([u_heading, u_lateral])
        u_l, u_r = u_k[0], u_k[1]

        u_l = np.clip(u_l, -1.0, 1.0)
        u_r = np.clip(u_r, -1.0, 1.0)

        robot.applied_force = np.zeros(2)
        robot.applied_torque = 0.0

        force = np.array([u_l + u_r, 0])
        torque = (u_r - u_l) * (robot.track_width / 2.0)

        robot.apply_force_body(force)
        robot.apply_torque(torque)

        environment.detect_collision(robot)
        environment.compute_loads(robot, dt)

        robot.compute_dynamics(t, dt)
        robot.compute_kinematics(t, dt)

        log["actual"][i] = robot.position.copy()
        log["est"][i] = estimator.x_hat.copy()
        log["target"][i] = x_star.copy()
        log["error"][i] = e_k.copy()
        log["control"][i] = u_k.copy()
        v_body = robot.linear_momentum / robot.mass
        log["speed"][i] = np.linalg.norm(v_body)
        log["heading"][i] = robot.orientation[:, 0].copy()
        log["u_l"][i] = u_l
        log["u_r"][i] = u_r
        log["u_heading"][i] = u_heading
        log["u_lateral"][i] = u_lateral

    return log, environment


def plot_environment_walls(ax, environment):
    if hasattr(environment, "_walls"):
        for wall in environment._walls:
            ax.plot(wall[:, 0], wall[:, 1], "k-", linewidth=2.0)


def main():
    #  Correct relative path
    # output_dir = "./examples/midterm2/outputs"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    delta = 0.0
    tag = f"delta_{delta:.1f}"
    prefix = "task4"

    print(f"Running simulation for delta = {delta}...")
    log, env = run_simulation(delta_sharpness=delta)

    # ===================== Trajectory =====================
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_environment_walls(ax1, env)

    ax1.plot(log["target"][:, 0], log["target"][:, 1], "g--", label="Target")
    ax1.plot(log["est"][:, 0], log["est"][:, 1], "b-.", label="Estimate")
    ax1.plot(log["actual"][:, 0], log["actual"][:, 1], "r-", label="Actual")

    ax1.set_title(f"Trajectory (delta={delta})")
    ax1.legend()
    ax1.grid(True)
    ax1.axis("equal")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_trajectory_{tag}.png", dpi=300)
    plt.close(fig1)

    # ===================== Time Series =====================
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax2_1.plot(log["t"], log["error"][:, 0], label="Error X")
    ax2_1.plot(log["t"], log["error"][:, 1], label="Error Y")
    ax2_1.legend()
    ax2_1.grid(True)

    ax2_2.plot(log["t"], log["control"][:, 0], label="u_x")
    ax2_2.plot(log["t"], log["control"][:, 1], label="u_y")
    ax2_2.legend()
    ax2_2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_time_series_{tag}.png", dpi=300)
    plt.close(fig2)

    # ===================== Animation (GIF) =====================
    fig_anim, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title(f"Animation (delta={delta})")

    (line,) = ax.plot([], [], "r-")

    def update(frame):
        line.set_data(log["actual"][:frame, 0], log["actual"][:frame, 1])
        return (line,)

    ani = animation.FuncAnimation(
        fig_anim,
        update,
        frames=range(0, len(log["t"]), 30),
        interval=50
    )

    ani.save(f"{output_dir}/{prefix}_animation_{tag}.gif", writer="pillow", fps=20)
    plt.close(fig_anim)

    # ===================== Position & Velocity =====================
    fig4, ax_pos = plt.subplots()
    ax_vel = ax_pos.twinx()

    ax_pos.plot(log["t"], log["actual"][:, 0], label="x1")
    ax_pos.plot(log["t"], log["actual"][:, 1], label="x2")
    ax_vel.plot(log["t"], log["speed"], "--", label="|v|")

    ax_pos.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_position_velocity_{tag}.png", dpi=300)
    plt.close(fig4)

    # ===================== Control Inputs =====================
    fig2b, (ax1b, ax2b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1b.plot(log["t"], log["u_heading"], label="heading")
    ax1b.plot(log["t"], log["u_lateral"], label="lateral")
    ax1b.legend()
    ax1b.grid(True)

    ax2b.plot(log["t"], log["u_l"], label="u_l")
    ax2b.plot(log["t"], log["u_r"], label="u_r")
    ax2b.legend()
    ax2b.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_control_inputs_{tag}.png", dpi=300)
    plt.close(fig2b)

    print(f"All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()