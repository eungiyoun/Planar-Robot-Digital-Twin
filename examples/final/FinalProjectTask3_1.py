import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from rse.system import Robot
from rse.controller import Controller
from rse.estimation import Estimator
from rse.environment import Environment
from rse.sensor import Sensor
from rse.planning import TEBPlanner, RoboticPlanner, ClampedCubicSpline, inflate_grid

GRID_CMAP = ListedColormap(['#ffffff', '#ffb380', '#404040'])
_grid_legend = [
    mpatches.Patch(color='#ffffff', ec='gray', label='Free'),
    mpatches.Patch(color='#ffb380', label='Inflated zone'),
    mpatches.Patch(color='#404040', label='Obstacle'),
]

KP_HEADING = 0.5
KI_HEADING = 0.2
DECAY_FACTOR_HEADING = 0.5

KP_LATERAL = 2.25
KI_LATERAL = 0.1
DECAY_FACTOR_LATERAL = 1.0



# ==========================================
# 1. Environment & Static Planner Setup
# ==========================================
def setup_environment_and_plan():
    res = 0.1
    width_m, height_m = 3.0, 4.0
    grid_w, grid_h = int(width_m / res), int(height_m / res)
    occ_grid = np.ones((grid_w, grid_h))

    def set_free(grid, x_min, x_max, y_min, y_max, res, grid_w, grid_h):
        ix_min = int(np.floor(x_min / res)) + (grid_w // 2)
        ix_max = int(np.floor(x_max / res)) + (grid_w // 2)
        iy_min = int(np.floor(y_min / res)) + (grid_h // 2)
        iy_max = int(np.floor(y_max / res)) + (grid_h // 2)
        grid[max(0, ix_min):min(grid_w, ix_max),
             max(0, iy_min):min(grid_h, iy_max)] = 0.0

    set_free(occ_grid, -1.5, 1.5, -2.0, -1.0, res, grid_w, grid_h)
    set_free(occ_grid, -0.9, 0.9, -1.0, 1.0, res, grid_w, grid_h)
    set_free(occ_grid, -1.5, 1.5, 1.0, 2.0, res, grid_w, grid_h)

    # --- Obstacle Inflation ---
    inf_grid = inflate_grid(occ_grid, 0.2, res)

    # --- A* Planning ---
    planner = RoboticPlanner(resolution=res, radius=0.2, grid_dims=(grid_w, grid_h))
    start_pos, goal_pos = (-1.2, -1.3), (1.2, 1.3)

    print("Running A* Guidance...")
    p_inf = planner.a_star(start_pos, goal_pos, inf_grid, use_inflation=True)

    # --- TEB Waypoint Generation ---
    print("Running Static TEB Planner...")
    obstacle_points = TEBPlanner.grid_to_obstacle_points(inf_grid, planner, include_soft=True)

    teb_params = {
        "vmax": 0.5,
        "dmin": 0.25,
        "n_waypoints": 20, 
        "gamma_time": 2.0, # total time optimize
        "gamma_smooth": 6.0, # segment length optimize big = tighter
        "gamma_obs": 40.0, # bigger is to avoid obstacle
        "gamma_vel": 200.0, # big is don't go over speed limit
        "gamma_bc": 20.0,
        "dt_min": 0.05,
        "dt_max": 1.0,
    }

    teb = TEBPlanner(obstacle_points=obstacle_points, **teb_params)
    teb_result = teb.solve(p_inf)

    print(f"TEB success: {teb_result.success}")
    print(f"TEB message: {teb_result.message}")

    times = []
    waypoints = []
    for tk, xk in teb_result.timed_waypoints:
        times.append(tk)
        waypoints.append(xk)

    return (
        np.array(times), 
        np.array(waypoints), 
        p_inf, 
        teb_result.segment_speed, 
        inf_grid, 
        width_m, 
        height_m,
        start_pos,
        goal_pos
    )


# ==========================================
# 2. Controller Simulation Loop
# ==========================================
def run_simulation(target_trajectory, max_t):
    dt = 0.005  
    t_steps = np.arange(0, max_t + 1.0, dt) 
    
    environment = Environment()
    robot = Robot()
    sensor = Sensor()

    robot.position = target_trajectory.get_pose(0.0)
    estimator = Estimator(sensor)

    heading_controller = Controller(estimator=estimator, Kp=KP_HEADING, Ki=KI_HEADING, decay_factor=DECAY_FACTOR_HEADING)
    lateral_controller = Controller(estimator=estimator, Kp=KP_LATERAL, Ki=KI_LATERAL, decay_factor=DECAY_FACTOR_LATERAL)

    log = {
        "t": t_steps,
        "actual": np.zeros((len(t_steps), 2)),
        "est": np.zeros((len(t_steps), 2)),
        "target": np.zeros((len(t_steps), 2)),
        "error": np.zeros((len(t_steps), 2)),
        "control": np.zeros((len(t_steps), 2)),
    }

    print("Running PI Tracking Control Loop...")
    for i, t in enumerate(tqdm(t_steps, desc="Simulation Steps", unit="step")):
        x_star = target_trajectory.get_pose(t)

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
        log["control"][i] = np.array([u_l, u_r])

    return log


# ==========================================
# 3. Main Execution & Deliverable Plotting
# ==========================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    prefix_t2 = "task2_1"
    prefix_t3 = "task3_1"

    (times, waypoints, p_inf, segment_speed, 
     inf_grid, width_m, height_m, start_pos, goal_pos) = setup_environment_and_plan()

    # ===================== Task 2.1: TEB vs A* Plot =====================
    print(f"\nSaving Task 2.1 Planner Plot to {output_dir}...")
    fig0, ax0 = plt.subplots(figsize=(7, 9))
    extent = [-width_m / 2, width_m / 2, -height_m / 2, height_m / 2]
    ax0.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
               extent=extent, interpolation='nearest')

    if p_inf:
        x_astar, y_astar = zip(*p_inf)
        ax0.plot(x_astar, y_astar, 'r--', linewidth=2, label='A* inflated path')

    ax0.plot(waypoints[:, 0], waypoints[:, 1], 'bo-', linewidth=2, markersize=4, label='TEB timed waypoints')
    ax0.scatter(*start_pos, color='lime', edgecolors='black', s=100, label='Start', zorder=5)
    ax0.scatter(*goal_pos, color='gold', edgecolors='black', s=100, label='Goal', zorder=5)

    ax0.set_title("Task 2.1: TEB-optimized timed waypoints")
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles=_grid_legend + handles, labels=[p.get_label() for p in _grid_legend] + labels)
    ax0.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix_t2}_teb_vs_astar.png"), dpi=300)
    plt.close(fig0)

    # ===================== Task 2.1: Segment Speed Plot =====================
    t_waypoints_sp = np.array(times)
    t_mid_sp = 0.5 * (t_waypoints_sp[:-1] + t_waypoints_sp[1:])
    fig0b, ax0b = plt.subplots(figsize=(8, 4))
    ax0b.step(t_waypoints_sp[:-1], segment_speed, where='post', color='b', alpha=0.4, label='_nolegend_')
    ax0b.scatter(t_mid_sp, segment_speed, color='b', s=40, zorder=3, label=r'Segment speed $\bar{v}_k$')
    ax0b.axhline(0.5, color='r', linestyle='--', label='vmax = 0.5 m/s')
    ax0b.set_title("Task 2.1: Inter-waypoint speed")
    ax0b.set_xlabel("Time t [s]")
    ax0b.set_ylabel("Speed [m/s]")
    ax0b.grid(True)
    ax0b.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix_t2}_segment_speed.png"), dpi=300)
    plt.close(fig0b)

    # ===================== Smoothing & Tracking (Task 3.1) =====================
    print("\nFitting C2-smooth Clamped Cubic Spline...")
    target_traj = ClampedCubicSpline(times, waypoints)
    log = run_simulation(target_traj, target_traj.max_t)

    # ===================== Task 3.1: 2D Trajectory Plot =====================
    print(f"Saving Task 3.1 Tracking Plots to {output_dir}...")
    fig1, ax1 = plt.subplots(figsize=(7, 9))
    ax1.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
               extent=extent, interpolation='nearest')
    ax1.plot(waypoints[:, 0], waypoints[:, 1], "m:", linewidth=2, alpha=0.7, label="Unsmoothed TEB $x_{TEB}(t)$")
    ax1.plot(log["target"][:, 0], log["target"][:, 1], "g--", linewidth=2, label="Smoothed Reference $x^*(t)$")
    ax1.plot(log["est"][:, 0], log["est"][:, 1], "b-.", alpha=0.6, label="Estimated Path $\\hat{x}_k$")
    ax1.plot(log["actual"][:, 0], log["actual"][:, 1], "r-", linewidth=2, alpha=0.8, label="Actual Path $x_k$")
    
    ax1.scatter(*start_pos, color='lime', edgecolors='black', s=100, label='Start', zorder=5)
    ax1.scatter(*goal_pos, color='gold', edgecolors='black', s=100, label='Goal', zorder=5)

    ax1.set_title("Task 3.1: 2D Trajectory Generation and Tracking")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=_grid_legend + handles, labels=[p.get_label() for p in _grid_legend] + labels)
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix_t3}_trajectory.png"), dpi=300)
    plt.close(fig1)

    # ===================== Task 3.1: Time Series Plots =====================
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax2_1.plot(log["t"], log["error"][:, 0], label="Heading Error $e_{heading}$")
    ax2_1.plot(log["t"], log["error"][:, 1], label="Lateral Error $e_{lateral}$")
    ax2_1.set_title("Tracking Error over Time")
    ax2_1.set_ylabel("Error [m]")
    ax2_1.legend()
    ax2_1.grid(True)

    ax2_2.plot(log["t"], log["control"][:, 0], label="Left Wheel Control $u_l$")
    ax2_2.plot(log["t"], log["control"][:, 1], label="Right Wheel Control $u_r$")
    ax2_2.set_title("Control Inputs over Time")
    ax2_2.set_xlabel("Time [s]")
    ax2_2.set_ylabel("Control Command")
    ax2_2.legend()
    ax2_2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix_t3}_time_series.png"), dpi=300)
    plt.close(fig2)

    # ===================== Task 3.1: Synchronized Animation =====================
    print(f"Generating Synchronized Animation in {output_dir} (This may take a minute)...")
    fig_anim, ax_anim = plt.subplots(figsize=(7, 9))
    ax_anim.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
                   extent=extent, interpolation='nearest')
    ax_anim.set_xlim(-width_m / 2, width_m / 2)
    ax_anim.set_ylim(-height_m / 2, height_m / 2)
    ax_anim.set_title("Task 3.1: Real-time Trajectory Tracking")
    ax_anim.set_xlabel("X [m]")
    ax_anim.set_ylabel("Y [m]")
    ax_anim.grid(True)

    ax_anim.plot(waypoints[:, 0], waypoints[:, 1], "m:", alpha=0.5, label="Unsmoothed TEB $x_{TEB}(t)$")

    actual_trace, = ax_anim.plot([], [], "r-", alpha=0.6, label="Actual Path")
    target_trace, = ax_anim.plot([], [], "g--", alpha=0.6, label="Smoothed Trajectory")
    robot_dot, = ax_anim.plot([], [], "ro", markersize=8, label="Robot")
    ref_dot, = ax_anim.plot([], [], "go", markersize=6, label="Reference Target")
    error_line, = ax_anim.plot([], [], "k-", linewidth=1, label="Tracking Error $e_k$")

    ax_anim.legend(loc="upper left", fontsize='small')

    frame_step = 10
    anim_frames = range(0, len(log["t"]), frame_step)

    def update(frame):
        actual_trace.set_data(log["actual"][:frame, 0], log["actual"][:frame, 1])
        target_trace.set_data(log["target"][:frame, 0], log["target"][:frame, 1])
        
        rx, ry = log["actual"][frame, 0], log["actual"][frame, 1]
        tx, ty = log["target"][frame, 0], log["target"][frame, 1]
        
        robot_dot.set_data([rx], [ry])
        ref_dot.set_data([tx], [ty])
        
        error_line.set_data([rx, tx], [ry, ty])

        return actual_trace, target_trace, robot_dot, ref_dot, error_line

    ani = animation.FuncAnimation(
        fig_anim, update, frames=tqdm(anim_frames, desc="Rendering Animation Frames", leave=False), 
        interval=50, blit=True
    )

    ani.save(os.path.join(output_dir, f"{prefix_t3}_animation.gif"), writer="pillow", fps=20)
    plt.close(fig_anim)

    print("\nAll deliverables successfully saved.")


if __name__ == "__main__":
    main()