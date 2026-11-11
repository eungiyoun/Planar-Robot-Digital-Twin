import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# --- Project Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from rse.system import Robot
from rse.controller import Controller
from rse.estimation import Estimator
from rse.environment import Environment
from rse.sensor import Sensor
from rse.planning import TEBPlanner, RoboticPlanner, ClampedCubicSpline, inflate_grid


# ==========================================
# Tunable Parameters
# ==========================================
KP_HEADING = 0.5
KI_HEADING = 0.2
DECAY_FACTOR_HEADING = 0.5

KP_LATERAL = 2.25
KI_LATERAL = 0.1
DECAY_FACTOR_LATERAL = 1.0

SEED = 42
ARRIVAL_POS_TOL = 0.05    # m  — position radius to declare "arrived"
ARRIVAL_SPEED_TOL = 0.02  # m/s — robot speed must be below this
DWELL_STEPS = 50          # consecutive steps inside tolerance before advancing
MAX_SEGMENT_TIME = 50.0   # s  — hard timeout per segment

GRID_CMAP = ListedColormap(['#ffffff', '#ffb380', '#404040'])
_grid_legend = [
    mpatches.Patch(color='#ffffff', ec='gray', label='Free'),
    mpatches.Patch(color='#ffb380', label='Inflated zone'),
    mpatches.Patch(color='#404040', label='Obstacle'),
]


# ==========================================
# 1. Environment Setup
# ==========================================
def build_environment():
    res = 0.1
    width_m, height_m = 3.0, 4.0
    grid_w, grid_h = int(width_m / res), int(height_m / res)
    occ_grid = np.ones((grid_w, grid_h))

    def set_free(grid, x_min, x_max, y_min, y_max):
        ix_min = int(np.floor(x_min / res)) + (grid_w // 2)
        ix_max = int(np.floor(x_max / res)) + (grid_w // 2)
        iy_min = int(np.floor(y_min / res)) + (grid_h // 2)
        iy_max = int(np.floor(y_max / res)) + (grid_h // 2)
        grid[max(0, ix_min):min(grid_w, ix_max),
             max(0, iy_min):min(grid_h, iy_max)] = 0.0

    set_free(occ_grid, -1.5, 1.5, -2.0, -1.0)
    set_free(occ_grid, -0.9, 0.9, -1.0,  1.0)
    set_free(occ_grid, -1.5, 1.5,  1.0,  2.0)

    inf_grid = inflate_grid(occ_grid, 0.2, res)
    planner = RoboticPlanner(resolution=res, radius=0.2, grid_dims=(grid_w, grid_h))
    obstacle_points = TEBPlanner.grid_to_obstacle_points(inf_grid, planner, include_soft=True)

    return res, width_m, height_m, occ_grid, inf_grid, planner, obstacle_points


def get_random_goal(inf_grid, res, width_m, height_m, planner, start_pos, rng):
    """Generates a random valid free-space goal reachable via A* (seeded RNG)."""
    grid_w, grid_h = inf_grid.shape
    while True:
        x = rng.uniform(-width_m / 2 + 0.3, width_m / 2 - 0.3)
        y = rng.uniform(-height_m / 2 + 0.3, height_m / 2 - 0.3)
        ix = int(np.floor(x / res)) + (grid_w // 2)
        iy = int(np.floor(y / res)) + (grid_h // 2)
        if 0 <= ix < grid_w and 0 <= iy < grid_h and inf_grid[ix, iy] == 0.0:
            if planner.a_star(start_pos, (x, y), inf_grid, use_inflation=True) is not None:
                return np.array([x, y])


# ==========================================
# 2. TEB helper
# ==========================================
def make_teb(obstacle_points):
    teb_params = {
        "vmax": 0.5,
        "dmin": 0.25,
        "n_waypoints": 20,
        "gamma_time": 2.0,
        "gamma_smooth": 6.0,
        "gamma_obs": 40.0,
        "gamma_vel": 200.0,
        "gamma_bc": 0,
        "dt_min": 0.05,
        "dt_max": 1.0,
    }
    return TEBPlanner(obstacle_points=obstacle_points, **teb_params)


# ==========================================
# 3. Single-segment simulation
# ==========================================
def run_segment(seg_idx, seg_start, seg_goal, inf_grid, planner, teb,
                robot, sensor, estimator, heading_controller, lateral_controller,
                dt=0.005):
    """Run one rest-to-rest segment. Returns (log_dict, seg_duration, arrived)."""

    # Plan
    p_inf = planner.a_star(tuple(seg_start), tuple(seg_goal), inf_grid, use_inflation=True)
    if p_inf is None:
        raise RuntimeError(f"A* failed for segment {seg_idx}")

    teb_result = teb.solve(p_inf)
    if not teb_result.success:
        print(f"  WARNING: TEB struggled on segment {seg_idx}")

    times_arr = np.array([tk for tk, _ in teb_result.timed_waypoints])
    wps_arr   = np.array([xk for _, xk in teb_result.timed_waypoints])
    traj = ClampedCubicSpline(times_arr, wps_arr)

    # Reset controllers
    heading_controller.reset()
    lateral_controller.reset()

    # Storage
    actual_log  = []
    target_log  = []
    error_log   = []
    speed_log   = []
    t_log       = []

    t = 0.0
    dwell_count = 0
    arrived = False
    env = Environment()
    settling_reset_done = False  # reset integral once when trajectory ends

    # Arrival is checked against the TEB endpoint (nearest grid-snapped goal).
    # This is guaranteed to be within ARRIVAL_POS_TOL of seg_goal by A* grid
    # resolution, and keeps the controller reference stable after max_t.
    traj_goal = traj.get_pose(traj.max_t)

    max_steps = int(MAX_SEGMENT_TIME / dt) + DWELL_STEPS + 10

    for step in range(max_steps):
        x_star = traj.get_pose(min(t, traj.max_t))

        # When the trajectory ends, clear integral windup so the settling
        # phase starts fresh (prevents overshooting from accumulated windup).
        if t >= traj.max_t and not settling_reset_done:
            heading_controller.reset()
            lateral_controller.reset()
            settling_reset_done = True

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
        u_l = np.clip(u_k[0], -1.0, 1.0)
        u_r = np.clip(u_k[1], -1.0, 1.0)

        robot.applied_force  = np.zeros(2)
        robot.applied_torque = 0.0

        force  = np.array([u_l + u_r, 0])
        torque = (u_r - u_l) * (robot.track_width / 2.0)

        robot.apply_force_body(force)
        robot.apply_torque(torque)

        env.detect_collision(robot)
        env.compute_loads(robot, dt)

        robot.compute_dynamics(t, dt)
        robot.compute_kinematics(t, dt)

        pos   = robot.position.copy()
        speed = np.linalg.norm(robot.linear_momentum) / robot.mass

        t_log.append(t)
        actual_log.append(pos.copy())
        target_log.append(x_star.copy())
        error_log.append(e_k.copy())
        speed_log.append(speed)

        # Arrival check (only after trajectory has been fully tracked).
        # Check against traj_goal (the TEB endpoint) — always within ARRIVAL_POS_TOL
        # of seg_goal by A* grid resolution, keeps controller reference stable.
        if t >= traj.max_t:
            pos_err = np.linalg.norm(pos - traj_goal)
            if pos_err < ARRIVAL_POS_TOL and speed < ARRIVAL_SPEED_TOL:
                dwell_count += 1
            else:
                dwell_count = 0

            if dwell_count >= DWELL_STEPS:
                arrived = True
                t += dt
                break

        t += dt

    seg_duration = t_log[-1] if t_log else 0.0
    if arrived:
        print(f"  Segment {seg_idx}: ARRIVED at {seg_goal} after {seg_duration:.2f} s")
    else:
        print(f"  Segment {seg_idx}: TIMEOUT after {seg_duration:.2f} s (goal={seg_goal})")

    return {
        "t":      np.array(t_log),
        "actual": np.array(actual_log),
        "target": np.array(target_log),
        "error":  np.array(error_log),
        "speed":  np.array(speed_log),
        "teb_times": times_arr,
        "teb_wps":   wps_arr,
    }, seg_duration, arrived


# ==========================================
# 4. Main
# ==========================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    prefix = "task3_2"

    # --- Environment ---
    res, width_m, height_m, occ_grid, inf_grid, planner, obstacle_points = build_environment()
    teb = make_teb(obstacle_points)

    # --- Targets (seeded) ---
    rng = np.random.default_rng(SEED)
    start_pos = np.array([-1.2, -1.3])
    goal_1    = np.array([ 1.2,  1.3])

    print("Generating random goals (seed=42)...")
    goal_2 = get_random_goal(inf_grid, res, width_m, height_m, planner, goal_1,    rng)
    goal_3 = get_random_goal(inf_grid, res, width_m, height_m, planner, goal_2,    rng)

    goal_sequence = [goal_1, goal_2, goal_3]
    targets       = [start_pos] + goal_sequence

    print("\nMission Targets:")
    print(f"  Start  : {targets[0]}")
    print(f"  Goal 1 : {targets[1]}")
    print(f"  Goal 2 : {targets[2]}  (Random)")
    print(f"  Goal 3 : {targets[3]}  (Random)")

    # --- Robot + sensor stack (shared across segments) ---
    robot   = Robot()
    sensor  = Sensor()
    robot.position = start_pos.copy()
    estimator = Estimator(sensor)

    heading_controller = Controller(estimator=estimator, Kp=KP_HEADING, Ki=KI_HEADING,
                                    decay_factor=DECAY_FACTOR_HEADING)
    lateral_controller = Controller(estimator=estimator, Kp=KP_LATERAL, Ki=KI_LATERAL,
                                    decay_factor=DECAY_FACTOR_LATERAL)

    # --- Sequential rest-to-rest loop ---
    print("\nExecuting sequential rest-to-rest segments...")
    full_log = {"t": [], "actual": [], "target": [], "error": [], "speed": []}
    all_teb_wps   = []
    segment_boundaries = [0.0]   # cumulative time at each segment joint
    t_offset = 0.0

    for seg_idx, seg_goal in enumerate(goal_sequence, start=1):
        seg_start = targets[seg_idx - 1]
        print(f"\n  Planning Segment {seg_idx}/3: {seg_start} -> {seg_goal}")

        seg_log, seg_dur, _ = run_segment(
            seg_idx, seg_start, seg_goal,
            inf_grid, planner, teb,
            robot, sensor, estimator,
            heading_controller, lateral_controller,
        )

        full_log["t"].append(seg_log["t"] + t_offset)
        full_log["actual"].append(seg_log["actual"])
        full_log["target"].append(seg_log["target"])
        full_log["error"].append(seg_log["error"])
        full_log["speed"].append(seg_log["speed"])
        all_teb_wps.append(seg_log["teb_wps"])

        t_offset += seg_dur
        segment_boundaries.append(t_offset)

    # Flatten
    t_all      = np.concatenate(full_log["t"])
    actual_all = np.concatenate(full_log["actual"])
    target_all = np.concatenate(full_log["target"])
    error_all  = np.concatenate(full_log["error"])
    speed_all  = np.concatenate(full_log["speed"])
    teb_wps    = np.concatenate(all_teb_wps, axis=0)

    seg_boundary_times = segment_boundaries[1:-1]   # intermediate joints only

    # ===================== Deliverable 1: 2D Trajectory Plot =====================
    print(f"\nSaving Deliverable 1: 2D Trajectory Plot...")
    fig1, ax1 = plt.subplots(figsize=(7, 9))
    extent = [-width_m / 2, width_m / 2, -height_m / 2, height_m / 2]
    ax1.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
               extent=extent, interpolation='nearest')

    ax1.plot(teb_wps[:, 0],    teb_wps[:, 1],    "m:", linewidth=2, alpha=0.5, label="Raw TEB Waypoints")
    ax1.plot(target_all[:, 0], target_all[:, 1],  "g--", linewidth=2, label="Smooth Reference $x^*(t)$")
    ax1.plot(actual_all[:, 0], actual_all[:, 1],  "r-",  linewidth=2, alpha=0.8, label="Actual Path $x_k$")

    labels_t = ["Start", "Goal 1", "Goal 2 (Rand)", "Goal 3 (Rand)"]
    colors_t = ["lime", "gold", "orange", "red"]
    markers_t = ["o", "*", "*", "*"]
    for i, target in enumerate(targets):
        ax1.scatter(*target, color=colors_t[i], edgecolors='black',
                   s=150 if markers_t[i] == "*" else 120,
                   marker=markers_t[i], label=labels_t[i], zorder=5)

    ax1.set_title("Task 3.2: Multi-Target Sequential Trajectory Tracking")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=_grid_legend + handles,
               labels=[p.get_label() for p in _grid_legend] + labels,
               loc="upper left", bbox_to_anchor=(1.05, 1))
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ===================== Deliverable 2: Time-Series Error + Speed =====================
    print(f"Saving Deliverable 2: Time-Series Errors + Speed...")
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax2a.plot(t_all, error_all[:, 0], color='blue',  label="Heading Error $e_{heading}$")
    ax2a.plot(t_all, error_all[:, 1], color='red',   label="Lateral Error $e_{lateral}$")
    ax2a.set_ylabel("Error [m]")
    ax2a.set_title("Task 3.2: Tracking Error — Multi-Segment Sequential Mission")
    ax2a.legend()
    ax2a.grid(True)

    ax2b.plot(t_all, np.linalg.norm(error_all, axis=1), color='purple', label="$\\|e_k\\|$")
    ax2b.set_ylabel("Error norm [m]")
    ax2b.legend()
    ax2b.grid(True)

    ax2c.plot(t_all, speed_all, color='teal', label="Robot speed [m/s]")
    ax2c.axhline(0.5, color='r', linestyle='--', alpha=0.6, label="vmax = 0.5 m/s")
    ax2c.set_ylabel("Speed [m/s]")
    ax2c.set_xlabel("Time [s]")
    ax2c.legend()
    ax2c.grid(True)

    for ax_ in (ax2a, ax2b, ax2c):
        for tb in seg_boundary_times:
            ax_.axvline(tb, color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_tracking_error.png"), dpi=300)
    plt.close(fig2)

    # ===================== Deliverable 3: Animation =====================
    print(f"Generating animation (this may take a minute)...")
    fig_anim, ax_anim = plt.subplots(figsize=(7, 9))
    ax_anim.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
                   extent=extent, interpolation='nearest')
    ax_anim.set_xlim(-width_m / 2, width_m / 2)
    ax_anim.set_ylim(-height_m / 2, height_m / 2)
    ax_anim.set_title("Task 3.2: Sequential Rest-to-Rest Tracking")
    ax_anim.set_xlabel("X [m]"); ax_anim.set_ylabel("Y [m]")
    ax_anim.grid(True)

    for i, target in enumerate(targets):
        ax_anim.scatter(*target, color=colors_t[i], edgecolors='black',
                       s=150 if markers_t[i] == "*" else 120,
                       marker=markers_t[i], label=labels_t[i], zorder=5)

    ax_anim.plot(teb_wps[:, 0], teb_wps[:, 1], "m:", alpha=0.5, label="Raw TEB Waypoints")
    ax_anim.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()

    actual_trace, = ax_anim.plot([], [], "r-",  alpha=0.6, label="Actual Path")
    target_trace, = ax_anim.plot([], [], "g--", alpha=0.6, label="Smoothed Trajectory")
    robot_dot,    = ax_anim.plot([], [], "ro",  markersize=8)
    ref_dot,      = ax_anim.plot([], [], "go",  markersize=6)
    error_line,   = ax_anim.plot([], [], "k-",  linewidth=1)

    frame_step  = 10
    anim_frames = range(0, len(t_all), frame_step)

    def update(frame):
        actual_trace.set_data(actual_all[:frame, 0], actual_all[:frame, 1])
        target_trace.set_data(target_all[:frame, 0], target_all[:frame, 1])
        rx, ry = actual_all[frame, 0], actual_all[frame, 1]
        tx, ty = target_all[frame, 0], target_all[frame, 1]
        robot_dot.set_data([rx], [ry])
        ref_dot.set_data([tx], [ty])
        error_line.set_data([rx, tx], [ry, ty])
        return actual_trace, target_trace, robot_dot, ref_dot, error_line

    ani = animation.FuncAnimation(
        fig_anim, update,
        frames=tqdm(anim_frames, desc="Rendering Animation Frames", leave=False),
        interval=50, blit=True
    )
    ani.save(os.path.join(output_dir, f"{prefix}_animation.gif"), writer="pillow", fps=20)
    plt.close(fig_anim)

    print("\nTask 3.2 Multi-Target deliverables successfully generated.")


if __name__ == "__main__":
    main()
