import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from rse.planning import RoboticPlanner, TEBDynPlanner, inflate_grid

GRID_CMAP = ListedColormap(['#ffffff', '#ffb380', '#404040'])
_grid_legend = [
    mpatches.Patch(color='#ffffff', ec='gray', label='Free'),
    mpatches.Patch(color='#ffb380', label='Inflated zone'),
    mpatches.Patch(color='#404040', label='Obstacle'),
]

def dynamic_obstacle_position(t: float) -> np.ndarray:
    return np.array([0.5 * np.sin(np.pi * t), 0.0], dtype=float)


# --- 1. Environment setup ---
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

# --- 2. Obstacle Inflation ---
inf_grid = inflate_grid(occ_grid, 0.2, res)

def add_dynamic_obstacle_to_grid(
    base_grid: np.ndarray,
    planner,
    obs_pos: np.ndarray,
    obs_radius: float,
    value: float = 1.0,
) -> np.ndarray:
    
    grid_new = base_grid.copy()

    for i in range(grid_new.shape[0]):
        for j in range(grid_new.shape[1]):
            cell_center = np.asarray(planner.gtw((i, j)), dtype=float)
            if np.linalg.norm(cell_center - obs_pos) <= obs_radius:
                grid_new[i, j] = max(grid_new[i, j], value)

    return grid_new   

# --- 2. A* initial guidance ---
planner = RoboticPlanner(resolution=res, radius=0.2, grid_dims=(grid_w, grid_h))
start_pos = np.array([-1.25, -1.35], dtype=float)
goal_pos = np.array([1.25, 1.35], dtype=float)

p_inf = planner.a_star(tuple(start_pos), tuple(goal_pos), inf_grid, use_inflation=True)


# --- 3. Dynamic TEB planner ---
obstacle_points = TEBDynPlanner.grid_to_obstacle_points(inf_grid, planner, include_soft=True)

# dynamic TEB parameters
tebDyn_params = {
    "vmax": 0.5,
    "dmin_static": 0.25,
    "dmin_dynamic": 0.65,
    "n_waypoints": 20, 
    "gamma_time": 4.1, # total time optimize
    "gamma_smooth": 7.0, # segment length optimize big = tighter
    "gamma_obs": 41.0, # bigger is to avoid obstacle
    "gamma_vel": 80.0, # big is don't go over speed limit
    "gamma_bc": 1.0,
    "dt_min": 0.1,
    "dt_max": 0.5,
    "horizon_time" : 1.0,
    "gamma_horizon" : 1.50,
    "robot_radius" : 0.2,
    "dynamic_obstacle_radius" : 0.4,
}

teb = TEBDynPlanner(
    obstacle_points=obstacle_points,
    dynamic_obstacle_fn=dynamic_obstacle_position,
    **tebDyn_params
)

# --- 4. Receding-horizon loop ---
robot_positions = [start_pos.copy()]
robot_times = [0.0]
dynamic_clearance_history = []
static_clearance_history = []
planned_segments = []

current_pos = start_pos.copy()
current_time = 0.0

max_cycles = 40
goal_tol = 0.15
exec_steps = 1

for cycle in range(max_cycles):
    # dynamic obstacle at current cycle time
    obs_now = dynamic_obstacle_position(current_time)

    # build temporary grid for this cycle
    grid_for_astar = add_dynamic_obstacle_to_grid(
        base_grid=inf_grid,
        planner=planner,
        obs_pos=obs_now,
        obs_radius=teb.dynamic_obstacle_radius + teb.robot_radius,
        value=1.0,
    )

    # refresh A* guidance from current position to goal on temporary grid
    p_guidance = planner.a_star(
        tuple(current_pos),
        tuple(goal_pos),
        grid_for_astar,
        use_inflation=True
    )
    if p_guidance is None:
        print("A* failed during replanning.")
        break

    teb_res, executed_waypoints, executed_dt = teb.solve_receding_horizon(
        p_guidance,
        t0=current_time,
        exec_steps=exec_steps
    )

    planned_segments.append(teb_res.waypoints.copy())

    for k in range(1, len(executed_waypoints)):
        current_pos = executed_waypoints[k].copy()
        current_time += executed_dt[k - 1]

        robot_positions.append(current_pos.copy())
        robot_times.append(current_time)

        d_dynamic = teb._dynamic_obstacle_distance(
            np.array([current_pos]),
            np.array([current_time])
        )[0]
        dynamic_clearance_history.append((current_time, d_dynamic))

        d_static = teb._nearest_obstacle_distance(np.array([current_pos]))[0]
        static_clearance_history.append((current_time, d_static))

    if np.linalg.norm(current_pos - goal_pos) < goal_tol:
        print("Goal reached.")
        break

current_pos = executed_waypoints[k].copy()
current_time += executed_dt[k - 1]
robot_positions = np.array(robot_positions)
robot_times = np.array(robot_times)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
# --- 5. Plot 1: robot path and dynamic obstacle snapshots ---
fig1 = plt.figure(figsize=(7, 9))
extent = [-width_m / 2, width_m / 2, -height_m / 2, height_m / 2]
plt.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
           extent=extent, interpolation='nearest')

if p_inf:
    x_astar, y_astar = zip(*p_inf)
    plt.plot(x_astar, y_astar, 'r--', alpha=0.5, label='Initial A* path')

for idx, seg in enumerate(planned_segments):
    plt.plot(seg[:, 0], seg[:, 1], alpha=0.3, label='TEB horizon' if idx == 0 else None)

plt.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=2, label='Executed path')
plt.scatter(*start_pos, color='lime', edgecolors='black', s=100, label='Start', zorder=5)
plt.scatter(*goal_pos, color='gold', edgecolors='black', s=100, label='Goal', zorder=5)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=_grid_legend + handles, labels=[p.get_label() for p in _grid_legend] + labels)

snapshot_times = np.linspace(0, max(current_time, 1e-3), 6)
for i, ts in enumerate(snapshot_times):
    obs = dynamic_obstacle_position(ts)
    circle = plt.Circle(obs, 0.4, color='orange', alpha=0.2)
    plt.gca().add_patch(circle)
    if i == 0:
        plt.scatter(obs[0], obs[1], color='orange', label='Dynamic obstacle')

plt.title("Task 2.2: Dynamic TEB with receding horizon")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(True)
fig1.savefig(os.path.join(output_dir, "task 2_2_path_plot.png"), dpi=300, bbox_inches="tight")

# --- 6. Plot 2: clearance over time ---
if len(dynamic_clearance_history) > 0 and len(static_clearance_history) > 0:
    dynamic_clearance_history = np.array(dynamic_clearance_history)
    static_clearance_history = np.array(static_clearance_history)
    static_clearance_history[:, 1] += 0.2 # robot boudary to non-inflated grid

    fig2 = plt.figure(figsize=(8, 4))

    plt.plot(
        dynamic_clearance_history[:, 0],
        dynamic_clearance_history[:, 1],
        'b-o',
        label='Dynamic obstacle clearance'
    )
    plt.plot(
        static_clearance_history[:, 0],
        static_clearance_history[:, 1],
        'm-s',
        label='Static obstacle clearance'
    )

    plt.axhline(0.0, color='b', linestyle='--', label='Dynamic dmin')
    plt.axhline(0.0, color='m', linestyle=':', label='Static dmin')

    plt.title("Task 2.2: Net clearance distance over time")
    plt.xlabel("Time [s]")
    plt.ylabel("Clearance [m]")
    plt.grid(True)
    plt.legend()
    fig2.savefig(os.path.join(output_dir, "task2_2_clearance_plot.png"), dpi=300, bbox_inches="tight")

# --- 7. Animation ---
display_fps = 20
t_end = robot_times[-1]
times_for_anim = np.arange(0.0, t_end + 1.0 / display_fps, 1.0 / display_fps)

robot_x = np.interp(times_for_anim, robot_times, robot_positions[:, 0])
robot_y = np.interp(times_for_anim, robot_times, robot_positions[:, 1])
robot_positions_anim = np.column_stack((robot_x, robot_y))

fig_anim, ax_anim = plt.subplots(figsize=(7, 9))
ax_anim.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
               extent=extent, interpolation='nearest')
ax_anim.set_xlim(-width_m / 2, width_m / 2)
ax_anim.set_ylim(-height_m / 2, height_m / 2)
ax_anim.set_aspect('equal')
ax_anim.set_title("Dynamic obstacle avoidance animation")
ax_anim.set_xlabel("X [m]")
ax_anim.set_ylabel("Y [m]")
ax_anim.grid(True)

robot_dot, = ax_anim.plot([], [], 'bo', label='Robot')
robot_trace, = ax_anim.plot([], [], 'b-', alpha=0.6)
robot_circle = plt.Circle((0, 0), 0.2, color='blue', alpha=0.15)
ax_anim.add_patch(robot_circle)

obs_dot, = ax_anim.plot([], [], 'ro', label='Dynamic obstacle')
obs_circle = plt.Circle((0, 0), 0.4, color='orange', alpha=0.2)
ax_anim.add_patch(obs_circle)

ax_anim.legend()

display_fps = 20
t_end = robot_times[-1]
times_for_anim = np.arange(0.0, t_end + 1.0 / display_fps, 1.0 / display_fps)

robot_x = np.interp(times_for_anim, robot_times, robot_positions[:, 0])
robot_y = np.interp(times_for_anim, robot_times, robot_positions[:, 1])
robot_positions_anim = np.column_stack((robot_x, robot_y))

def update(frame):
    x = robot_positions_anim[frame, 0]
    y = robot_positions_anim[frame, 1]

    robot_dot.set_data([x], [y])
    robot_trace.set_data(robot_positions_anim[:frame + 1, 0],
                         robot_positions_anim[:frame + 1, 1])
    robot_circle.center = (x, y)

    t = times_for_anim[frame]
    obs = dynamic_obstacle_position(t)
    obs_dot.set_data([obs[0]], [obs[1]])
    obs_circle.center = (obs[0], obs[1])

    return robot_dot, robot_trace, robot_circle, obs_dot, obs_circle

ani = FuncAnimation(
    fig_anim,
    update,
    frames=len(times_for_anim),
    interval=1000 / display_fps,
    blit=True
)

ani.save(
    os.path.join(output_dir, "task2_2_dynamic_teb_real_time.gif"),
    writer="pillow",
    fps=display_fps
)