import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from rse.planning import RoboticPlanner, TEBPlanner, inflate_grid

GRID_CMAP = ListedColormap(['#ffffff', '#ffb380', '#404040'])
_grid_legend = [
    mpatches.Patch(color='#ffffff', ec='gray', label='Free'),
    mpatches.Patch(color='#ffb380', label='Inflated zone'),
    mpatches.Patch(color='#404040', label='Obstacle'),
]

# --- 1. Arena Construction ---
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

# --- 3. A* Planning ---
planner = RoboticPlanner(resolution=res, radius=0.2, grid_dims=(grid_w, grid_h))
start_pos, goal_pos = (-1.2, -1.3), (1.2, 1.3)

p_inf = planner.a_star(start_pos, goal_pos, inf_grid, use_inflation=True)

# --- 4. TEB Waypoint Generation ---
obstacle_points = TEBPlanner.grid_to_obstacle_points(inf_grid, planner, include_soft=True)

# --- Define TEB Parameters:---
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


teb = TEBPlanner(
    obstacle_points=obstacle_points,
    **teb_params
)

teb_result = teb.solve(p_inf)

print("TEB success:", teb_result.success)
print("TEB message:", teb_result.message)
print("Timed waypoints:")
for tk, xk in teb_result.timed_waypoints:
    print(f"t = {tk:.3f}, x = {xk}")

_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(_out_dir, exist_ok=True)

# --- 5. Deliverable Plot 1: TEB waypoints vs A* path ---
fig1 = plt.figure(figsize=(7, 9))
extent = [-width_m / 2, width_m / 2, -height_m / 2, height_m / 2]
plt.imshow(inf_grid.T, origin='lower', cmap=GRID_CMAP, vmin=0, vmax=1,
           extent=extent, interpolation='nearest')

if p_inf:
    x_astar, y_astar = zip(*p_inf)
    plt.plot(x_astar, y_astar, 'r--', linewidth=2, label='A* inflated path')

wp = teb_result.waypoints
plt.plot(wp[:, 0], wp[:, 1], 'bo-', linewidth=2, markersize=4, label='TEB timed waypoints')

plt.scatter(*start_pos, color='lime', edgecolors='black', s=100, label='Start', zorder=5)
plt.scatter(*goal_pos, color='gold', edgecolors='black', s=100, label='Goal', zorder=5)

plt.title("Task 2.1: TEB-optimized timed waypoints")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=_grid_legend + handles, labels=[p.get_label() for p in _grid_legend] + labels)
plt.grid(True)
plt.tight_layout()
fig1.savefig(os.path.join(_out_dir, "task2_1_teb_vs_astar.png"), dpi=300, bbox_inches="tight")

# --- 6. Deliverable Plot 2: Segment speed vs. time ---
t_waypoints = teb_result.t          # shape (N,) cumulative waypoint times
t_mid = 0.5 * (t_waypoints[:-1] + t_waypoints[1:])  # midpoint of each segment

fig2 = plt.figure(figsize=(8, 4))
plt.step(t_waypoints[:-1], teb_result.segment_speed, where='post', color='b', alpha=0.4, label='_nolegend_')
plt.scatter(t_mid, teb_result.segment_speed, color='b', s=40, zorder=3, label=r'Segment speed $\bar{v}_k$')
plt.axhline(0.5, color='r', linestyle='--', label=r'$\bar{v}_{max} = 0.5$ m/s')
plt.title("Task 2.1: Inter-Waypoint Speeds")
plt.xlabel("Time t [s]")
plt.ylabel("Speed [m/s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
fig2.savefig(os.path.join(_out_dir, "task2_1_segment_speed.png"), dpi=300, bbox_inches="tight")

plt.show()
