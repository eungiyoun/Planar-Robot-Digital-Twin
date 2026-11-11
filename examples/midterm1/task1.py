import sys
from pathlib import Path

# Connect the src folder
src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from rse.system._robot import Robot
from rse.environment._environment import EnvironmentProtocol
from rse.integrator._euler import EulerIntegrator
from rse.integrator._position_verlet import PositionVerletIntegrator
from rse.integrator._symplectic import SymplecticIntegrator
from rse.simulator._simulator import Simulator
from rse.action._action import ActionProtocol


# --- 1. The "Unbounded" Environment ---
class UnboundedEnvironment(EnvironmentProtocol):
    """An empty environment to satisfy the 'unbounded plane' requirement."""
    def detect_collision(self, robot: Robot) -> None:
        pass  # No walls!

    def compute_loads(self, robot: Robot, time_step: float) -> None:
        pass  # No friction or contact forces!

# --- 2. The Four-Stage Action ---
class Task1Action(ActionProtocol):
    """Implements the specific 4-stage timing from the assignment."""
    def __init__(self, track_width: float):
        self.w = track_width

    def apply(self, robot: Robot, current_time: float) -> None:
        # Determine wheel thrusts based on the current time
        if 0.0 <= current_time < 1.0:
            ul, ur = 1.0, 1.0    # 1s forward
        elif 1.0 <= current_time < 5.0:
            ul, ur = 0.0, 0.0    # 4s coast
        elif 5.0 <= current_time < 6.0:
            ul, ur = 1.0, -1.0   # 1s rotation
        else:
            ul, ur = 0.0, 0.0    # final 4s coast

        # Convert wheel thrusts to body forces and torques
        force_x = ul + ur
        # Torque is (Right Thrust - Left Thrust) * (track_width / 2)
        torque = (ur - ul) * (self.w / 2.0)
        
        robot.apply_force_body(np.array([force_x, 0.0]))
        robot.apply_torque(torque)

# --- 3. Analytical Solutions ---
def get_analytical_v(t: float) -> float:
    """Calculates exact linear speed based on F=ma."""
    if t < 1.0: return 1.0 * t  # a = 2N / 2kg = 1 m/s^2
    return 1.0                  # Coasts at 1 m/s

def get_analytical_w(t: float) -> float:
    """Calculates exact angular speed based on tau=J*alpha."""
    if t < 5.0: return 0.0
    if t < 6.0: return -3.0 * (t - 5.0) # alpha = -0.15Nm / 0.05kgm^2 = -3 rad/s^2
    return -3.0                         # Coasts at -3 rad/s



def main():
    # --- Initialize Engine ---
    robot = Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)
    environment = UnboundedEnvironment() # Use the empty plane
    integrator = PositionVerletIntegrator()
    action = Task1Action(track_width=robot.track_width)
    
    time_step = 0.001
    simulator = Simulator(robot, environment, integrator, time_step)
    
    # --- Simulation Setup ---
    time_horizon = 10.0
    num_steps = int(time_horizon / time_step)
    
    # Data logging arrays
    times = []
    num_v, num_w = [], []
    ana_v, ana_w = [], []
    history_x, history_y = [], []
    history_R = []

    print("Running Task 1 Open-Loop Simulation...")
    
    # --- Simulation Loop ---
    for _ in range(num_steps):
        t = simulator.time_current
        
        # Log Numerical Data (v = ||p||/m, w = l/J)
        v_num = np.linalg.norm(robot.linear_momentum) / robot.mass
        w_num = robot.angular_momentum / robot.moment_of_inertia
        
        times.append(t)
        num_v.append(v_num)
        num_w.append(w_num)
        ana_v.append(get_analytical_v(t))
        ana_w.append(get_analytical_w(t))
        history_x.append(robot.position[0])
        history_y.append(robot.position[1])
        history_R.append(np.array(robot.orientation, dtype=float))
        
        simulator.step(action)

    # --- Plotting Results ---
    # --- Plot 1: Workspace Trajectory ---
    plt.figure(figsize=(6, 6))
    plt.plot(history_x, history_y, 'b-', linewidth=2, label='Robot Path')
    plt.plot(history_x[0], history_y[0], 'go', markersize=8, label='Start')
    plt.plot(history_x[-1], history_y[-1], 'rx', markersize=8, label='End')
    
    # Draw body-fixed frames every N steps
    frame_stride = 1000
    axis_length = 0.2

    arrow_len = 0.4    # arrow length
    eps = 1e-12

    for i in range(0, len(history_x), frame_stride):
        x = history_x[i]
        y = history_y[i]
        R = history_R[i]  # 2x2

        # position vector
        d1 = R[:, 0].astype(float)  # body x-axis direction in world
        d2 = R[:, 1].astype(float)  # body y-axis direction in world

        # normal quiver length
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 < eps or n2 < eps:
            continue

        d1 = d1 / n1
        d2 = d2 / n2

        bx, by = arrow_len * d1[0], arrow_len * d1[1]
        lx, ly = arrow_len * d2[0], arrow_len * d2[1]

        circle = Circle((x, y), 0.2, fill=False, zorder=2)
        plt.gca().add_patch(circle)

        plt.quiver(
            x, y, bx, by,
            angles='xy', scale_units='xy', scale=1,   
            width=0.003, headwidth=2.5, headlength=3.0, headaxislength=3,
            color='k', zorder=3
        )
        plt.quiver(
            x, y, lx, ly,
            angles='xy', scale_units='xy', scale=1,
            width=0.003, headwidth=2.5, headlength=3.0, headaxislength=3,
            color='k', zorder=3
        )
    plt.title("Spatial Trajectory (Workspace Map)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig("task1_trajectory.png", dpi=300) # Uncomment to save image

    # --- Plot 2: Linear Speed Validation ---
    plt.figure(figsize=(8, 5))
    plt.plot(times, num_v, 'b-', linewidth=4, alpha=0.5, label=f'Numerical ({integrator.__class__.__name__})')
    plt.plot(times, ana_v, 'r--', linewidth=2, label='Analytical')
    plt.title("Linear Speed $v_t$ vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.legend()
    plt.savefig("task1_linear_speed.png", dpi=300) # Uncomment to save image

    # --- Plot 3: Angular Velocity Validation ---
    plt.figure(figsize=(8, 5))
    plt.plot(times, num_w, 'b-', linewidth=4, alpha=0.5, label=f'Numerical ({integrator.__class__.__name__})')
    plt.plot(times, ana_w, 'r--', linewidth=2, label='Analytical')
    plt.title(r"Angular Velocity $\omega_t$ vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.grid(True)
    plt.legend()
    plt.savefig("task1_angular_velocity.png", dpi=300) # Uncomment to save image



    # --- Build wheel force profiles matching your policy ---
    T = np.array(times, dtype=float)
    t_min = T[0] - 1
    t_max = T[-1] + 0.5
    V = np.array(num_v, dtype=float)   # linear speed vs time
    W = np.array(num_w, dtype=float)   # angular velocity vs time
    ul = np.zeros_like(T)
    ur = np.zeros_like(T)

    ul[(T >= 0.0) & (T < 1.0)] = 1.0
    ur[(T >= 0.0) & (T < 1.0)] = 1.0

    ul[(T >= 5.0) & (T < 6.0)] = 1.0
    ur[(T >= 5.0) & (T < 6.0)] = -1.0

    # --- Animation with force subplot ---
    fig_anim, (ax_map, ax_force, ax_v, ax_w) = plt.subplots(
        4, 1, figsize=(12, 11),
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        sharex=False
    )
    # ===== Linear speed profile =====
    ax_v.plot(T, V, color='b', linewidth=2, label='v(t)')
    ax_v.set_xlim(t_min, t_max)
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Speed (m/s)")
    ax_v.set_title("Linear Speed")
    ax_v.grid(True)
    ax_v.legend(loc='upper right')

    v_time_line = ax_v.axvline(T[0], color='k', linewidth=2)
    v_marker, = ax_v.plot([T[0]], [V[0]], marker='o', color='k', markersize=6)

    # ===== Angular velocity profile =====
    ax_w.plot(T, W, color='m', linewidth=2, label='ω(t)')
    ax_w.set_xlim(t_min, t_max)
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Angular vel (rad/s)")
    ax_w.set_title("Angular Velocity")
    ax_w.grid(True)
    ax_w.legend(loc='upper right')

    w_time_line = ax_w.axvline(T[0], color='k', linewidth=2)
    w_marker, = ax_w.plot([T[0]], [W[0]], marker='o', color='k', markersize=6)

    # ===== Top: workspace animation =====
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xlim(min(history_x) - 1, max(history_x) + 1)
    ax_map.set_ylim(min(history_y) - 1, max(history_y) + 1)
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title("Robot Motion")
    ax_map.grid(True)

    line_path, = ax_map.plot([], [], 'b-', linewidth=2)

    robot_circle = Circle((history_x[0], history_y[0]), 0.2, fill=False, zorder=2)
    ax_map.add_patch(robot_circle)

    # Longer arrows, smaller arrowhead/shaft

    eps = 1e-12

    arrow_x = ax_map.quiver(
        0, 0, 0, 0,
        angles='xy', scale_units='xy', scale=1,   
        color='k', zorder=3,
        width=0.003,        
        headwidth=2.0,       
        headlength=3.0,
        headaxislength=2.5
    )
    arrow_y = ax_map.quiver(
        0, 0, 0, 0,
        angles='xy', scale_units='xy', scale=1,
        color='k', zorder=3,
        width=0.003,
        headwidth=2.0,
        headlength=3.0,
        headaxislength=2.5
    )

    arrow_len = 0.4
    
    
    eps = 1e-12

    # ===== Bottom: force profile =====
    ax_force.plot(T, ul, color='g',linewidth=2, label='u_l')
    ax_force.plot(T, ur, color='r', linewidth=2, linestyle='--', label='u_r')
    ax_force.set_xlim(t_min, t_max)
    ax_force.set_ylim(-1.2, 1.2)
    ax_force.set_xlabel("Time (s)")
    ax_force.set_ylabel("Wheel force")
    ax_force.set_title("Wheel Force Profile")
    ax_force.grid(True)
    ax_force.legend(loc='upper right')

    # moving time indicator
    time_line = ax_force.axvline(T[0], color='k', linewidth=2)

    # moving markers at current ul/ur
    ul_marker, = ax_force.plot([T[0]], [ul[0]], marker='o', color='k', markersize=6)
    ur_marker, = ax_force.plot([T[0]], [ur[0]], marker='o', color='0.5', markersize=6)

    # optional time text
    time_text = ax_map.text(
        0.02, 0.98, "",
        transform=ax_map.transAxes,
        va='top', ha='left'
    )
    # Half track width (distance from center to each wheel)
    # Half track width (distance from center to each wheel)
    track_half = robot.track_width / 2.0

    # Scale factor to visualize wheel forces
    force_scale = 0.25   # adjust if arrows look too small or too large

    # Initialize wheel force arrows (left = red, right = blue)
    arrow_ul = ax_map.quiver(
        0, 0, 0, 0,
        angles='xy', scale_units='xy', scale=1,
        color='g', width=0.004,
        headwidth=2.0,
        headlength=3.0,
        headaxislength=2.5
    )

    arrow_ur = ax_map.quiver(
        0, 0, 0, 0,
        angles='xy', scale_units='xy', scale=1,
        color='r', width=0.004,
        headwidth=2.0,
        headlength=3.0,
        headaxislength=2.5
    )

    def update(frame_idx):
        # frame_idx is an index into the logged arrays
        line_path.set_data(history_x[:frame_idx + 1], history_y[:frame_idx + 1])

        x = history_x[frame_idx]
        y = history_y[frame_idx]
        R = history_R[frame_idx]

        robot_circle.center = (x, y)

        d1 = R[:, 0].astype(float)
        d2 = R[:, 1].astype(float)
        # Wheel positions in world frame
        left_pos  = np.array([x, y]) + track_half * d2
        right_pos = np.array([x, y]) - track_half * d2

        # Wheel forces at current frame
        ul_now = ul[frame_idx]
        ur_now = ur[frame_idx]

        # Force vectors point along body x-axis (d1)
        ul_vec = force_scale * ul_now * d1
        ur_vec = force_scale * ur_now * d1

        # Update left wheel arrow
        arrow_ul.set_offsets(left_pos)
        arrow_ul.set_UVC(ul_vec[0], ul_vec[1])

        # Update right wheel arrow
        arrow_ur.set_offsets(right_pos)
        arrow_ur.set_UVC(ur_vec[0], ur_vec[1])

        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 > eps:
            d1 = d1 / n1
        if n2 > eps:
            d2 = d2 / n2

        bx, by = arrow_len * d1[0], arrow_len * d1[1]
        lx, ly = arrow_len * d2[0], arrow_len * d2[1]

        arrow_x.set_offsets([x, y])
        arrow_x.set_UVC(bx, by)

        arrow_y.set_offsets([x, y])
        arrow_y.set_UVC(lx, ly)

        t_now = T[frame_idx]
        time_line.set_xdata([t_now, t_now])
        # Synchronize linear speed and angular velocity indicators
        v_time_line.set_xdata([t_now, t_now])
        w_time_line.set_xdata([t_now, t_now])

        v_marker.set_data([t_now], [V[frame_idx]])
        w_marker.set_data([t_now], [W[frame_idx]])

        ul_marker.set_data([t_now], [ul[frame_idx]])
        ur_marker.set_data([t_now], [ur[frame_idx]])

        time_text.set_text(f"t = {t_now:.2f} s")

        return (
            line_path, robot_circle, arrow_x, arrow_y,
            arrow_ul, arrow_ur,
            time_line, ul_marker, ur_marker,
            v_time_line, v_marker,
            w_time_line, w_marker,
            time_text
        )

    # Reduce frames for smaller gif
    frame_step = 20
    frame_indices = list(range(0, len(T), frame_step))

    anim = FuncAnimation(fig_anim, update, frames=frame_indices, interval=20, blit=False)

    anim.save("task1_demo_with_forces.gif", writer=PillowWriter(fps=30))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()