import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import TargetTrajectory
from pathlib import Path

# 1. Setup Data
sigma_values = [0.0, 0.5, 1.0]
dt = 0.05
t_steps = np.arange(0, 10.0, dt)

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print("Saving files to:", output_dir.resolve())

paths = {}
for sigma in sigma_values:
    target = TargetTrajectory(delta=sigma)
    paths[sigma] = np.array([target.get_pose(t) for t in t_steps])

# 2. Static overview plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for sigma in sigma_values:
    path = paths[sigma]

    ax1.plot(path[:, 0], path[:, 1], label=f'delta = {sigma}')
    ax2.plot(t_steps, path[:, 0], label=f'$x_1^*$, delta={sigma}')
    ax2.plot(t_steps, path[:, 1], '--', label=f'$x_2^*$, delta={sigma}')

ax1.set_title("2D Target Trajectories")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.grid(True)
ax1.axis("equal")

ax2.set_title("Target Position vs Time")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Position")
ax2.legend(fontsize=8)
ax2.grid(True)

plt.tight_layout()
fig.savefig(output_dir / "target_trajectory.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Generate and save one position+speed figure for each sigma
for sigma in sigma_values:
    path = paths[sigma]

    x1 = path[:, 0]
    x2 = path[:, 1]

    dx1_dt = np.gradient(x1, dt)
    dx2_dt = np.gradient(x2, dt)
    speed = np.sqrt(dx1_dt**2 + dx2_dt**2)

    fig_ps, ax_pos = plt.subplots(figsize=(10, 6))
    ax_speed = ax_pos.twinx()

    (line1,) = ax_pos.plot(t_steps, x1, label=r"$x_{1,t}^*$")
    (line2,) = ax_pos.plot(t_steps, x2, label=r"$x_{2,t}^*$")
    (line3,) = ax_speed.plot(t_steps, speed, "--", label=r"$\|\dot{x}_t^*\|$")


    ax_pos.set_title(f'Task 3 Target Position Components and Target Speed vs Time at delta = {sigma}')
    ax_pos.set_xlabel('Time [s]')
    ax_pos.set_ylabel('Position')
    ax_speed.set_ylabel('Speed')
    ax_pos.grid(True)

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax_pos.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    fig_ps.savefig(
        output_dir / f"Task 3target_position_and_speed_delta_{sigma}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# 4. Animation
fig_anim, ax_anim = plt.subplots()
ax_anim.set_xlim(-2.5, 2.5)
ax_anim.set_ylim(-2.0, 2.0)
ax_anim.set_title("Animated Target Trajectories")
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")
ax_anim.grid(True)
ax_anim.axis("equal")

colors = ["r", "g", "b"]
dots = []
traces = []

for sigma, color in zip(sigma_values, colors):
    trace, = ax_anim.plot([], [], '-', color=color, alpha=0.7)
    dot, = ax_anim.plot([], [], 'o', color=color, label=f'delta = {sigma}')
    traces.append(trace)
    dots.append(dot)

ax_anim.legend()


def update(frame):
    artists = []

    for dot, trace, sigma in zip(dots, traces, sigma_values):
        path = paths[sigma]

        # current moving dot
        dot.set_data([path[frame, 0]], [path[frame, 1]])

        # trace of all previous positions
        trace.set_data(path[: frame + 1, 0], path[: frame + 1, 1])

        artists.extend([dot, trace])

    return tuple(artists)


ani = FuncAnimation(fig_anim, update, frames=len(t_steps), interval=50, blit=True)

# Save animation
try:
    ani.save(
        output_dir / "Task 3 target_trajectories_animation.gif",
        writer="pillow",
        dpi=120,
    )
    print(f"Saved animation to {output_dir / 'target_trajectories_animation.gif'}")
except Exception as e:
    print("Animation was displayed but not saved.")
    print(f"Reason: {e}")

plt.show()
