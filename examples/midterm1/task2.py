import sys
from pathlib import Path

# Connect the src folder
src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt

from rse.system._robot import Robot
from rse.environment._environment import EnvironmentProtocol
from rse.integrator._euler import EulerIntegrator
from rse.integrator._symplectic import SymplecticIntegrator
from rse.integrator._position_verlet import PositionVerletIntegrator
from rse.simulator._simulator import Simulator
from rse.action._action import ActionProtocol
from matplotlib.ticker import MaxNLocator

# --- 1. The "Unbounded" Environment ---
class UnboundedEnvironment(EnvironmentProtocol):
    """An empty plane with no walls or friction."""
    def detect_collision(self, robot: Robot) -> None:
        pass  

    def compute_loads(self, robot: Robot, time_step: float) -> None:
        pass  

# --- 2. The Spring Action ---
class SpringAction(ActionProtocol):
    """Applies a restorative force: u_l = u_r = -K * x."""
    def __init__(self, stiffness: float = 1.0):
        self.K = stiffness

    def apply(self, robot: Robot, current_time: float) -> None:
        # x_t dot e_1 is simply the x-coordinate of the position
        x_coord = robot.position[0]
        
        # Calculate wheel thrusts
        u = -self.K * x_coord
        
        # Net body force (both wheels push together)
        force_x = u + u
        
        # Apply pure forward/backward force (no torque)
        robot.apply_force_body(np.array([force_x, 0.0]))

# --- 3. Simulation Helper ---
def run_simulation(integrator_class, time_step=0.01, duration=10.0):
    robot = Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)
    robot.position = np.array([1.0, 0.0]) 
    
    environment = UnboundedEnvironment()
    integrator = integrator_class()
    action = SpringAction(stiffness=1.0)
    
    simulator = Simulator(robot, environment, integrator, time_step)
    
    times = []
    energies = []
    xs = []
    pxs = []
    
    num_steps = int(duration / time_step)
    
    for _ in range(num_steps):
        p = robot.linear_momentum
        l = robot.angular_momentum
        x = robot.position
        
        kinetic_energy = (np.linalg.norm(p)**2) / (2 * robot.mass) + (l**2) / (2 * robot.moment_of_inertia)
        potential_energy = 1.0 * np.linalg.norm(x)**2
        total_energy = kinetic_energy + potential_energy
        
        times.append(simulator.time_current)
        energies.append(total_energy)

        xs.append(x[0])
        pxs.append(p[0])
        
        simulator.step(action)
        
    return times, energies, xs, pxs
    def save_phase_energy_gif(times, energies, xs, pxs, filename, fps=30, frame_step=5):
    """
    Create a GIF with two synchronized subplots:
    - Left: phase portrait (x vs p)
    - Right: energy vs time
    """
    T = np.asarray(times, dtype=float)
    E = np.asarray(energies, dtype=float)
    X = np.asarray(xs, dtype=float)
    P = np.asarray(pxs, dtype=float)

    fig, (ax_phase, ax_energy) = plt.subplots(1, 2, figsize=(12, 5))

    # ----- Phase portrait axis setup -----
    ax_phase.set_title("Phase Portrait")
    ax_phase.set_xlabel("x (m)")
    ax_phase.set_ylabel("p_x (kg·m/s)")
    ax_phase.grid(True)

    # Set limits with padding
    x_pad = 0.05 * (X.max() - X.min() + 1e-12)
    p_pad = 0.05 * (P.max() - P.min() + 1e-12)
    ax_phase.set_xlim(X.min() - x_pad, X.max() + x_pad)
    ax_phase.set_ylim(P.min() - p_pad, P.max() + p_pad)

    # Phase portrait artists (line + moving point)
    phase_line, = ax_phase.plot([], [], color='k', linewidth=1.5)
    phase_dot,  = ax_phase.plot([], [], marker='o', color='r', markersize=6)

    # ----- Energy axis setup -----
    ax_energy.set_title("Energy vs Time")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("Total Energy (J)")
    ax_energy.grid(True)

    t_pad = 0.02 * (T[-1] - T[0] + 1e-12)
    e_pad = 0.05 * (E.max() - E.min() + 1e-12)
    ax_energy.set_xlim(T[0] - t_pad, T[-1] + t_pad)
    ax_energy.set_ylim(E.min() - e_pad, E.max() + e_pad)

    # Energy artists (line + moving point + vertical time cursor)
    energy_line, = ax_energy.plot([], [], color='b', linewidth=1.5)
    energy_dot,  = ax_energy.plot([], [], marker='o', color='r', markersize=6)
    time_cursor = ax_energy.axvline(T[0], color='k', linewidth=1.5, linestyle='--')

    # Frame indices to reduce GIF size
    frame_indices = list(range(0, len(T), frame_step))

    def init():
        phase_line.set_data([], [])
        phase_dot.set_data([], [])
        energy_line.set_data([], [])
        energy_dot.set_data([], [])
        time_cursor.set_xdata([T[0], T[0]])
        return phase_line, phase_dot, energy_line, energy_dot, time_cursor

    def update(k):
        # k is an index into arrays
        phase_line.set_data(X[:k+1], P[:k+1])
        phase_dot.set_data([X[k]], [P[k]])

        energy_line.set_data(T[:k+1], E[:k+1])
        energy_dot.set_data([T[k]], [E[k]])
        time_cursor.set_xdata([T[k], T[k]])

        return phase_line, phase_dot, energy_line, energy_dot, time_cursor

    anim = FuncAnimation(
        fig, update,
        frames=frame_indices,
        init_func=init,
        interval=1000 / fps,
        blit=False
    )

    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

def main():
    print("Running Task 2 Simulations...")

    dt1 = 0.001
    dt2 = 0.01
    dt3 = 0.1

    # dt = 0.001
    t_e1, e_e1, x_e1, p_e1 = run_simulation(EulerIntegrator, time_step=dt1)
    t_s1, e_s1, x_s1, p_s1 = run_simulation(SymplecticIntegrator, time_step=dt1)
    t_p1, e_p1, x_p1, p_p1 = run_simulation(PositionVerletIntegrator, time_step=dt1)

    # dt = 0.01
    t_e2, e_e2, x_e2, p_e2 = run_simulation(EulerIntegrator, time_step=dt2)
    t_s2, e_s2, x_s2, p_s2 = run_simulation(SymplecticIntegrator, time_step=dt2)
    t_p2, e_p2, x_p2, p_p2 = run_simulation(PositionVerletIntegrator, time_step=dt2)
    
    # dt = 0.1
    t_e3, e_e3, x_e3, p_e3 = run_simulation(EulerIntegrator, time_step=dt3)
    t_s3, e_s3, x_s3, p_s3 = run_simulation(SymplecticIntegrator, time_step=dt3)
    t_p3, e_p3, x_p3, p_p3 = run_simulation(PositionVerletIntegrator, time_step=dt3)

    # --- Phase Portraits (left-right subplots) ---
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    
    ax[0].plot(x_e1, p_e1, 'r-', linewidth=1.5, label='Explicit Euler')
    ax[0].plot(x_s1, p_s1, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax[0].plot(x_p1, p_p1, 'g-.', linewidth=1.5, label='Position Verlet')
    ax[0].set_title(f"Phase Portrait (dt = {dt1})")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("p_x (kg·m/s)")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(x_e2, p_e2, 'r-', linewidth=1.5, label='Explicit Euler')
    ax[1].plot(x_s2, p_s2, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax[1].plot(x_p2, p_p2, 'g-.', linewidth=1.5, label='Position Verlet')
    ax[1].set_title(f"Phase Portrait (dt = {dt2})")
    ax[1].set_xlabel("x (m)")
    ax[1].grid(True)
    ax[1].legend()
    
    ax[2].plot(x_e3, p_e3, 'r-', linewidth=1.5, label='Explicit Euler')
    ax[2].plot(x_s3, p_s3, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax[2].plot(x_p3, p_p3, 'g-.', linewidth=1.5, label='Position Verlet')
    ax[2].set_title(f"Phase Portrait (dt = {dt3})")
    ax[2].set_xlabel("x (m)")
    ax[2].grid(True)
    ax[2].legend()
    
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    

        
    # --- Figure 2: Energy Comparison for Two Time Steps ---
    fig2, ax2 = plt.subplots(1, 3, figsize=(14, 6), sharey=False)

    # Left: dt = 0.001
    ax2[0].plot(t_e1, e_e1, 'r-', linewidth=1.5, label='Explicit Euler')
    ax2[0].plot(t_s1, e_s1, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax2[0].plot(t_p1, e_p1, 'g-.', linewidth=1.5, label='Position Verlet')
    ax2[0].axhline(y=e_e1[0], color='k', linestyle=':', linewidth=1.5, label='Exact Energy')
    ax2[0].set_title(f"Energy vs Time (dt = {dt1})")
    ax2[0].set_xlabel("Time (s)")
    ax2[0].set_ylabel("Total Energy (J)")
    ax2[0].grid(True)
    ax2[0].legend()

    # mid: dt = 0.01
    ax2[1].plot(t_e2, e_e2, 'r-', linewidth=1.5, label='Explicit Euler')
    ax2[1].plot(t_s2, e_s2, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax2[1].plot(t_p2, e_p2, 'g-.', linewidth=1.5, label='Position Verlet')
    ax2[1].axhline(y=e_e2[0], color='k', linestyle=':', linewidth=1.5, label='Exact Energy')
    ax2[1].set_title(f"Energy vs Time (dt = {dt2})")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].grid(True)
    ax2[1].legend()
    
    # right: dt = 0.01
    ax2[2].plot(t_e3, e_e3, 'r-', linewidth=1.5, label='Explicit Euler')
    ax2[2].plot(t_s3, e_s3, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax2[2].plot(t_p3, e_p3, 'g-.', linewidth=1.5, label='Position Verlet')
    ax2[2].axhline(y=e_e2[0], color='k', linestyle=':', linewidth=1.5, label='Exact Energy')
    ax2[2].set_title(f"Energy vs Time (dt = {dt3})")
    ax2[2].set_xlabel("Time (s)")
    ax2[2].grid(True)
    ax2[2].legend()
    
    def set_ylim_from_series(ax, *series, pad=0.05):
        y = np.concatenate([np.asarray(s, dtype=float) for s in series])
        y_min = np.min(y)
        y_max = np.max(y)
        rng = y_max - y_min
        if rng == 0:
            rng = 1.0
        ax.set_ylim(y_min - pad * rng, y_max + pad * rng)

    set_ylim_from_series(ax2[0], e_e1, e_s1, e_p1)
    set_ylim_from_series(ax2[1], e_e2, e_s2, e_p2)
    set_ylim_from_series(ax2[2], e_e3, e_s3, e_p3)

    fig3, ax3 = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    # Explicit Euler
    ax3[0].plot(x_e2, p_e2, color='k', linewidth=1.5)
    ax3[0].set_title("Explicit Euler (dt = 0.01)")
    ax3[0].set_xlabel("x (m)")
    ax3[0].set_ylabel("p_x (kg·m/s)")
    ax3[0].grid(True)

    # Symplectic Euler
    ax3[1].plot(x_s2, p_s2, color='k', linewidth=1.5)
    ax3[1].set_title("Symplectic Euler (dt = 0.01)")
    ax3[1].set_xlabel("x (m)")
    ax3[1].grid(True)

    # Position Verlet
    ax3[2].plot(x_p2, p_p2, color='k', linewidth=1.5)
    ax3[2].set_title("Position Verlet (dt = 0.01)")
    ax3[2].set_xlabel("x (m)")
    ax3[2].grid(True)



    # Optional: enforce equal aspect so circles look like circles
    for a in ax3:
        a.set_aspect('equal', adjustable='box')

    import sys
from pathlib import Path

# Connect the src folder
src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt

from rse.system._robot import Robot
from rse.environment._environment import EnvironmentProtocol
from rse.integrator._euler import EulerIntegrator
from rse.integrator._symplectic import SymplecticIntegrator
from rse.integrator._position_verlet import PositionVerletIntegrator
from rse.simulator._simulator import Simulator
from rse.action._action import ActionProtocol
from matplotlib.ticker import MaxNLocator

# --- 1. The "Unbounded" Environment ---
class UnboundedEnvironment(EnvironmentProtocol):
    """An empty plane with no walls or friction."""
    def detect_collision(self, robot: Robot) -> None:
        pass  

    def compute_loads(self, robot: Robot, time_step: float) -> None:
        pass  

# --- 2. The Spring Action ---
class SpringAction(ActionProtocol):
    """Applies a restorative force: u_l = u_r = -K * x."""
    def __init__(self, stiffness: float = 1.0):
        self.K = stiffness

    def apply(self, robot: Robot, current_time: float) -> None:
        # x_t dot e_1 is simply the x-coordinate of the position
        x_coord = robot.position[0]
        
        # Calculate wheel thrusts
        u = -self.K * x_coord
        
        # Net body force (both wheels push together)
        force_x = u + u
        
        # Apply pure forward/backward force (no torque)
        robot.apply_force_body(np.array([force_x, 0.0]))

# --- 3. Simulation Helper ---
def run_simulation(integrator_class, time_step=0.01, duration=10.0):
    robot = Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)
    robot.position = np.array([1.0, 0.0]) 
    
    environment = UnboundedEnvironment()
    integrator = integrator_class()
    action = SpringAction(stiffness=1.0)
    
    simulator = Simulator(robot, environment, integrator, time_step)
    
    times = []
    energies = []
    xs = []
    pxs = []
    
    num_steps = int(duration / time_step)
    
    for _ in range(num_steps):
        p = robot.linear_momentum
        l = robot.angular_momentum
        x = robot.position
        
        kinetic_energy = (np.linalg.norm(p)**2) / (2 * robot.mass) + (l**2) / (2 * robot.moment_of_inertia)
        potential_energy = 1.0 * np.linalg.norm(x)**2
        total_energy = kinetic_energy + potential_energy
        
        times.append(simulator.time_current)
        energies.append(total_energy)

        xs.append(x[0])
        pxs.append(p[0])
        
        simulator.step(action)
        
    return times, energies, xs, pxs


def main():
    print("Running Task 2 Simulations...")

    dt1 = 0.001
    dt2 = 0.01
    dt3 = 0.1

    # dt = 0.001
    t_e1, e_e1, x_e1, p_e1 = run_simulation(EulerIntegrator, time_step=dt1)
    t_s1, e_s1, x_s1, p_s1 = run_simulation(SymplecticIntegrator, time_step=dt1)
    t_p1, e_p1, x_p1, p_p1 = run_simulation(PositionVerletIntegrator, time_step=dt1)

    # dt = 0.01
    t_e2, e_e2, x_e2, p_e2 = run_simulation(EulerIntegrator, time_step=dt2)
    t_s2, e_s2, x_s2, p_s2 = run_simulation(SymplecticIntegrator, time_step=dt2)
    t_p2, e_p2, x_p2, p_p2 = run_simulation(PositionVerletIntegrator, time_step=dt2)
    
    # dt = 0.1
    t_e3, e_e3, x_e3, p_e3 = run_simulation(EulerIntegrator, time_step=dt3)
    t_s3, e_s3, x_s3, p_s3 = run_simulation(SymplecticIntegrator, time_step=dt3)
    t_p3, e_p3, x_p3, p_p3 = run_simulation(PositionVerletIntegrator, time_step=dt3)

    # --- Phase Portraits (left-right subplots) ---
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    
    ax[0].plot(x_e1, p_e1, 'r-', linewidth=1.5, label='Explicit Euler')
    ax[0].plot(x_s1, p_s1, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax[0].plot(x_p1, p_p1, 'g-.', linewidth=1.5, label='Position Verlet')
    ax[0].set_title(f"Phase Portrait (dt = {dt1})")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("p_x (kg·m/s)")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(x_e2, p_e2, 'r-', linewidth=1.5, label='Explicit Euler')
    ax[1].plot(x_s2, p_s2, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax[1].plot(x_p2, p_p2, 'g-.', linewidth=1.5, label='Position Verlet')
    ax[1].set_title(f"Phase Portrait (dt = {dt2})")
    ax[1].set_xlabel("x (m)")
    ax[1].grid(True)
    ax[1].legend()
    
    ax[2].plot(x_e3, p_e3, 'r-', linewidth=1.5, label='Explicit Euler')
    ax[2].plot(x_s3, p_s3, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax[2].plot(x_p3, p_p3, 'g-.', linewidth=1.5, label='Position Verlet')
    ax[2].set_title(f"Phase Portrait (dt = {dt3})")
    ax[2].set_xlabel("x (m)")
    ax[2].grid(True)
    ax[2].legend()
    
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    
    # --- Figure 2: Energy Comparison for Two Time Steps ---
    fig2, ax2 = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    # Left: dt = 0.001
    ax2[0].plot(t_e1, e_e1, 'r-', linewidth=1.5, label='Explicit Euler')
    ax2[0].plot(t_s1, e_s1, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax2[0].plot(t_p1, e_p1, 'g-.', linewidth=1.5, label='Position Verlet')
    ax2[0].axhline(y=e_e1[0], color='k', linestyle=':', linewidth=1.5, label='Exact Energy')
    ax2[0].set_title(f"Energy vs Time (dt = {dt1})")
    ax2[0].set_xlabel("Time (s)")
    ax2[0].set_ylabel("Total Energy (J)")
    ax2[0].grid(True)
    ax2[0].legend()

    # mid: dt = 0.01
    ax2[1].plot(t_e2, e_e2, 'r-', linewidth=1.5, label='Explicit Euler')
    ax2[1].plot(t_s2, e_s2, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax2[1].plot(t_p2, e_p2, 'g-.', linewidth=1.5, label='Position Verlet')
    ax2[1].axhline(y=e_e2[1], color='k', linestyle=':', linewidth=1.5, label='Exact Energy')
    ax2[1].set_title(f"Energy vs Time (dt = {dt2})")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].grid(True)
    ax2[1].legend()
    
    # right: dt = 0.01
    ax2[2].plot(t_e3, e_e3, 'r-', linewidth=1.5, label='Explicit Euler')
    ax2[2].plot(t_s3, e_s3, 'b--', linewidth=1.5, label='Symplectic Euler')
    ax2[2].plot(t_p3, e_p3, 'g-.', linewidth=1.5, label='Position Verlet')
    ax2[2].axhline(y=e_e2[2], color='k', linestyle=':', linewidth=1.5, label='Exact Energy')
    ax2[2].set_title(f"Energy vs Time (dt = {dt3})")
    ax2[2].set_xlabel("Time (s)")
    ax2[2].grid(True)
    ax2[2].legend()

    fig3, ax3 = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    # Explicit Euler
    ax3[0].plot(x_e2, p_e2, color='k', linewidth=1.5)
    ax3[0].set_title("Explicit Euler (dt = 0.01)")
    ax3[0].set_xlabel("x (m)")
    ax3[0].set_ylabel("p_x (kg·m/s)")
    ax3[0].grid(True)

    # Symplectic Euler
    ax3[1].plot(x_s2, p_s2, color='k', linewidth=1.5)
    ax3[1].set_title("Symplectic Euler (dt = 0.01)")
    ax3[1].set_xlabel("x (m)")
    ax3[1].grid(True)

    # Position Verlet
    ax3[2].plot(x_p2, p_p2, color='k', linewidth=1.5)
    ax3[2].set_title("Position Verlet (dt = 0.01)")
    ax3[2].set_xlabel("x (m)")
    ax3[2].grid(True)



    # Optional: enforce equal aspect so circles look like circles
    for a in ax3:
        a.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    fig.savefig("task2_phase_portraits_all_dt.png", dpi=300)
    fig2.savefig("task2_energy_comparison_all_dt.png", dpi=300)
    fig3.savefig("task2_phase_portraits_dt001.png", dpi=300)

    save_phase_energy_gif(t_e2, e_e2, x_e2, p_e2, "phase_energy_euler_dt001.gif")
    save_phase_energy_gif(t_s2, e_s2, x_s2, p_s2, "phase_energy_symplectic_dt001.gif")
    save_phase_energy_gif(t_p2, e_p2, x_p2, p_p2, "phase_energy_verlet_dt001.gif")
    plt.show()

  

if __name__ == "__main__":
    main()