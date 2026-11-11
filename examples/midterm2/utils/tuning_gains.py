import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import concurrent.futures
import multiprocessing
import itertools

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

# --- Project Imports ---
from rse.system import Robot
from rse.controller import Controller
from rse.estimation import Estimator
from rse.environment import Environment
from rse.sensor import Sensor  # <-- Reverted to YOUR exact import!
from utils import TargetTrajectory


def run_simulation_with_gains(
    heading_kp: float,
    heading_ki: float,
    lateral_kp: float,
    lateral_ki: float,
    decay_factor: float = 5.0,
    delta_sharpness: float = 0.5,
    use_environment: bool = True,
):
    """Run one simulation for a given set of controller gains and return logged error data."""
    dt = 0.001
    t_steps = np.arange(0, 10.0, dt)

    target = TargetTrajectory(delta=delta_sharpness)
    environment = Environment()
    robot = Robot()

    # Using your exact Sensor initialization
    sensor = Sensor()

    robot.position = np.array([-1.2, -1.3], dtype=float)

    estimator = Estimator(sensor)

    heading_controller = Controller(
        estimator=estimator,
        Kp=heading_kp,
        Ki=heading_ki,
        decay_factor=decay_factor,
    )
    lateral_controller = Controller(
        estimator=estimator,
        Kp=lateral_kp,
        Ki=lateral_ki,
        decay_factor=decay_factor,
    )

    e_heading_log = np.zeros(len(t_steps))
    e_lateral_log = np.zeros(len(t_steps))
    u_heading_log = np.zeros(len(t_steps))
    u_lateral_log = np.zeros(len(t_steps))
    u_l_log = np.zeros(len(t_steps))
    u_r_log = np.zeros(len(t_steps))

    for i, t in enumerate(t_steps):
        x_star = np.asarray(target.get_pose(t))

        # Your original sensor update
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

        # Saturation limits
        u_l = np.clip(u_l, -1.0, 1.0)
        u_r = np.clip(u_r, -1.0, 1.0)

        robot.applied_force = np.zeros(
            2,
        )
        robot.applied_torque = 0.0

        force = np.array([u_l + u_r, 0.0])
        torque = (u_r - u_l) * (robot.track_width / 2.0)

        robot.apply_force_body(force)
        robot.apply_torque(torque)

        if use_environment:
            environment.detect_collision(robot)
            environment.compute_loads(robot, dt)

        robot.compute_dynamics(t, dt)
        robot.compute_kinematics(t, dt)

        e_heading_log[i] = e_heading
        e_lateral_log[i] = e_lateral
        u_heading_log[i] = u_heading
        u_lateral_log[i] = u_lateral
        u_l_log[i] = u_l
        u_r_log[i] = u_r

    return {
        "t": t_steps,
        "e_heading": e_heading_log,
        "e_lateral": e_lateral_log,
        "u_heading": u_heading_log,
        "u_lateral": u_lateral_log,
        "u_l": u_l_log,
        "u_r": u_r_log,
    }


def compute_cost(log: dict) -> dict:
    """Compute weighted error metrics from one simulation log."""
    e_heading = log["e_heading"]
    e_lateral = log["e_lateral"]
    t = log["t"]

    w_heading = 3.0
    w_lateral = 3.0

    time_weights = np.linspace(1.0, 3.0, len(t))

    weighted_sq_error = time_weights * (
        w_heading * e_heading**2 + w_lateral * e_lateral**2
    )

    weighted_mse_total = np.mean(weighted_sq_error)
    weighted_rmse_total = np.sqrt(weighted_mse_total)

    total_sq_error = e_heading**2 + e_lateral**2
    mse_total = np.mean(total_sq_error)
    rmse_total = np.sqrt(mse_total)
    max_abs_error = np.max(np.sqrt(total_sq_error))
    final_error = np.sqrt(e_heading[-1] ** 2 + e_lateral[-1] ** 2)

    return {
        "weighted_mse_total": float(weighted_mse_total),
        "weighted_rmse_total": float(weighted_rmse_total),
        "mse_total": float(mse_total),
        "rmse_total": float(rmse_total),
        "max_abs_error": float(max_abs_error),
        "final_error": float(final_error),
    }


def evaluate_params(args):
    """Worker function to evaluate a single parameter combination across all deltas."""
    (
        heading_kp,
        heading_ki,
        lateral_kp,
        lateral_ki,
        decay_factor,
        delta_list,
        use_environment,
    ) = args

    combined_mse = 0.0
    combined_rmse = 0.0
    combined_max_err = 0.0
    combined_final_err = 0.0

    # Run the simulation for this controller on ALL curves
    for delta in delta_list:
        log = run_simulation_with_gains(
            heading_kp=heading_kp,
            heading_ki=heading_ki,
            lateral_kp=lateral_kp,
            lateral_ki=lateral_ki,
            decay_factor=decay_factor,
            delta_sharpness=delta,
            use_environment=use_environment,
        )

        metrics = compute_cost(log)

        combined_mse += metrics["weighted_mse_total"]
        combined_rmse += metrics["rmse_total"]
        combined_max_err = max(combined_max_err, metrics["max_abs_error"])
        combined_final_err += metrics["final_error"]

    num_deltas = len(delta_list)
    avg_mse = combined_mse / num_deltas

    return {
        "heading_kp": heading_kp,
        "heading_ki": heading_ki,
        "lateral_kp": lateral_kp,
        "lateral_ki": lateral_ki,
        "weighted_mse_total": avg_mse,
        "mse_total": avg_mse,
        "rmse_total": combined_rmse / num_deltas,
        "max_abs_error": combined_max_err,
        "final_error": combined_final_err / num_deltas,
    }


def grid_search():
    """Try multiple controller gain combinations in PARALLEL."""

    # heading_kp_list = np.linspace(1.0, 2.5, num=5)
    # heading_ki_list = np.linspace(0.5, 2.0, num=5)
    # lateral_kp_list = np.linspace(0.05, 0.30, num=5)
    # lateral_ki_list = np.linspace(0.05, 0.45, num=5)
    heading_kp_list = np.linspace(0.05, 1, num=5)
    heading_ki_list = np.linspace(0.05, 1, num=5)

    # Steering (Wheel): Needs to be gentle to avoid violent oscillations
    lateral_kp_list = np.linspace(0.05, 3, num=5)
    lateral_ki_list = np.linspace(0.05, 1, num=5)

    decay_factor = 0.0
    use_environment = False
    delta_list = [0.0, 0.5, 1.0]

    # 1. Build a list of all possible combinations
    tasks = []
    for hkp, hki, lkp, lki in itertools.product(
        heading_kp_list, heading_ki_list, lateral_kp_list, lateral_ki_list
    ):
        tasks.append((hkp, hki, lkp, lki, decay_factor, delta_list, use_environment))

    total_cases = len(tasks)
    num_cores = multiprocessing.cpu_count()

    print(f"Total Parameter Combinations: {total_cases}")
    print(f"Firing up {num_cores} CPU cores for parallel processing...\n")

    results = []

    # 2. Distribute the tasks across all available CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for i, res in enumerate(executor.map(evaluate_params, tasks), start=1):
            results.append(res)
            # Update progress on the same line
            sys.stdout.write(f"\r[{i}/{total_cases}] Combinations Evaluated...")
            sys.stdout.flush()

    print("\n\nSweep Complete!")
    # Sort so the lowest average error is at the top
    results.sort(key=lambda x: x["weighted_mse_total"])
    return results


def print_top_results(results, top_n=10):
    """Print the top-N best parameter combinations."""
    print("\nTop Robust Parameter Combinations (Averaged across Deltas):")
    print(
        f"{'Rank':<6}{'hKp':<8}{'hKi':<8}{'lKp':<8}{'lKi':<8}"
        f"{'Avg MSE':<14}{'Avg RMSE':<14}{'MaxErr':<14}{'Avg FinalErr':<14}"
    )
    print("-" * 92)

    for i, item in enumerate(results[:top_n], start=1):
        print(
            f"{i:<6}"
            f"{item['heading_kp']:<8.1f}"
            f"{item['heading_ki']:<8.1f}"
            f"{item['lateral_kp']:<8.1f}"
            f"{item['lateral_ki']:<8.1f}"
            f"{item['mse_total']:<14.6f}"
            f"{item['rmse_total']:<14.6f}"
            f"{item['max_abs_error']:<14.6f}"
            f"{item['final_error']:<14.6f}"
        )


def build_ki_heatmap(
    results, selected_heading_kp, selected_lateral_kp, cost_key="mse_total"
):
    heading_ki_vals = sorted(
        {
            r["heading_ki"]
            for r in results
            if r["heading_kp"] == selected_heading_kp
            and r["lateral_kp"] == selected_lateral_kp
        }
    )
    lateral_ki_vals = sorted(
        {
            r["lateral_ki"]
            for r in results
            if r["heading_kp"] == selected_heading_kp
            and r["lateral_kp"] == selected_lateral_kp
        }
    )

    heatmap = np.full((len(lateral_ki_vals), len(heading_ki_vals)), np.nan)

    for i, lki in enumerate(lateral_ki_vals):
        for j, hki in enumerate(heading_ki_vals):
            candidates = [
                r[cost_key]
                for r in results
                if r["heading_kp"] == selected_heading_kp
                and r["lateral_kp"] == selected_lateral_kp
                and r["heading_ki"] == hki
                and r["lateral_ki"] == lki
            ]
            if candidates:
                heatmap[i, j] = candidates[0]

    return heading_ki_vals, lateral_ki_vals, heatmap


def plot_ki_heatmap(
    results, selected_heading_kp, selected_lateral_kp, cost_key="mse_total"
):
    heading_ki_vals, lateral_ki_vals, heatmap = build_ki_heatmap(
        results,
        selected_heading_kp=selected_heading_kp,
        selected_lateral_kp=selected_lateral_kp,
        cost_key=cost_key,
    )

    if np.isnan(heatmap).all():
        print(
            f"\nCould not generate heatmap: No data found for heading_kp={selected_heading_kp} and lateral_kp={selected_lateral_kp}"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(heatmap, origin="lower", aspect="auto")

    ax.set_title(
        f"Average Robustness Heatmap ({cost_key})\n"
        f"(Fixed at Heading Kp={selected_heading_kp}, Lateral Kp={selected_lateral_kp})"
    )
    ax.set_xlabel("Heading Ki")
    ax.set_ylabel("Lateral Ki")

    ax.set_xticks(range(len(heading_ki_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in heading_ki_vals])

    ax.set_yticks(range(len(lateral_ki_vals)))
    ax.set_yticklabels([f"{v:.1f}" for v in lateral_ki_vals])

    for i in range(len(lateral_ki_vals)):
        for j in range(len(heading_ki_vals)):
            if not np.isnan(heatmap[i, j]):
                ax.text(
                    j,
                    i,
                    f"{heatmap[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if heatmap[i, j] < np.nanmean(heatmap) else "black",
                )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Averaged {cost_key}")

    plt.tight_layout()
    plt.show()


def main():
    print("Starting Grid Search Sweep...")
    results = grid_search()

    print_top_results(results, top_n=10)

    best = results[0]
    print("\n==============================================")
    print("ABSOLUTE BEST ROBUST COMBINATION FOUND:")
    print("==============================================")
    print(
        f"Heading Controller : Kp={best['heading_kp']}, Ki={best['heading_ki']}\n"
        f"Lateral Controller : Kp={best['lateral_kp']}, Ki={best['lateral_ki']}\n"
        f"Average MSE        : {best['mse_total']:.6f}"
    )

    plot_ki_heatmap(
        results,
        selected_heading_kp=best["heading_kp"],
        selected_lateral_kp=best["lateral_kp"],
        cost_key="mse_total",
    )


if __name__ == "__main__":
    main()
