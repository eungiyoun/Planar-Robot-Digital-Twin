import numpy as np
import itertools
import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from rse.system import Robot
from rse.controller import Controller
from rse.estimation import Estimator
from rse.sensor import Sensor
from utils import TargetTrajectory


def evaluate_gains(
    Kp: float, Ki: float, decay_factor: float, delta: float = 0.5
) -> float:
    """
    Runs the simulation WITHOUT the Environment to purely evaluate tracking performance.
    Returns the Root Mean Square Error (RMSE) between target and actual position.
    """
    dt = 0.05
    t_steps = np.arange(0, 10.0, dt)

    target = TargetTrajectory(delta=delta)
    robot = Robot()
    sensor = Sensor()

    # Apply initial conditions (Starting at bottom-left)
    robot.position = np.array([-1.2, -1.3], dtype=float)

    # Setup Estimator
    estimator = Estimator(sensor, alpha_x=0.1, alpha_theta=0.9, dt=dt)

    # Initialize Controller with the current test gains
    controller = Controller(
        estimator=estimator, Kp=Kp, Ki=Ki, decay_factor=decay_factor, dt=dt
    )

    total_squared_error = 0.0

    for t in t_steps:
        # 1. Target
        x_star = target.get_pose(t)

        # 2. Sense & Estimate
        sensor.measure(robot, dt)
        estimator.update()

        # 3. Control
        u_k = controller.control(x_star)

        # 4. Metric Calculation: True Tracking Error (Target vs Actual Position)
        true_error_magnitude = np.linalg.norm(x_star - robot.position)
        total_squared_error += true_error_magnitude**2

        # 5. Actuate
        robot.applied_force = np.array([u_k[0], 0.0])
        robot.applied_torque = u_k[1]

        robot.apply_force_body(robot.applied_force)
        robot.apply_torque(robot.applied_torque)

        # Environment step is REMOVED to purely test trajectory tracking.

        # 6. Dynamics & Kinematics
        robot.compute_dynamics(t, dt)
        robot.compute_kinematics(t, dt)

    # Calculate and return Root Mean Square Error (RMSE)
    rmse = np.sqrt(total_squared_error / len(t_steps))
    return rmse


def main():
    print("Starting Pure Tracking Parameter Sweep (Grid Search)...")

    # Define the ranges of parameters to test
    # Expand or narrow these lists based on your initial results
    Kp_values = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    Ki_values = [0.0, 0.05, 0.1, 0.5, 1.0]
    decay_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Generate all possible combinations
    combinations = list(itertools.product(Kp_values, Ki_values, decay_values))
    print(f"Total combinations to test: {len(combinations)}\n")

    results = []

    for idx, (Kp, Ki, decay) in enumerate(combinations):
        # Run simulation for this specific combination
        rmse = evaluate_gains(Kp=Kp, Ki=Ki, decay_factor=decay)
        results.append({"Kp": Kp, "Ki": Ki, "decay": decay, "rmse": rmse})

        # Print progress every 20 iterations
        if (idx + 1) % 20 == 0:
            print(f"Tested {idx + 1}/{len(combinations)} combinations...")

    # Sort results by lowest tracking error (RMSE)
    results.sort(key=lambda x: x["rmse"])

    print("\n==================================================")
    print("🌟 TOP 5 PURE TRACKING GAIN COMBINATIONS 🌟")
    print("==================================================")
    for i in range(5):
        res = results[i]
        print(
            f"Rank {i + 1}: Kp = {res['Kp']:<5} | Ki = {res['Ki']:<5} | Decay = {res['decay']:<5} | RMSE = {res['rmse']:.4f} [m]"
        )


if __name__ == "__main__":
    main()
