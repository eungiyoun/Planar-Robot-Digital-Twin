# 2026-Spring-ME498-Group-2

This project implements a full robotics software pipeline developed for ME498 (Robotics Software Engineering), Spring 2026.

It covers key components of autonomous robotic systems, including:

* Digital twin simulation
* State estimation
* Path planning (A*)
* Trajectory generation
* Feedback control

The goal is to design and evaluate algorithms that enable a robot to perceive, plan, and navigate in both static and dynamic environments.

---

## System Architecture

The system follows a standard robotics pipeline:

Sensing → Estimation → Planning → Control → Simulation (Digital Twin)

Each module is implemented independently and tested through example tasks.

---

## How to Clone

To clone the repo, simply run:
```bash
git clone https://github.com/STAGE-lab/2026-Spring-ME498-Group-2.git
```

---


## Prerequisites

To work with this repository, you need **Python**, **Git**, and **Git Large File Storage (LFS)** installed.

### Install Python

This project requires **Python 3.10 or later**.

* **Windows / macOS:** Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)
* **macOS (Homebrew):**

  ```bash
  brew install python
  ```
* **Linux:**

  ```bash
  sudo apt install python3 python3-pip
  ```

Verify installation:

```bash
python --version
```

---

## Setup (Recommended)

This project uses **uv** for dependency and environment management.

### 1. Install uv

```bash
pip install uv
```

### 2. Sync dependencies and create environment

```bash
uv sync
```

This will automatically:

* Create a virtual environment
* Install all required dependencies
* Ensure consistent versions across all users

---

## Run Examples 

To run example scripts (including Digital Twin tasks), use:

```bash
uv run python ./examples/<midterm_number>/<task_name>.py
```

### Example:

```bash
uv run python ./examples/midterm2/task4.py
```

### Example (Tasks) Directory Overview
#### Midterm 1 (Digital Twins)
* `Task1.py`: Open-Loop Dynamics Validation
* `Task2.py`: Energy Conservation Analysis
* `Task3.py`: Integrated Environmental Interaction

#### Midterm 2 (Estimation and Control)
* `Task1.py`: Sensing
* `Task2.py`: Estimation
* `Task3.py`: Target
* `Task4.py`: Control

#### Final Project 
* `Task1.py`: Path Planning (A*)
* `Task2_1.py`: Waypoint Generation with TEB
* `Task2_2.py`: Waypoint Generation with Dynamic Obstacle 
* `Task3_1.py`: Smooth Trajectory Gen. and Tracking
* `Task3_2.py`: Multi-Waypoint Trajectory Gen. and Tracking

Replace `<task_name>` and `<midterm_number>` with the desired script inside the `examples/` folder.

---

## Run Tests

To run tests:

```bash
uv run python ./tests/<test_name>.py
```

### Example:

```bash
uv run python ./tests/test_controller.py
```

---

## Repository Structure

* **Code:** `src/`, `examples/`
* **Results:** `examples/<midterm_directory>/outputs/`
* **Tests:** `tests/`
* **Large Files (LFS):**

  * Documents: `.pdf`
  * Media: `.png`, `.jpg`, `.gif`, `.mp4`, `.mov`, `.avi`

---
