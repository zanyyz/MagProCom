# MagProCom
## Code for the paper: *Magnetically reconfigurable living neural networks for programmable assembly and reservoir computing*

---

# Overview

This repository contains the computational framework used in the manuscript describing **MagProCom**, a magnetically programmable living neural network platform for programmable biological computation.

The framework evaluates neural computation capabilities of living neural networks through multiple computational tasks, including:

- Continuous maze navigation
- Go life-and-death (Tsumego) decision-making
- Reservoir-based neural state analysis and decoding

The computational pipeline integrates:

- Neural activity preprocessing
- Biological reservoir computing
- Neural state-space analysis
- State clustering and assembly analysis
- Neural decoding
- Task performance evaluation
- Visualization of neural dynamics

For improved reproducibility and maintainability, the original monolithic script has been reorganized into a modular framework.

---

# Repository Structure

```text
MagProCom/
│
├── MagProCom_maze_Go/
│   │
│   ├── common.py
│   │   └── Common configurations, imports, plotting parameters, and shared utilities
│   │
│   ├── goenvironment.py
│   │   └── GoEnvironment: Go board environment for life-and-death tasks
│   │
│   ├── gogamedatabaseenvironment.py
│   │   └── GoGameDatabaseEnvironment: Database-based Go game environment
│   │
│   ├── mazeenvironment.py
│   │   └── MazeEnvironment: Continuous maze navigation environment
│   │
│   ├── bioreservoirdecoder.py
│   │   └── BioReservoirDecoder: Neural preprocessing, reservoir decoding,
│   │       visualization, state analysis, and assembly analysis
│   │
│   ├── run_experiment.py
│   │   └── Main execution script for experiments
│   │
│   └── __init__.py
│
├── Go_biogame/
│   │
│   └── (Full Go gameplay framework)
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Module Description

## 1. MagProCom_maze_Go

This module implements the core computational framework for evaluating neural computation in continuous control and discrete decision-making tasks.

The module contains the following components:

---

### `common.py`

Provides shared components used throughout the framework, including:

- Required Python imports
- Global parameters
- Visualization settings
- Color configurations
- Common utility functions

---

### `mazeenvironment.py`

Implements the continuous maze navigation environment.

Main functions include:

- Maze generation and initialization
- Agent movement simulation
- Position tracking
- Navigation performance evaluation

This environment is used to evaluate continuous neural control capabilities of the biological reservoir.

---

### `goenvironment.py`

Implements the Go life-and-death (Tsumego) task environment.

Functions include:

- Go board initialization
- Legal move checking
- Board state updating
- Task-specific evaluation

This environment supports neural-state-based decision decoding.

---

### `gogamedatabaseenvironment.py`

Provides a database-based Go environment for loading and evaluating predefined Go problems.

Functions include:

- Loading Go game records
- Reconstructing board states
- Generating decision-making tasks
- Evaluating predicted moves

---

### `bioreservoirdecoder.py`

Contains the main computational analysis framework.

The module implements:

- Neural activity preprocessing
- Reservoir state construction
- Neural state transformation
- Dimensionality reduction
- State clustering
- Neural manifold visualization
- Decoder training and prediction
- Assembly dynamics analysis
- Performance evaluation

This file contains the primary analysis pipeline used for the reported computational results.

---

### `run_experiment.py`

Main entry point for running experiments.

Example:

```bash
cd MagProCom_maze_Go
python run_experiment.py
```

---

# 2. Go_biogame

The `Go_biogame` module implements the complete Go gameplay framework based on biological neural reservoir computation.

The framework includes:

- Full-board Go gameplay
- Neural activity decoding
- Reservoir-based move generation
- Board-state updating
- Gameplay evaluation against AI opponents

Usage:

```bash
cd Go_biogame
python run.py
```

---

# Requirements

Python version:

```
Python >= 3.11
```

Required packages:

```
numpy
scipy
scikit-learn
matplotlib
pandas
h5py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Code

## Maze Navigation and Go Life-and-Death Tasks

Run:

```bash
cd MagProCom_maze_Go
python run_experiment.py
```

The program performs:

1. Neural data loading and preprocessing
2. Reservoir state analysis
3. Neural decoding
4. Task execution
5. Visualization generation
6. Performance evaluation

---

## Complete Go Gameplay

Run:

```bash
cd Go_biogame
python run.py
```

---

# Data Availability

The experimental neural datasets supporting the findings of this study are available from the corresponding author upon reasonable request.

---

# Reproducibility

All computational analyses were performed using Python.

This repository provides:

- Modular task environments
- Neural reservoir analysis pipeline
- Decoding algorithms
- Neural state visualization
- Assembly and manifold analysis
- Performance evaluation procedures

The modular structure allows independent reproduction of each computational task.

---

# Code Availability

The source code used in this study is publicly available in this repository.

The repository will be updated with the final publication information after acceptance.

---

# Citation

If you use this code in your research, please cite:

```
MagProCom: Magnetically reconfigurable living neural networks for programmable assembly and reservoir computing.

Manuscript under review.
```

---

# License

This project is released under the MIT License.
