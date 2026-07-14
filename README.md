# MagProCom
## Code for the paper: *Magnetically reconfigurable living neural networks for programmable assembly and reservoir computing*

---

# Overview

This repository contains the computational framework used in the manuscript describing **MagProCom**, a magnetically programmable living neural network platform for programmable biological computation.

The framework evaluates computational capabilities of living neural networks through multiple tasks, including:

- Continuous maze navigation
- Go life-and-death (Tsumego) decision-making
- Full-board Go gameplay

The computational pipeline integrates:

- Neural activity preprocessing
- Biological reservoir computing
- Leaky Echo State Network (LeakyESN)
- Neural state-space analysis
- Neural decoding
- State clustering and assembly analysis
- Decision prediction
- Gameplay evaluation

For improved reproducibility and maintainability, the original monolithic scripts have been reorganized into two independent and modular frameworks:

1. **MagProCom_maze_Go**  
   Framework for continuous navigation and discrete Go decision-making tasks.

2. **Go_biogame**  
   Framework for complete Go gameplay using neural reservoir-based decision generation.

---

# Repository Structure

```text
MagProCom/
│
├── MagProCom_maze_Go/
│   │
│   ├── common.py
│   │   └── Common imports, global parameters, visualization settings,
│   │       color configurations, and shared utilities
│   │
│   ├── goenvironment.py
│   │   └── GoEnvironment:
│   │       Go board environment for life-and-death decision tasks
│   │
│   ├── gogamedatabaseenvironment.py
│   │   └── GoGameDatabaseEnvironment:
│   │       Database-based Go problem environment
│   │
│   ├── mazeenvironment.py
│   │   └── MazeEnvironment:
│   │       Continuous maze navigation environment
│   │
│   ├── bioreservoirdecoder.py
│   │   └── BioReservoirDecoder:
│   │       Neural preprocessing, decoding, visualization,
│   │       neural state analysis, and assembly analysis
│   │
│   ├── run_experiment.py
│   │   └── Main execution script for maze and Tsumego experiments
│   │
│   └── __init__.py
│
│
├── Go_biogame/
│   │
│   ├── common.py
│   │   └── Common imports, global parameters, color mappings,
│   │       font settings, and shared configurations
│   │
│   ├── reservoir.py
│   │   └── Leaky Echo State Network (LeakyESN) implementation
│   │
│   ├── goboard.py
│   │   └── Complete Go board implementation including:
│   │       board initialization, move validation,
│   │       state updating, and game rules
│   │
│   ├── mcts.py
│   │   └── Monte Carlo Tree Search implementation:
│   │       MCTSNode and AI opponent (WhiteAI)
│   │
│   ├── neural_system.py
│   │   └── NeuralGoSystem:
│   │       Complete neural reservoir-based Go gameplay framework
│   │
│   ├── main.py
│   │   └── Main execution script for Go gameplay experiments
│   │
│   ├── __init__.py
│   │
│   └── README.md
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Module Description

# 1. MagProCom_maze_Go

The `MagProCom_maze_Go` module implements the computational framework for evaluating neural computation during continuous control and discrete decision-making tasks.

---

## `common.py`

Provides shared components used throughout the framework:

- Python imports
- Global parameters
- Visualization configurations
- Color settings
- Utility functions

---

## `mazeenvironment.py`

Implements the continuous maze navigation environment.

Functions include:

- Maze generation
- Environment initialization
- Agent movement simulation
- Position tracking
- Navigation performance evaluation

This environment is used to evaluate continuous neural control capabilities.

---

## `goenvironment.py`

Implements the Go life-and-death (Tsumego) decision environment.

Functions include:

- Go board initialization
- Legal move checking
- Board state updating
- Decision task evaluation

---

## `gogamedatabaseenvironment.py`

Provides database-based Go task generation.

Functions include:

- Loading predefined Go problems
- Reconstructing board states
- Generating decision tasks
- Evaluating predicted moves

---

## `bioreservoirdecoder.py`

Contains the main biological reservoir analysis pipeline.

Implemented functions include:

- Neural activity preprocessing
- Reservoir state analysis
- Neural manifold visualization
- State clustering
- Neural decoding
- Prediction evaluation
- Neural assembly analysis
- Visualization of computational dynamics

This module contains the core analysis used for neural computation characterization.

---

## `run_experiment.py`

Main entry point for maze navigation and Go life-and-death experiments.

Run:

```bash
cd MagProCom_maze_Go
python run_experiment.py
```

---

# 2. Go_biogame

The `Go_biogame` module implements complete Go gameplay based on biological neural reservoir computation.

---

## `common.py`

Contains:

- Python imports
- Global parameters
- Visualization settings
- Color mapping
- Font configurations

---

## `reservoir.py`

Implements the neural reservoir model:

- Leaky Echo State Network (LeakyESN)
- Reservoir state evolution
- Temporal neural dynamics processing

---

## `goboard.py`

Implements the complete Go board system:

- Board initialization
- Stone placement
- Legal move checking
- Capture rules
- Board state updating
- Game termination conditions

---

## `mcts.py`

Implements Monte Carlo Tree Search-based AI.

Includes:

- MCTSNode
- Tree search procedure
- WhiteAI opponent

This module provides benchmark AI opponents for evaluating neural Go gameplay performance.

---

## `neural_system.py`

Implements the complete neural Go gameplay framework.

Main functions include:

- Neural activity processing
- Reservoir computation
- Move decoding
- Board-state prediction
- Neural decision generation
- Gameplay evaluation

---

## `main.py`

Main execution script for complete Go gameplay experiments.

Run:

```bash
cd Go_biogame
python main.py
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

```bash
cd MagProCom_maze_Go
python run_experiment.py
```

---

## Full Go Gameplay

```bash
cd Go_biogame
python main.py
```

---

# Data Availability

The experimental neural datasets supporting the findings of this study are available from the corresponding author upon reasonable request.

---

# Reproducibility

All computational analyses were performed using Python.

This repository provides:

- Task environments
- Neural reservoir implementations
- Neural decoding pipelines
- Go gameplay framework
- AI benchmark opponent
- Neural state visualization
- Computational performance evaluation

The modular organization enables independent reproduction of each computational task.

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
