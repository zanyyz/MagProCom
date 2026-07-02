# MagProCom
Code for the paper: Magnetically reconfigurable living neural networks for programmable assembly and reservoir computing


# Overview

This repository contains the source code used in the manuscript describing **MagProCom**, a magnetically programmable living neural network platform for programmable biological computation.

The repository includes the computational framework for evaluating neural computation in multiple cognitive tasks, including continuous control (maze navigation), discrete decision making (Go life-and-death problems), and complete Go gameplay. The analysis pipeline also supports neural state visualization and reservoir computing-based decoding.

---

# Repository Structure

```text
MagProCom/
│
├── MagProCom_maze_Go.py      # Maze navigation and Go life-and-death tasks
├── NeuralGoSystem.py         # Full Go gameplay framework
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Main Scripts

#### `MagProCom_maze_Go.py`

This script implements the computational framework for evaluating the neural reservoir in:

* Continuous maze navigation tasks
* Go life-and-death (Tsumego) decision-making tasks
* Reservoir state processing
* Neural activity preprocessing
* Reservoir decoding and performance evaluation
* Visualization of neural dynamics and task performance

#### `NeuralGoSystem.py`

This script implements the complete Go gameplay framework based on the biological reservoir, including:

* Full-board Go games
* Neural activity decoding
* Reservoir computing
* Move prediction
* Board state updating
* Performance evaluation and visualization

---

## Requirements

Python 3.11 or later.

Required packages:

* numpy
* scipy
* scikit-learn
* matplotlib
* pandas
* h5py

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the Code

### Maze Navigation and Go Life-and-Death Tasks

```bash
python MagProCom_maze_Go.py
```

### Full Go Gameplay

```bash
python NeuralGoSystem.py
```

---

## Data

The raw experimental data supporting the findings of this study are available from the corresponding author upon reasonable request.

---

## Code Availability

The source code used in this study is provided in this repository.

---

# Reproducibility

All computational analyses were performed using Python. The software dependencies required to reproduce the reported results are listed in `requirements.txt`.

---

# Citation

If you use this code in your research, please cite the associated manuscript.

```
Manuscript under review.
```

This repository will be updated with the publication information after acceptance.

---

# License

This project is released under the MIT License.
