# Sabrina's Game

Code for the bachelor thesis Sabrina's Game: From Unsampleability to Metaheuristic Optimisation.

Each cell of an N×N grid of integers is paired with one of its orthogonal neighbors. A pairing costs the absolute difference between the two cells, and the total objective value of a solution is the sum over all pairings. A solution is valid only if every cell is paired.

## Contents

**Algorithms**
- `random_algorithm.py`, `greedy.py` — constructive algorithms and their unstuck variants
- `hill_climbing.py`, `simulated_annealing.py`, `plant_propagation.py` — metaheuristics, each with a flip and a hybrid version

**Experiments**
- `run_random.py`, `run_greedy.py` — run the constructive algorithms on every grid
- `run_hill_climbing.py`, `run_simulated_annealing.py`, `run_plant_propagation.py` — run the metaheuristics on the constructive solutions

**Data**
- `grids/` — the grid instances, one subfolder per size (`grids/6x6/*.npy`)
- `results/` — the result files used in the thesis

## Usage

Install the requirements:

```
pip install -r requirements.txt
```

Run the constructive algorithms first, since the metaheuristics start from their solutions:

```
python run_random.py
python run_greedy.py
```

These produce `random_results.csv` and `greedy_results.csv`. Combine them into `random_greedy_results.csv`, which is the baseline file the metaheuristics read. Then:

```
python run_hill_climbing.py
python run_simulated_annealing.py
python run_plant_propagation.py
```

Each of these writes two files, one per mutation operator (for example `sa_results.csv` for the flip mutation and `sa_hybrid_results.csv` for the hybrid mutation).

Settings such as the evaluation budget, the number of runs and the number of cores are at the top of each script. The plant propagation runs on the largest grids are heavy and can take many hours; the grid sizes can be narrowed down in `SIZES_TO_RUN` and run separately.

The figures in the thesis were made from the result files in `results/`.
