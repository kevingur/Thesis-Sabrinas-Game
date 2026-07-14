from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os
import ast
import time
import random

from plant_propagation import plant_propagation, plant_propagation_hybrid  

BASELINE_CSV   = "random_greedy_results.csv"                     
GRIDS_ROOT     = "grids"                                        
SOURCES_TO_RUN = ["random", "random_unstuck", "greedy", "greedy_unstuck"]
SIZES_TO_RUN   = ["6x6", "8x8", "12x12", "16x16", "18x18"]
MAX_EVALS      = 10000
N_JOBS         = 5
SEED_BASE      = 1000

PPA_N            = 20    
PPA_MAX_RUNNERS  = 5     
PPA_INIT_FLIPS   = 5    
PPA_N_RIP        = 4    
PPA_P_RIPREPAIR  = 0.3   

INIT_OBJ_COL   = "objective_value"

VARIANTS = [
    ("flip",   plant_propagation,        "ppa_results.csv",        False),
    ("hybrid", plant_propagation_hybrid, "ppa_hybrid_results.csv", True),
]

_grid_cache = {}
def get_grid(size, grid_id):
    key = (size, grid_id)
    if key not in _grid_cache:
        folder = os.path.join(GRIDS_ROOT, size)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        _grid_cache[key] = np.load(os.path.join(folder, files[grid_id]))
    return _grid_cache[key]

def _run_one(idx, variant_fn, is_hybrid, source, size, grid_id,
             baseline_run_id, couplings_str, initial_obj):
    random.seed(SEED_BASE + idx)
    grid = get_grid(size, grid_id)
    couplings = ast.literal_eval(couplings_str)

    if is_hybrid:
        (init, best, evals, runtime, final_coups) = variant_fn(
            couplings, initial_obj, grid,
            N=PPA_N, max_evaluations=MAX_EVALS,
            n_max_runners=PPA_MAX_RUNNERS, init_flips=PPA_INIT_FLIPS,
            n=PPA_N_RIP, p_riprepair=PPA_P_RIPREPAIR)
    else:
        (init, best, evals, runtime, final_coups) = variant_fn(
            couplings, initial_obj, grid,
            N=PPA_N, max_evaluations=MAX_EVALS,
            n_max_runners=PPA_MAX_RUNNERS, init_flips=PPA_INIT_FLIPS)

    return {"run_id": baseline_run_id, "source_algorithm": source,
            "grid_size": size, "grid_id": grid_id,
            "initial_total_objective_value": int(init), "improved_total_objective_value": int(best),
            "evaluations": evals, "runtime": runtime,
            "couplings": str(final_coups)}

def run_variant(tag, variant_fn, out_csv, is_hybrid, df):
    print(f"\n########## plant propagation : {tag} ##########")

    tasks = [(idx, variant_fn, is_hybrid, row["algorithm"], row["grid_size"],
              int(row["grid_id"]), int(row["run_id"]), row["couplings"],
              int(row[INIT_OBJ_COL]))
             for idx, row in df.iterrows()]
    total = len(tasks)

    t0 = time.perf_counter()
    results = []
    try:
        gen = Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(_run_one)(*t) for t in tasks)
    except TypeError:
        gen = Parallel(n_jobs=N_JOBS)(delayed(_run_one)(*t) for t in tasks)
    for k, r in enumerate(gen, 1):
        results.append(r)
        elapsed = time.perf_counter() - t0
        eta = (elapsed / k) * (total - k)
        print(f"\r[{100*k/total:5.1f}%] {k}/{total}  "
              f"elapsed {elapsed/60:5.1f}m  ETA {eta/60:5.1f}m      ",
              end="", flush=True)
    print()

    out_df = pd.DataFrame(results).sort_values(
        ["source_algorithm", "grid_size", "grid_id", "run_id"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    print(f"\n--- {tag} improvement per source x size ---")
    g = out_df.groupby(["source_algorithm", "grid_size"]).agg(
        init_mean=("initial_total_objective_value", "mean"),
        improved_mean=("improved_total_objective_value", "mean"),
        n=("run_id", "count")).round(2)
    g["gain_pct"] = (100 * (g["init_mean"] - g["improved_mean"]) / g["init_mean"]).round(2)
    print(g)
    print(f"\nSaved: {out_csv}  ({len(out_df)} rows)")


def main():
    df = pd.read_csv(BASELINE_CSV)
    df = df[df["algorithm"].isin(SOURCES_TO_RUN) & df["grid_size"].isin(SIZES_TO_RUN)]
    df = df.reset_index(drop=True)
    print(f"Baseline solutions to process: {len(df)}")
    print(df.groupby(["algorithm", "grid_size"]).size().to_string())

    for tag, variant_fn, out_csv, is_hybrid in VARIANTS:
        run_variant(tag, variant_fn, out_csv, is_hybrid, df)


if __name__ == "__main__":
    main()