import os
import glob
import numpy as np
import pandas as pd

from greedy import greedy, greedy_unstuck  

GRIDS_ROOT = "grids"          
N_RUNS = 30            
OUT_CSV = "greedy_results.csv"

ALGORITHMS = [
    ("greedy",          greedy,          ["6x6", "8x8", "12x12"]),
    ("greedy_unstuck",  greedy_unstuck,  ["6x6", "8x8", "12x12", "16x16", "18x18"]),
]


def load_grids(size):
    folder = os.path.join(GRIDS_ROOT, size)
    paths = sorted(glob.glob(os.path.join(folder, "*.npy")))
    grids = []
    for p in paths:
        grid_id = os.path.splitext(os.path.basename(p))[0]
        grid = np.load(p)
        grids.append((grid_id, grid))
    return grids


def main():
    header_written = os.path.exists(OUT_CSV)

    for algo_name, algo_fn, sizes in ALGORITHMS:
        print(f"\n########## {algo_name} ##########")

        for size in sizes:
            grids = load_grids(size)
            cell_count = int(size.split("x")[0]) ** 2
            print(f"\n=== {algo_name} : {size} : {len(grids)} grids ===")

            for grid_id, grid in grids:
                rows = []
                for run in range(N_RUNS):
                    total_cost, failed_attempts, runtime, couplings = algo_fn(grid)
                    rows.append({
                        "run_id": run + 1,
                        "algorithm": algo_name,
                        "grid": grid_id,
                        "grid_size": size,
                        "cell_count": cell_count,
                        "total_cost": total_cost,
                        "failed_attempts": failed_attempts,
                        "runtime_seconds": runtime,
                        "couplings": str(couplings),
                    })

                df = pd.DataFrame(rows)
                df.to_csv(OUT_CSV, mode="a", index=False, header=not header_written)
                header_written = True
                print(f"  grid {grid_id} done ({N_RUNS} runs)")


if __name__ == "__main__":
    main()