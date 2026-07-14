import numpy as np
import random 
import time
from collections import Counter

def calculate_objective_value(grid,r1,c1,r2,c2):
    return abs(grid[r1, c1] - grid[r2, c2])


def apply_flip(id_grid, i, j):
    a = id_grid[i, j]
    b = id_grid[i, j+1]
    c = id_grid[i+1, j]
    d = id_grid[i+1, j+1]
    
    if a == b and c == d:
        id_grid[i, j]     = a
        id_grid[i+1, j]   = a
        id_grid[i, j+1]   = c
        id_grid[i+1, j+1] = c
    elif a == c and b == d:
        id_grid[i, j]     = a
        id_grid[i, j+1]   = a
        id_grid[i+1, j]   = b
        id_grid[i+1, j+1] = b


def find_flippable_blocks(id_grid):
    n_rows, n_cols = id_grid.shape
    flippable = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            a = id_grid[i, j]
            b = id_grid[i, j+1]
            c = id_grid[i+1, j]
            d = id_grid[i+1, j+1]

            if (a == b and c == d) or (a == c and b == d):
                flippable.append((i, j))
    return flippable


def block_objective_value(grid, id_grid,i,j):
    a = id_grid[i,j]
    if id_grid[i,j+1] == a:
        b = id_grid[i+1,j]
    else:
        b = id_grid[i,j+1]


    a_cells = grid[id_grid ==a]
    b_cells = grid[id_grid ==b]

    cost_a = abs(int(a_cells[0]) - int(a_cells[1]))
    cost_b = abs(int(b_cells[0]) - int(b_cells[1]))

    return cost_a + cost_b

def give_id(couplings):
    couples_dict = {}

    number = 1

    for couple in couplings:
        couples_dict[couple] = number
        number +=1

    return couples_dict


def write_id_in_grid(id_grid,couples_dict):
    for (key_1,key_2), idd in couples_dict.items():
        
        i,j = key_1
        m,n = key_2

        id_grid[i,j] = idd
        id_grid[m,n] = idd

def id_grid_to_couplings(id_grid):
    couplings = []
    seen = set()
    n_rows, n_cols = id_grid.shape
    for i in range(n_rows):
        for j in range(n_cols):
            tid = id_grid[i, j]
            if tid in seen:
                continue
            seen.add(tid)
            positions = np.argwhere(id_grid == tid)
            if len(positions) != 2:                          
                print(f"PROBLEM: ID {tid} appears in {len(positions)} cells: {positions.tolist()}")
                continue
            (r1, c1), (r2, c2) = positions
            couplings.append(((int(r1), int(c1)), (int(r2), int(c2))))
    return couplings


def total_objective_value_from_beginning(grid, id_grid):
    total = 0
    for tid in np.unique(id_grid):
        cells = grid[id_grid == tid]
        total += cells.max() - cells.min()
    return total


def hill_climbing(couplings, initial_total_objective_value, grid, max_evaluations = 10000):
    couples_dict = give_id(couplings)
    grid_shape = grid.shape
    id_grid = np.zeros(grid_shape, dtype=int)
    write_id_in_grid(id_grid, couples_dict)

    improved_total_objective_value = initial_total_objective_value
    improvements_made = 0
    sideways_made = 0

    evaluations = 0
    start = time.perf_counter()

    while evaluations < max_evaluations:
        blocks = find_flippable_blocks(id_grid)
        if not blocks:
            break
        i, j = random.choice(blocks)

        old_block = block_objective_value(grid, id_grid, i, j)
        apply_flip(id_grid, i, j)
        new_block = block_objective_value(grid, id_grid, i, j)
        delta = new_block - old_block

        evaluations +=1

        if delta < 0:
            improved_total_objective_value += delta
            improvements_made += 1
        elif delta == 0:
            sideways_made += 1

        else:
            apply_flip(id_grid, i, j)

    final_couplings = id_grid_to_couplings(id_grid)
    assert total_objective_value_from_beginning(grid, id_grid) == improved_total_objective_value
    runtime = time.perf_counter()-start
    return int(initial_total_objective_value),int(improved_total_objective_value), improvements_made,evaluations, runtime, final_couplings,sideways_made


def get_available_neighbors(grid, cell,available_indices):
    rows, cols = grid.shape
    r,c = cell

    possible_neighbors = [
        (r - 1, c),  
        (r + 1, c),  
        (r, c - 1),  
        (r, c + 1)   
    ]

    neighbors = []

    for nr, nc in possible_neighbors:
        if 0 <= nr < rows and 0 <= nc < cols:
            if (nr, nc) in available_indices:
                neighbors.append((nr, nc))

    return neighbors

def random_unstuck(grid):  

    failed_attempts = 0  
    start = time.perf_counter()

    while True:
        available = list(np.ndindex(grid.shape))
        random.shuffle(available)
        
        couplings = []
        total_objective_value = 0
        failed = False


        while len(available) > 0:
            cell = min(available, 
                       key=lambda c: len(get_available_neighbors(grid, c, available)))

            neighbors = get_available_neighbors(grid, cell, available)
            if len(neighbors) == 0:
                failed = True
                break

            chosen_neighbor = random.choice(neighbors)

            r1, c1 = cell
            r2, c2 = chosen_neighbor

            objective_value = calculate_objective_value(grid,r1,c1,r2,c2)
            total_objective_value += objective_value

            couplings.append((cell, chosen_neighbor))
            available.remove(cell)
            available.remove(chosen_neighbor)            
        if not failed:
            runtime = time.perf_counter() - start
            return int(total_objective_value), failed_attempts, runtime, couplings
        else:
            failed_attempts +=1


def find_clean_windows(id_grid, n):
    n_rows, n_cols = id_grid.shape
    clean = []
    for wi in range(n_rows - n + 1):
        for wj in range(n_cols - n + 1):
            counts = Counter(id_grid[wi:wi+n, wj:wj+n].flatten())
            if all(c == 2 for c in counts.values()):
                clean.append((wi, wj))
    return clean


def window_objective(grid, id_grid, wi, wj, n):
    id_cells = {}
    for r in range(wi, wi + n):
        for c in range(wj, wj + n):
            tid = id_grid[r, c]
            id_cells.setdefault(tid, []).append((r, c))
    total = 0
    for (r1, c1), (r2, c2) in id_cells.values():
        total += abs(int(grid[r1, c1]) - int(grid[r2, c2]))
    return total

def hill_climbing_hybrid(couplings, initial_total_objective_value, grid,
                         max_evaluations=10000, p_riprepair=0.3):
    
    repair_function= random_unstuck

    couples_dict = give_id(couplings)
    id_grid = np.zeros(grid.shape, dtype=int)
    write_id_in_grid(id_grid, couples_dict)

    improved_total= initial_total_objective_value
    improvements_made = 0
    sideways_made = 0
    flip_count = 0
    riprepair_count = 0
    next_id = id_grid.max() + 1

    evaluations = 0
    start = time.perf_counter()

    while evaluations < max_evaluations:
        clean_windows = find_clean_windows(id_grid, 4)
        do_riprepair = clean_windows and (random.random() < p_riprepair)

        if do_riprepair:
            wi, wj = random.choice(clean_windows)
            old_window_objective = window_objective(grid, id_grid, wi, wj, 4)
            old_block_ids = id_grid[wi:wi+4, wj:wj+4].copy()

            sub_grid = grid[wi:wi+4, wj:wj+4]
            _, _, _, local_couplings = repair_function(sub_grid)

            new_window_obj = 0
            for (lr1, lc1), (lr2, lc2) in local_couplings:
                gr1, gc1 = lr1 + wi, lc1 + wj
                gr2, gc2 = lr2 + wi, lc2 + wj
                id_grid[gr1, gc1] = next_id
                id_grid[gr2, gc2] = next_id
                next_id += 1
                new_window_obj += abs(int(grid[gr1, gc1]) - int(grid[gr2, gc2]))

            delta = new_window_obj - old_window_objective
            evaluations += 1
            riprepair_count += 1

            if delta < 0:
                improved_total+= delta
                improvements_made += 1
            elif delta == 0:
                sideways_made += 1
            else:
                id_grid[wi:wi+4, wj:wj+4] = old_block_ids   

        else:
            blocks = find_flippable_blocks(id_grid)
            if not blocks:
                if not clean_windows:
                    break
                else:
                    continue
            i, j = random.choice(blocks)
            old_block = block_objective_value(grid, id_grid, i, j)
            apply_flip(id_grid, i, j)
            new_block = block_objective_value(grid, id_grid, i, j)
            delta = new_block - old_block
            evaluations += 1
            flip_count += 1

            if delta < 0:
                improved_total+= delta
                improvements_made += 1
            elif delta == 0:
                sideways_made += 1
            else:
                apply_flip(id_grid, i, j)  

    final_couplings = id_grid_to_couplings(id_grid)
    assert total_objective_value_from_beginning(grid, id_grid) == improved_total
    runtime = time.perf_counter() - start
    return (int(initial_total_objective_value), int(improved_total), improvements_made, evaluations, runtime, final_couplings, sideways_made, flip_count, riprepair_count)