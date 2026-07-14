import random
import numpy as np
import time

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


def calculate_objective_value(grid,r1,c1,r2,c2):
    return abs(grid[r1, c1] - grid[r2, c2])

# RANDOM 
def random_algorithm(grid):  

    failed_attempts = 0  
    start = time.perf_counter()
    while True:

        available = list(np.ndindex(grid.shape))
    
        couplings = []
        total_objective_value = 0
        failed = False
        while len(available) > 0:

            cell = random.choice(available)

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
            return int(total_objective_value), failed_attempts,runtime, couplings
        else:
            failed_attempts +=1

# RANDOM UNSTUCK
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