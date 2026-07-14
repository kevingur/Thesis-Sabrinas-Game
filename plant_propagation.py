import numpy as np
import random 
import time

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

            (r1, c1), (r2, c2) = positions
            couplings.append(((int(r1), int(c1)), (int(r2), int(c2))))
    return couplings


def total_cost_from_beginning(grid,id_grid):

    final_couplings = id_grid_to_couplings(id_grid)   

    total_cost = sum( max(grid[c1], grid[c2]) - min(grid[c1], grid[c2]) for c1, c2 in final_couplings)
    
    return total_cost


def make_offspring(parent_id_grid, grid, n_flips):
    child = parent_id_grid.copy()       
    
    for _ in range(n_flips):
        flippable = find_flippable_blocks(child)
        if not flippable:
            break
        i, j = random.choice(flippable)
        apply_flip(child, i, j)
    
    child_cost = total_cost_from_beginning(grid, child)
    return child, child_cost


def population(parent_id_grid,grid,N,n_flips):
    
    population = []

    for i in range(N):
        child_id_grid, child_cost = make_offspring(parent_id_grid,grid, n_flips)
        population.append((child_id_grid, int(child_cost)))

    return population


def normalize_fitness_scores(population):
    costs = [child_cost for (child_id_grid, child_cost) in population] 
    max_cost = max(costs)
    min_cost = min(costs)
    
    fitness_scores = []
    for (child_id_grid, child_cost) in population:
        if max_cost == min_cost:              
            f = 1.0                           
        else:
            f = (max_cost - child_cost) / (max_cost - min_cost)
        fitness_scores.append(round(f, 2))

    return fitness_scores


def generate_offspring(population, fitness_scores, grid, n_max_runners, max_flips):
    offspring = []
    
    for (parent_id_grid, parent_cost), f in zip(population, fitness_scores):
        n_runners = max(1, round(n_max_runners * f * random.random()))
        
        for _ in range(n_runners):
            n_flips = max(1, round((1 - f) * max_flips * random.random()))
            
            child, child_cost = make_offspring(parent_id_grid, grid, n_flips)
            offspring.append((child, child_cost))
    
    return offspring


def select_next_population(population, offspring, N):
    combined = population + offspring          
    combined.sort(key=lambda x: x[1])          
    return combined[:N]                   


def plant_propagation(couplings, initial_cost, grid, N=20, max_evaluations=10000,
                            n_max_runners=5, init_flips=5):
    max_flips = grid.size // 4
    start = time.perf_counter()

    couples_dict = give_id(couplings)
    parent_id_grid = np.zeros(grid.shape, dtype=int)
    write_id_in_grid(parent_id_grid, couples_dict)

    pop = population(parent_id_grid, grid, N, init_flips)
    evaluations = N
    best_id_grid, best_cost = min(pop, key=lambda x: x[1])
    best_id_grid = best_id_grid.copy()

    while evaluations < max_evaluations:
        fitness = normalize_fitness_scores(pop)
        offspring = generate_offspring(pop, fitness, grid, n_max_runners, max_flips)
        evaluations += len(offspring)
        pop = select_next_population(pop, offspring, N)

        gen_best_grid, gen_best_cost = min(pop, key=lambda x: x[1])
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_id_grid = gen_best_grid.copy()

    runtime = time.perf_counter() - start
    final_couplings = id_grid_to_couplings(best_id_grid)


    return int(initial_cost), int(best_cost), evaluations, runtime, final_couplings


def calculate_objective_value(grid,r1,c1,r2,c2):
    return abs(grid[r1, c1] - grid[r2, c2])


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
            window = id_grid[wi:wi+n, wj:wj+n]
            _, counts = np.unique(window, return_counts=True)
            if np.all(counts == 2):
                clean.append((wi, wj))
    return clean


def make_offspring_hybrid(parent_id_grid, grid, n_flips, n=4, p_riprepair=0.3,
                          repair_fn=None):
    if repair_fn is None:
        repair_fn = random_unstuck

    child = parent_id_grid.copy()
    next_id = child.max() + 1

    for _ in range(n_flips):
        clean_windows = find_clean_windows(child, n)
        do_riprepair = clean_windows and (random.random() < p_riprepair)

        if do_riprepair:
            wi, wj = random.choice(clean_windows)
            sub = grid[wi:wi+n, wj:wj+n]
            _, _, _, local_couplings = repair_fn(sub)
            for (lr1, lc1), (lr2, lc2) in local_couplings:
                child[lr1+wi, lc1+wj] = next_id
                child[lr2+wi, lc2+wj] = next_id
                next_id += 1
        else:
            flippable = find_flippable_blocks(child)
            if not flippable:
                if not clean_windows:
                    break
                else:
                    continue
            i, j = random.choice(flippable)
            apply_flip(child, i, j)

    child_cost = total_cost_from_beginning(grid, child)
    return child, child_cost


def generate_offspring_hybrid(population, fitness_scores, grid, n_max_runners,
                              max_flips, n=4, p_riprepair=0.3, repair_fn=None):
    offspring = []
    for (parent_id_grid, parent_cost), f in zip(population, fitness_scores):
        n_runners = max(1, round(n_max_runners * f * random.random()))
        for _ in range(n_runners):
            n_flips = max(1, round((1 - f) * max_flips * random.random()))
            child, child_cost = make_offspring_hybrid(
                parent_id_grid, grid, n_flips, n, p_riprepair, repair_fn)
            offspring.append((child, child_cost))
    return offspring


def population_hybrid(parent_id_grid, grid, N, n_flips, n=4, p_riprepair=0.3,
                      repair_fn=None):
    pop = []
    for _ in range(N):
        child, child_cost = make_offspring_hybrid(parent_id_grid, grid, n_flips,
                                                  n, p_riprepair, repair_fn)
        pop.append((child, int(child_cost)))
    return pop

def plant_propagation_hybrid(couplings, initial_cost, grid, N=20, max_evaluations=10000,
                             n_max_runners=5, init_flips=5, n=4, p_riprepair=0.3,
                             repair_fn=None):
    max_flips = grid.size // 4
    start = time.perf_counter()

    couples_dict = give_id(couplings)
    parent_id_grid = np.zeros(grid.shape, dtype=int)
    write_id_in_grid(parent_id_grid, couples_dict)

    pop = population_hybrid(parent_id_grid, grid, N, init_flips, n, p_riprepair, repair_fn)
    evaluations = N
    best_id_grid, best_cost = min(pop, key=lambda x: x[1])
    best_id_grid = best_id_grid.copy()

    while evaluations < max_evaluations:
        fitness = normalize_fitness_scores(pop)
        offspring = generate_offspring_hybrid(pop, fitness, grid, n_max_runners,
                                              max_flips, n, p_riprepair, repair_fn)
        evaluations += len(offspring)
        pop = select_next_population(pop, offspring, N)

        gen_best_grid, gen_best_cost = min(pop, key=lambda x: x[1])   
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_id_grid = gen_best_grid.copy()

    runtime = time.perf_counter() - start
    final_couplings = id_grid_to_couplings(best_id_grid)
    return int(initial_cost), int(best_cost), evaluations, runtime, final_couplings