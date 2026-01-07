import numpy as np
import time
import pandas as pd
import random
from queue import PriorityQueue
from scipy.sparse.csgraph import minimum_spanning_tree
import csv
import psutil
import sys

# Helper function to calculate total distance of a TSP tour
def total_distance(tour, distance_matrix):
    return sum(distance_matrix[tour[i - 1]][tour[i]] for i in range(len(tour)))


# Nearest Neighbors (NN) algorithm
def NN(matrix):
    n = len(matrix)
    unvisited = set(range(n))
    tour = [0]
    unvisited.remove(0)
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda city: matrix[last][city])
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

# 2-Opt algorithm
def two_opt(tour, matrix):
    best = tour
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                if total_distance(new_tour, matrix) < total_distance(best, matrix):
                    best = new_tour
                    improved = True
        tour = best
    return best

# Nearest Neighbors with 2-Opt (NN2O) algorithm
def NN2O(matrix):
    tour = nearest_neighbors(matrix)
    return two_opt(tour, matrix)

# Repeated Randomized Nearest Neighbors with 2-Opt (RNN) algorithm
def RNN(matrix, n):
    best_tour = None
    best_cost = float('inf')
    for start in range(len(matrix)):
        unvisited = set(range(len(matrix)))
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            last = tour[-1]
            candidates = sorted(unvisited, key=lambda city: matrix[last][city])[:n]
            next_city = random.choice(candidates)
            tour.append(next_city)
            unvisited.remove(next_city)
        tour = two_opt(tour, matrix)
        cost = total_distance(tour, matrix)
        if cost < best_cost:
            best_tour = tour
            best_cost = cost
    return best_tour

#MST Heuristic
def mst_heuristic(matrix, visited):
    n = len(matrix)
    remaining = [i for i in range(n) if i not in visited]
    if not remaining:
        return 0
    sub_matrix = matrix[np.ix_(remaining, remaining)]
    mst = minimum_spanning_tree(sub_matrix).toarray().astype(int)
    return mst[mst != 0].sum()

# A* algorithm for TSP with MST heuristic
def A_MST(matrix):
    n = len(matrix)
    start = 0
    pq = PriorityQueue()
    pq.put((0, [start], 0))  # (cost, path, g(n))
    best_cost = float('inf')
    best_path = None

    while not pq.empty():
        cost, path, g = pq.get()
        current = path[-1]

        if len(path) == n:
            total_cost = g + matrix[current][start]
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path + [start]
            continue

        for next_city in range(n):
            if next_city not in path:
                new_g = g + matrix[current][next_city]
                h = mst_heuristic(matrix, path + [next_city])
                f = new_g + h
                pq.put((f, path + [next_city], new_g))

    return best_path, best_cost

# Hill-Climbing method for TSP
def hillClimbing(matrix, num_restarts=3):
    def get_neighbors(tour):
        neighbors = []
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                neighbor = tour[:]
                neighbor[i:j] = reversed(neighbor[i:j])
                neighbors.append(neighbor)
        return neighbors

    def random_tour(n):
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    best_tour = None
    best_cost = float('inf')
    n = len(matrix)

    for _ in range(num_restarts):
        current_tour = random_tour(n)
        current_cost = total_distance(current_tour, matrix)
        improved = True

        while improved:
            improved = False
            neighbors = get_neighbors(current_tour)
            for neighbor in neighbors:
                neighbor_cost = total_distance(neighbor, matrix)
                if neighbor_cost < current_cost:
                    current_tour = neighbor
                    current_cost = neighbor_cost
                    improved = True

        if current_cost < best_cost:
            best_tour = current_tour
            best_cost = current_cost

    return best_tour, best_cost

# Simulated Annealing method for TSP
def simuAnnealing(matrix, initial_temp=1000, cooling_rate=0.995, num_restarts=3):
    def get_neighbors(tour):
        neighbors = []
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                neighbor = tour[:]
                neighbor[i:j] = reversed(neighbor[i:j])
                neighbors.append(neighbor)
        return neighbors

    def random_tour(n):
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    def acceptance_probability(old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / temperature)

    best_tour = None
    best_cost = float('inf')
    n = len(matrix)

    for _ in range(num_restarts):
        current_tour = random_tour(n)
        current_cost = total_distance(current_tour, matrix)
        temperature = initial_temp

        while temperature > 1:
            neighbors = get_neighbors(current_tour)
            next_tour = random.choice(neighbors)
            next_cost = total_distance(next_tour, matrix)

            if acceptance_probability(current_cost, next_cost, temperature) > random.random():
                current_tour = next_tour
                current_cost = next_cost

            temperature *= cooling_rate

        if current_cost < best_cost:
            best_tour = current_tour
            best_cost = current_cost

    return best_tour, best_cost

# Genetic Algorithm method for TSP
def genetic(matrix, num_generations=50, population_size=10, mutation_rate=0.01, crossover_rate=0.7):
    def random_tour(n):
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    def crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        pointer = 0
        for i in range(size):
            if child[i] is None:
                while parent2[pointer] in child:
                    pointer += 1
                child[i] = parent2[pointer]
        return child

    def mutate(tour):
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]

    def select(population, fitnesses):
        total_fitness = sum(fitnesses)
        pick = random.uniform(0, total_fitness)
        current = 0
        for tour, fitness in zip(population, fitnesses):
            current += fitness
            if current > pick:
                return tour

    n = len(matrix)
    population = [random_tour(n) for _ in range(population_size)]
    best_tour = min(population, key=lambda tour: total_distance(tour, matrix))
    best_cost = total_distance(best_tour, matrix)

    for _ in range(num_generations):
        fitnesses = [1 / total_distance(tour, matrix) for tour in population]
        new_population = []

        for _ in range(population_size // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)

            if random.random() < crossover_rate:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]

            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)

            new_population.extend([child1, child2])

        population = new_population
        current_best_tour = min(population, key=lambda tour: total_distance(tour, matrix))
        current_best_cost = total_distance(current_best_tour, matrix)

        if current_best_cost < best_cost:
            best_tour = current_best_tour
            best_cost = current_best_cost

    return best_tour, best_cost

# Function to read adjacency matrix from file
def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        size = int(lines[0].strip())
        matrix = []
        for line in lines[1:]:
            matrix.append(list(map(int, line.strip().split())))
    return np.array(matrix)

# Function to write results to CSV file
def write_results_to_csv(filename, total_cost, num_nodes, cpu_runtime, real_runtime):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([total_cost, num_nodes, cpu_runtime, real_runtime])

def main():
    # Check for correct usage
    if len(sys.argv) < 2:
        print("Usage: python tsp_solver.py <infile.txt>")
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        # Read the input file and process it into a matrix
        with open(input_file, 'r') as file:
            lines = file.readlines()
            
            # Read the number of nodes from the first line
            num_nodes = int(lines[0].strip())
            
            # Read the adjacency matrix, converting each line into a list of integers
            distance_matrix = np.array([[int(x) for x in line.strip().split()] for line in lines[1:num_nodes+1]])

        print("Successfully read the input file.")
    
    except Exception as e:
        print(f"Error reading the input file: {e}")
        sys.exit(1)

    # List of algorithms to run
    algorithms = [
        ("nearest_neighbors", nearest_neighbors),
        ("nearest_neighbors_2opt", nearest_neighbors_2opt),
        ("randomized_nearest_neighbors_2opt", lambda matrix: randomized_nearest_neighbors_2opt(matrix, 3)),
        ("a_star_tsp", a_star_tsp),
        ("hill_climbing", hill_climbing),
        ("simulated_annealing", simulated_annealing),
        ("genetic_algorithm", genetic_algorithm)
    ]

    # Run each algorithm and write results to a corresponding CSV file
    for name, algorithm in algorithms:
        # Measure real-world and CPU time with high resolution
        real_time_start = time.perf_counter()
        cpu_time_start = time.perf_counter()

        # Handle algorithms that return either a single or double result
        result = algorithm(distance_matrix)
        if isinstance(result, tuple):  # if the result is a tuple (best_tour, best_cost)
            best_tour, best_cost = result
        else:  # otherwise, it's a single value (best_tour only)
            best_tour = result
            best_cost = total_distance(best_tour, distance_matrix)

        # Measure real-world and CPU time at the end
        real_time_end = time.perf_counter()
        cpu_time_end = time.perf_counter()

        # Calculate times
        real_time = real_time_end - real_time_start
        cpu_time = cpu_time_end - cpu_time_start

        # Number of nodes in the tour, assuming the length of the best tour is the count
        nodes_expanded = len(best_tour)

        # Write the results to a CSV file
        with open(f"{name}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([best_cost, nodes_expanded, cpu_time, real_time])

        print(f"Results for {name} saved to {name}.csv")

if __name__ == "__main__":
    main()
