# TSP Search and Optimization
This project investigates a wide spectrum of search algorithms for solving the **Traveling Salesperson Problem (TSP)** and compares them empirically against optimal solutions computed by **A\* with a Minimum Spanning Tree heuristic**.

Course: CMSC 421 – Artificial Intelligence  


# Project Goals

- Implement classical greedy and randomized TSP solvers
- Improve naive solutions using 2-Opt local refinement
- Evaluate stochastic and population-based optimizers
- Implement **A*** using a graph-theoretic admissible heuristic (MST)
- Analyze scalability as the TSP size increases

---

# Algorithms Implemented

| Category | Algorithm | Notes |
|----------|-----------|-------|
| Greedy | **Nearest Neighbor (NN)** | Very fast, low quality |
| Local Refinement | **NN + 2-Opt (NN2O)** | Fixes bad edges; improves NN |
| Randomized Search | **Repeated Randomized NN + 2-Opt (RNN)** | Multiple starts + randomness |
| Optimal Search | **A\* with MST heuristic** | Guarantees optimality; slow |
| Local Search | **Hill Climbing** | Plateaus & local minima |
| Stochastic Local Search | **Simulated Annealing (SA)** | Escapes local minima |
| Evolutionary | **Genetic Algorithm (GA)** | Best large-scale heuristic |


# Results Summary

### Part 1 — NN / NN2O / RNN Comparison  
**RNN consistently produces the lowest-cost tours.**
- NN is fastest but worst quality
- NN2O improves results significantly
- RNN gives best solutions but takes longer  
Plots example:  
- plots/part1_cost_nodes.png  
- plots/part1_runtime.png


# Part 2 — A\* + MST vs Part 1 Heuristics  
**A\* always finds optimal tours (when tractable).**
- Confirms gap between greedy/random heuristics and optimal cost
- Large graphs make A\* explode in runtime  
Plots example:  
- plots/part2_cost_diff.png  
- plots/part2_nodes_diff.png


# Part 3 — Local Search & Metaheuristics  
Compared to A\* ground truth:
- **Hill climbing** improves NN but gets stuck often
- **Simulated annealing** balances runtime & quality
- **Genetic algorithm** performs best on larger graphs  
Plot examples:  
- plots/part3_hill.png 
- plots/part3_simanneal.png 
- plots/part3_genetic.png
