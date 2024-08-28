import numpy as np
import math
import time
from ortools.linear_solver import pywraplp
from data_gen import data_generator
import heapq


def generate_pattern(nb_piece: int, data: data_generator.data_cs) -> np.array:
    """Generate a feasible pattern to initialize the algorithm."""
    pattern = np.zeros((nb_piece, nb_piece))
    for i in range(nb_piece):
        pattern[i, i] = np.floor(data.W / data.pieces[i].w)
    return pattern


def master_problem(pattern: np.array, data: data_generator.data_cs, nb_piece: int):
    """Master problem of the column generation algorithm."""
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError("Failed to create solver.")
    
    num_patterns = pattern.shape[0]
    x = [solver.NumVar(0.0, solver.infinity(), f"x({p})") for p in range(num_patterns)]
    
    objective = solver.Objective()
    for var in x:
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()
    
    constraints = []
    for i in range(nb_piece):
        constraint = solver.Add(sum(pattern[p, i] * x[p] for p in range(num_patterns)) >= data.pieces[i].d)
        constraints.append(constraint)
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        dual_variables = [const.dual_value() for const in constraints]
        return dual_variables, objective.Value(), [var.solution_value() for var in x], status
    else:
        print('The problem does not have an optimal solution.')
        return None, None, None, status


def pricing_problem(data, dual_variables):
    """Create a feasible pattern to feed the master problem."""
    solver_pricing = pywraplp.Solver.CreateSolver('SCIP')
    if not solver_pricing:
        raise RuntimeError("Failed to create solver.")
    
    y = [solver_pricing.IntVar(0, solver_pricing.infinity(), f"y({i})") for i in range(len(dual_variables))]
    
    objective = solver_pricing.Objective()
    for i, dual in enumerate(dual_variables):
        objective.SetCoefficient(y[i], dual)
    objective.SetMaximization()
    
    solver_pricing.Add(sum(data.pieces[i].w * y[i] for i in range(len(dual_variables))) <= data.W)
    
    status = solver_pricing.Solve()
    
    if status == pywraplp.Solver.OPTIMAL and objective.Value() > 1 + 1e-8:
        return np.array([y[i].solution_value() for i in range(len(dual_variables))])
    else:
        return None


def master_problem_integer(pattern: np.array, data: data_generator.data_cs, nb_piece: int):
    """Master problem of the column generation algorithm with integer constraints."""
    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')
    if not solver:
        raise RuntimeError("Failed to create solver.")
    
    num_patterns = pattern.shape[0]
    x = [solver.IntVar(0, solver.infinity(), f"x({p})") for p in range(num_patterns)]
    
    objective = solver.Objective()
    for var in x:
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()
    
    constraints = []
    for i in range(nb_piece):
        constraint = solver.Add(sum(pattern[p, i] * x[p] for p in range(num_patterns)) >= data.pieces[i].d)
        constraints.append(constraint)
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        return objective.Value(), [var.solution_value() for var in x], status
    else:
        print('The problem does not have an optimal solution.')
        return None, None, status


def branch_and_price(data: data_generator.data_cs, nb_piece: int, nb_rouleaux: int, time_limit_seconds: int = 120):
    """Main branch-and-price algorithm for the cutting stock problem with a time limit."""

    print(time_limit_seconds)
    
    start_time = time.time()  # Start the timer

    pattern = generate_pattern(nb_piece, data)
    upper_bound = float('inf')
    lower_bound = -float('inf')
    heap = []
    solution_vector = [None]

    # Solve initial master problem
    initial_duals, master_obj, sol_vector, status = master_problem(pattern, data, nb_piece)
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Initial LP relaxation did not solve optimally.")

    # Initialize heap
    heapq.heappush(heap, (master_obj, pattern))

    while heap:
        # Check the elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit_seconds:
            print(f"Time limit of {time_limit_seconds} seconds reached. Stopping the algorithm.")
            return None, None

        current_obj, current_pattern = heapq.heappop(heap)

        if current_obj >= upper_bound:
            continue

        # Resolve the master problem to get the duals and new solution
        duals, master_obj, sol_vector, status = master_problem(current_pattern, data, nb_piece)
        if status != pywraplp.Solver.OPTIMAL:
            continue

        new_pattern = pricing_problem(data, duals)

        # Stop if no new pattern is found
        if new_pattern is None or np.all(new_pattern == 0):
            int_obj, int_sol, int_status = master_problem_integer(current_pattern, data, nb_piece)
            if int_status == pywraplp.Solver.OPTIMAL:
                if int_obj < upper_bound:
                    upper_bound = int_obj
                    solution_vector = int_sol
        else:
            # Add the new pattern to the pool and continue
            current_pattern = np.vstack((current_pattern, new_pattern))
            heapq.heappush(heap, (master_obj, current_pattern))

    if upper_bound < float('inf'):
        print("Optimal integer solution found with objective value:", upper_bound)
        return upper_bound, solution_vector
    else:
        raise RuntimeError("Branch-and-Price did not find an optimal solution.")