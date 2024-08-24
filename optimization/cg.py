import numpy as np
from ortools.linear_solver import pywraplp
from data_gen import data_generator


def generate_pattern(nb_piece: int, data: data_generator.data_cs) -> np.array:
    """Generate a feasible pattern to initialize the algorithm."""

    pattern = np.zeros((nb_piece, nb_piece))
    for i in range(nb_piece):
        pattern[i, i] = np.floor(data.W / data.pieces[i].w)

    return pattern


def master_problem(pattern: np.array,
                   data: data_generator.data_cs,
                   nb_piece: int,
                   nb_rouleaux: int):
    """Master problem of the column generation algorithm."""

    # OR-Tools model
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError("Failed to create solver.")

    # Decision variables
    num_patterns = pattern.shape[0]
    x = [solver.NumVar(0.0, solver.infinity(), f"x({p})") for p in range(num_patterns)]
    
    # Objective function: Minimize the number of chosen patterns
    objective = solver.Objective()
    for var in x:
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()

    # Constraints: The sum of chosen patterns should fit the demand
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
    
    # OR-Tools model
    solver_pricing = pywraplp.Solver.CreateSolver('SCIP')
    if not solver_pricing:
        raise RuntimeError("Failed to create solver.")

    # Decision variables
    y = [solver_pricing.IntVar(0, solver_pricing.infinity(), f"y({i})") for i in range(len(dual_variables))]
    
    # Objective function: Minimize the reduced costs
    objective = solver_pricing.Objective()
    for i, dual in enumerate(dual_variables):
        objective.SetCoefficient(y[i], dual)
    objective.SetMaximization() 

    # Constraint: Pattern should meet the length constraint
    solver_pricing.Add(sum(data.pieces[i].w * y[i] for i in range(len(dual_variables))) <= data.W)

    status = solver_pricing.Solve()

    if status == pywraplp.Solver.OPTIMAL and objective.Value() > 1 + 1e-8:
        return np.array([y[i].solution_value() for i in range(len(dual_variables))])
    else:
        return None


def master_problem_integer(pattern: np.array, data: data_generator.data_cs, nb_piece: int):
    """Master problem of the column generation algorithm with integer constraints."""
  
    # OR-Tools model
    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')
    if not solver:
        raise RuntimeError("Failed to create solver.")

    # Decision variables
    num_patterns = pattern.shape[0]
    x = [solver.IntVar(0, 1, f"x({p})") for p in range(num_patterns)]

    # Objective function: Minimize the number of chosen patterns
    objective = solver.Objective()
    for var in x:
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()

    # Constraints: The sum of chosen patterns should fit the demand
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


def main_gc(data: data_generator.data_cs, nb_piece: int, nb_rouleaux: int):
    """Run the main algorithm for the cutting stock problem."""

    pattern = generate_pattern(nb_piece, data)
    max_iter = nb_rouleaux
    iter_count = 0

    while iter_count < max_iter:
        dual_variables, master_objective, solution_vector, status = master_problem(pattern,
                                                                                   data,
                                                                                   nb_piece,
                                                                                   nb_rouleaux)

        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("Master problem did not converge to an optimal solution.")

        new_pattern = pricing_problem(data, dual_variables)

        if new_pattern is None:
            print("Initial Objective value: ", master_objective)
            final_objective, final_solution_vector, final_status = master_problem_integer(pattern, data, nb_piece)
            
            if final_status == pywraplp.Solver.OPTIMAL:
                print("Final Objective value with integer solution: ", final_objective)
                return final_objective, final_solution_vector
            else:
                raise RuntimeError("Final integer problem did not converge to an optimal solution.")

        pattern = np.vstack((pattern, new_pattern))
        iter_count += 1

    raise ValueError("GC method did not converge within the maximum number of iterations.")

