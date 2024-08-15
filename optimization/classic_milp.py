"""This file contain a classic formulation and resolution for a cutting stock problem using integer programming"""

import numpy as np
from ortools.linear_solver import pywraplp
from data_gen import data_generator

def classic_MILP(data: data_generator.data_cs, nb_piece: int):
    """Resolve a cutting stock problem with a classic formulation.
    
    (1): Minimize the number of used rolls.
    (2): Do not exceed rolls' lengths.
    (3): Every order has to be cut sufficiently often to fit the demand.
    (4): Cutting a roll is equivalent to using it.
    """
    
    # OR-Tools model
    solver_classic = pywraplp.Solver('SolveMIP_pricing', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    # Decision variables
    y = [solver_classic.IntVar(0, 1, f"y({j})") for j in range(nb_piece)]
    x = [
        [
            solver_classic.IntVar(0, solver_classic.infinity(), f"x({i},{j})") for j in range(nb_piece)
        ] 
        for i in range(len(data.pieces))
    ]

    ### (1) Objective function: Minimize the number of used rolls
    objective = solver_classic.Objective()
    for j in range(nb_piece):
        objective.SetCoefficient(y[j], 1)
    objective.SetMinimization()

    ### (2) Constraint length: Do not exceed roll's length
    for j in range(nb_piece):
        solver_classic.Add(
            sum(data.pieces[i].w * x[i][j] for i in range(len(data.pieces))) <= data.W
        )

    ### (3) Constraint demand: Meet the demand for each piece type
    for i in range(len(data.pieces)):
        solver_classic.Add(
            sum(x[i][j] for j in range(nb_piece)) >= data.pieces[i].d
        )

    ### (4) Constraint using a roll: A roll is used if any piece is cut from it
    for i in range(len(data.pieces)):
        for j in range(nb_piece):
            solver_classic.Add(
                x[i][j] <= data.pieces[i].d * y[j]
            )

    # Solve the problem
    status = solver_classic.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found.")
        print(f"Objective value: {solver_classic.Objective().Value()}")
        solution_x = np.zeros((len(data.pieces), nb_piece))
        solution_y = np.zeros(nb_piece)
        for j in range(nb_piece):
            solution_y[j] = y[j].solution_value()
            for i in range(len(data.pieces)):
                solution_x[i][j] = x[i][j].solution_value()
        
        return solver_classic.Objective().Value(), solution_y
    else:
        print("No optimal solution found.")
        return None, None
