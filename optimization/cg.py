import numpy as np
from pyscipopt import Model, quicksum, multidict,SCIP_PARAMSETTING
from ortools.linear_solver import pywraplp
from data_gen import data_generator



# Commencer par générer un premier ensemble de pattern 
# Pattern est un matrice de taille (nb_pattern, nb_piece) ex pattern[1] = pour chaque piece le nombre de coupe possible dans le rouleau



def generate_pattern(nb_piece: int, data: data_generator.data_cs) ->np.array:
    """Generate a feasible pattern to initilize the algorithm."""
    
    pattern = np.zeros((nb_piece,nb_piece))
    for i in range(nb_piece):
        pattern[i,i] = np.floor(data.W/data.pieces[i].w)
        
    return pattern



def master_problem(pattern: np.array,
                   data: data_generator.data_cs,
                   nb_piece: int):
    """Master problem of the column generation algorithm.

    Minimize the number of chosen patterns so that,
    the sum of all piece in chosen patterns satisfy the demand.
    
    Inputs:
    - pattern : 2D array : a feasible pattern
    - data: Data_CS : data for the cutting stock problem
    - nb_piece: int : number of piece in total for the demand
    
    Returns:
    - dual variables
    - objective value
    - decision variables
    - solving status
    
    
    (1): Minimize the number of chosen patterns
    (2): The sum of chosen pattern should fit the demand
    
    """
    #  OR-Tools model
    solver = pywraplp.Solver('SolveLP', pywraplp.Solver.CLP_LINEAR_PROGRAMMING )
    constraints = []

    # Decision variables
    x = [solver.NumVar(0.0, solver.infinity(), f"x({p})") for p in range(pattern.shape[0])]
    
    # (1) Objective function
    objective = solver.Objective()
    for var in x:
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()


    # (2) Constraints
    for i in range(nb_piece):
        constraint = solver.Add(sum(pattern[p,i]* x[p] for p in range(pattern.shape[0])) >= data.pieces[i].d)
        constraints.append(constraint)

    # Optimisation
    status = solver.Solve()
    dual_variables = []

    if status == pywraplp.Solver.OPTIMAL:
        for index, const in enumerate(constraints):
            dual_variables.append(const.dual_value())
    else:
        print('The problem does not have an optimal solution.')

    return dual_variables, objective.Value(), [x[p].solution_value() for p in range(pattern.shape[0])], status




def pricing_problem(data,dual_variables):
    """Create feasible pattern to feed the master problem.
    
    It optimize the reduced costs and return pattern that 
    meet the max length constraint
    
    Input:
    - data : Data_Cs : data for the cutting stock problem
    - dual_variables : dual variables from the master problem
    
    Returns: 
    - Decision variables for the master problem
    
    (1): Minimize the reduced costs
    (2): Patterns should meet the length constraint
    """
    
    # OR-Tools model
    solver_pricing = pywraplp.Solver('SolveMIP_pricing', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    # Decisions variables
    y = [solver_pricing.IntVar(0, solver_pricing.infinity(), f"y({i})") for i in range(len(dual_variables))]
    
    # (1): Objective function 
    objective = solver_pricing.Objective()
    for i in range(len(dual_variables)):
        objective.SetCoefficient(y[i], dual_variables[i])  # SetCoefficient prend la variable et son coefficient
    objective.SetMaximization() 

    # (2) Constraints
    for i in range(len(dual_variables)):
        constraint = solver_pricing.Add(sum(data.pieces[i].w * y[i]  for i in range(len(dual_variables))) <= data.W)

    status = solver_pricing.Solve()
    number_of_rolls = objective.Value()
    
    if number_of_rolls > 1 + 1*10**(-8):
        return np.array([y[i].solution_value() for i in range(len(dual_variables))])
    
    else : 
        return None



def main_gc(data: data_generator.data_cs, nb_piece: int):
    """This function run the main algorithm for cutting stock problem.

    It contain 3 steps:
    --> generation of a feasible pattern to initialise the algorithm
    --> run master problem and sub problem until convergence (or max iter)
    --> re run the master problem without linear relaxation
    
    Inputs:
    - data: Data_CS : data for the cutting stock problem
    - nb_piece: int : number of pieces
    
    Returns:
    - master objective : Objective value after optimization part
    - solution vector : solution
    
    """
    pattern = generate_pattern(nb_piece, data)

    iter = 0
    while True :

        dual_variables, master_objective, solution_vector, status = master_problem(pattern,
                                                                                   data,
                                                                                   nb_piece)

        new_pattern = pricing_problem(data, dual_variables)

        try :
            if new_pattern == None:
                return master_objective, solution_vector
        except :
            iter +=1
            pass

        if iter >= 400:
            raise ValueError("GC method did not converge")

        pattern = np.vstack((pattern,new_pattern))

        # print(f"Found new pattern. Total patterns = {len(pattern)}")
        


    

