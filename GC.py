

import random
from dataclasses import dataclass, field
from typing import List
import numpy as np
from pyscipopt import Model, quicksum, multidict,SCIP_PARAMSETTING
from ortools.linear_solver import pywraplp

@dataclass
class Piece:
    w: float  # Largeur de la pièce
    d: int    # Demande pour cette pièce

@dataclass
class Data:
    pieces: List[Piece]  # Liste des pièces
    W: float             # Largeur maximale de la bobine

    def __str__(self):
        result = "Data for the cutting stock problem:\n"
        result += f"  W = {self.W}\n"
        result += "with pieces:\n"
        result += "   i   w_i d_i\n"
        result += "  ------------\n"
        for i, p in enumerate(self.pieces, start=1):
            result += f"{i:4} {p.w:5.1f} {p.d:3}\n"
        return result

def generate_raw_data(num_pieces):
    return [(random.uniform(5.0, 75.0), random.randint(20, 50)) for _ in range(num_pieces)]

def get_data(num_pieces=20, max_width=100.0):
    raw_data = generate_raw_data(num_pieces)
    pieces = [Piece(w=d[0], d=d[1]) for d in raw_data]
    return Data(pieces=pieces, W=max_width)

# Exemple d'utilisation

Num_piece, largeur_max = 23,100

data = get_data(Num_piece,largeur_max)
print(data)
# data = generate_raw_data(5)



# Commencer par générer un premier ensemble de pattern 
# Pattern est un matrice de taille (nb_pattern, nb_piece) ex pattern[1] = pour chaque piece le nombre de coupe possible dans le rouleau

pattern = np.zeros((Num_piece,Num_piece))
for i in range(Num_piece):
    pattern[i,i] = np.floor(data.W/data.pieces[i].w)




def master_problem(pattern,data ):


    # Modèle OR-Tools
    # solver = pywraplp.Solver.CreateSolver('SCIP')
    solver = pywraplp.Solver('SolveLP', pywraplp.Solver.CLP_LINEAR_PROGRAMMING )
    constraints = []


    # Variables
    x = [solver.NumVar(0.0, solver.infinity(), f"x({p})") for p in range(pattern.shape[0])]

    # Contraintes
    for i in range(Num_piece):
        constraint = solver.Add(sum(pattern[p,i]* x[p] for p in range(pattern.shape[0])) >= data.pieces[i].d)
        constraints.append(constraint)



    # Objectif
    objective = solver.Objective()
    for var in x:
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()

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
    
    
    solver_pricing = pywraplp.Solver('SolveMIP_pricing', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    y = [solver_pricing.IntVar(0, solver_pricing.infinity(), f"y({i})") for i in range(len(dual_variables))]
    
    for i in range(len(dual_variables)):
        constraint = solver_pricing.Add(sum(data.pieces[i].w * y[i]  for i in range(len(dual_variables))) <= data.W)

    objective = solver_pricing.Objective()
    for i in range(len(dual_variables)):
        objective.SetCoefficient(y[i], dual_variables[i])  # SetCoefficient prend la variable et son coefficient
    objective.SetMaximization() 
    status = solver_pricing.Solve()
    number_of_rolls = objective.Value()
    
            
    if number_of_rolls > 1 + 1*10**(-8):
        return np.array([y[i].solution_value() for i in range(len(dual_variables))])
    
    else : 
        return None
    
    
    
while True :
    
    dual_variables, master_objective, solution_vector, status = master_problem(pattern,data)
    
        
    new_pattern = pricing_problem(data, dual_variables)
    
    
    try :
        if new_pattern == None:
            print("No new pattern generated")
            print('Objectif function : ')
            print(master_objective)
            print('Solution')
            print(solution_vector)
            break
    except :
        pass

    pattern = np.vstack((pattern,new_pattern))

    print(f"Found new pattern. Total patterns = {len(pattern)}")
