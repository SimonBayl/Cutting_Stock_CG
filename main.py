"""This code allows to run a benchmark of method for a cutting stock problem."""

from data_gen.data_generator import get_data
from optimization.cg import branch_and_price
from optimization.classic_milp import classic_MILP
import time
import csv

def compute_n_compare():
    

    rouleaux_range = [x+4 for x in range(100)]
    pieces_range = [x for x in range(100)]
    largeur_max = 1000

    results = []
    nb_experience = len(rouleaux_range)


    for index in range(nb_experience):
        data = get_data(pieces_range[index], largeur_max)

        t1 = time.time()
        master_objective_bp, solution_vector_bp = branch_and_price(data, pieces_range[index], rouleaux_range[index])
        t2 = time.time()
        if master_objective_bp  == None and solution_vector_bp == None:
            time_bp = None
        else:
            time_bp = t2 - t1

        t3 = time.time()
        master_objective_milp, solution_vector_milp = classic_MILP(data, pieces_range[index], rouleaux_range[index])
        t4 = time.time()
        if master_objective_milp == None and solution_vector_milp == None:
            time_milp = None
        else:
            time_milp = t4 - t3
            
            
        results.append({
            'nb_rouleaux': rouleaux_range[index],
            'nb_piece': pieces_range[index],
            'time_branch_and_price': time_bp,
            'time_classic_MILP': time_milp,
            'objective_branch_and_price': master_objective_bp,
            'objective_classic_MILP': master_objective_milp
        })

    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['nb_rouleaux', 'nb_piece', 'time_branch_and_price', 'time_classic_MILP',
                      'objective_branch_and_price', 'objective_classic_MILP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("Les résultats ont été sauvegardés dans benchmark_results.csv")


def main():
    """This function runs a benchmark over 2 differents method for a cutting stock problem."""
    
    
    # compute_n_compare()

    nb_rouleaux = 10
    nb_piece, largeur_max = 8, 1000
    data = get_data(nb_piece,largeur_max) # demande_max = 35*nb_pièces*40 <= nb_rouleau*largeur_max



    t1 = time.time()
    master_objective, solution_vector = branch_and_price(data, nb_piece, nb_rouleaux)
    t2 = time.time()
    print(t2-t1)
    print(solution_vector)
    master_objective, solution_vector = classic_MILP(data, nb_piece, nb_rouleaux)
    t3 = time.time()
    print(t3-t2)


if __name__ == "__main__":
    main()