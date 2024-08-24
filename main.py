"""This code allows to run a benchmark of method for a cutting stock problem."""

from data_gen.data_generator import get_data
from optimization.cg import main_gc
from optimization.classic_milp import classic_MILP
import time


def main():
    """This function runs a benchmark over 2 differents method for a cutting stock problem."""

    nb_rouleaux = 1000
    nb_piece, largeur_max = 20, 1000
    data = get_data(nb_piece,largeur_max) # demande_max = 35*nb_pièces*40 <= nb_rouleau*largeur_max

    results_milp = []
    results_gc = []

    t1 = time.time()
    master_objective, solution_vector = main_gc(data, nb_piece, nb_rouleaux)
    t2 = time.time()
    print(t2-t1)
    print(solution_vector)
    master_objective, solution_vector = classic_MILP(data, nb_piece, nb_rouleaux)
    t3 = time.time()
    print(t3-t2)
    # print(solution_vector)


if __name__ == "__main__":
    main()