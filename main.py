"""This code allows to run a benchmark of method for a cutting stock problem."""

from data_gen.data_generator import get_data
from optimization.cg import main_gc
from optimization.classic_milp import classic_MILP
import time


def main():
    """This function runs a benchmark over 2 differents method for a cutting stock problem."""
    
    nb_piece, largeur_max = 30, 1000
    data = get_data(nb_piece,largeur_max)

    t1 = time.time()
    master_objective, solution_vector = main_gc(data, nb_piece)
    t2 = time.time()
    print(t2-t1)
    master_objective, solution_vector = classic_MILP(data, nb_piece)
    t3 = time.time()
    print(t3-t2)


if __name__ == "__main__":
    main()