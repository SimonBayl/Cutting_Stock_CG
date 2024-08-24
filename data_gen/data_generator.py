"""This file create random data for a cutting stock problem"""

from dataclasses import dataclass
from typing import List
import random


@dataclass
class Piece:
    w: float  # Largeur de la pièce
    d: int    # Demande pour cette pièce

@dataclass
class data_cs:
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
    return [(random.uniform(5.0, 40.0), random.randint(20, 35)) for _ in range(num_pieces)]

    # data = [
    #    (75.0, 38)
    #     (75.0, 44)
    #     (75.0, 30)
    #     (75.0, 41)
    #     (75.0, 36)
    #     (53.8, 33)
    #     (53.0, 36)
    #     (51.0, 41)
    #     (50.2, 35)
    #     (32.2, 37)
    #     (30.8, 44)
    #     (29.8, 49)
    #     (20.1, 37)
    #     (16.2, 36)
    #     (14.5, 42)
    #     (11.0, 33)
    #     (8.6, 47)
    #     (8.2, 35)
    #     (6.6, 49)
    #     (5.1, 42)
    # ]
        
    # return data

def get_data(num_pieces=20, max_width=100.0):
    raw_data = generate_raw_data(num_pieces)
    pieces = [Piece(w=d[0], d=d[1]) for d in raw_data]
    return data_cs(pieces=pieces, W=max_width)
