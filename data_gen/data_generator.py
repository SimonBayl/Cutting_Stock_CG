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
    return [(random.uniform(5.0, 75.0), random.randint(20, 50)) for _ in range(num_pieces)]

def get_data(num_pieces=20, max_width=100.0):
    raw_data = generate_raw_data(num_pieces)
    pieces = [Piece(w=d[0], d=d[1]) for d in raw_data]
    return data_cs(pieces=pieces, W=max_width)
