# Substitua o import numpy as np por:
import math
from typing import List, Tuple

# E crie uma matriz simples com list comprehension
def create_matrix(size):
    return [[0.0 for _ in range(size)] for _ in range(size)]