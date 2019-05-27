from timeit import timeit

setup = """
import numpy as np
from numba import njit

@njit
def create_diff_matrix(p):
    D = np.full((p, p), fill_value=-1.0, dtype=np.float_)
    # np.fill_diagonal(D, np.float_(p-1.0))
    for i in range(p):
        D[i, i] = p - 1.0
    return D


lam = 10000
n = 100
p = 3
D = create_diff_matrix(p)
X = np.random.normal(size=(n, p))
true_beta = np.random.normal(size=(p, 1))
eps = np.random.normal(loc=0, scale=0.1, size=(n, 1))
y = X @ true_beta + eps


def stew_reg(X, y, D, lam):
    return np.linalg.inv(X.T @ X + lam * D) @ X.T @ y

@njit
def stew_reg_n(X, y, D, lam):
    return np.linalg.inv(X.T @ X + lam * D) @ X.T @ y
    
a = stew_reg(X, y, D, lam)
b = stew_reg_n(X, y, D, lam)

"""
n = 100000
print(timeit(stmt="stew_reg(X, y, D, lam)", setup=setup, number=n))
print(timeit(stmt="stew_reg_n(X, y, D, lam)", setup=setup, number=n))

