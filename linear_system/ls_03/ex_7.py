import scipy.linalg as la
import numpy as np
import sympy as sp
A = np.array([[-1, 1], 
    [2, -3]])
Q = np.eye(2)
print(Q)
P = la.solve_continuous_lyapunov(A, -Q)
print(P)
P_sym = sp.Matrix(P)
print(P_sym.is_positive_definite)
