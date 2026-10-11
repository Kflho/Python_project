import sympy as sp
from utils.cal_matrix_operation.cal_optimal_control.cal_algebra_Riccati_function import (
    cal_algebra_Riccati_function
)
omega = sp.symbols('omega')
A = sp.Matrix([
    [0, omega],
    [-omega, 0]
])
B = sp.Matrix([
    [0],
    [1]
])
Q = sp.Matrix([
    [1, 0],
    [0, 1]
])
R = sp.Matrix([
    [1]
])
cal_algebra_Riccati_function(A, B, None, Q, R)
