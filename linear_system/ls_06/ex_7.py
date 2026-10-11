import sympy as sp
from utils.cal_matrix_operation.cal_decoupling_control.cal_decoupling_control import(
    cal_decoupling_control
)
A = sp.Matrix([
    [0, 1, 1, 0],
    [3, 0, 0, 2],
    [0, 0, 0, 1],
    [0, -2, 0, 0]
])
B = sp.Matrix([
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 1]
])
C = sp.Matrix([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
s = sp.symbols('s')
gamma = [s**2 + 2*s + 2, s**2 + 2*s + 3]
K, G = cal_decoupling_control(A, B, C, gamma)
print("K =", K)
print("G =", G)