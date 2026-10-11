import sympy as sp
from utils.cal_matrix_operation.cal_decoupling_control.cal_decoupling_control import(
    cal_Tau
)
from utils.cal_matrix_operation.cal_decoupling_control.cal_decoupling_control import(
    is_decoupling_system
)
A = sp.Matrix([
    [0, 1, -1, 2, -1, -1],
    [0, -1, 0, -1, 0, 1],
    [0, 1, 0, 2, 0, -1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, -1, 0, -1, 0, 1]
])
B = sp.Matrix([
    [1, 0, 0],
    [0, -1, 0],
    [1, 1, 0],
    [0, 1, 1],
    [-1, 0, 1],
    [0, 1, 1]
])
C_1 = sp.Matrix([
    [0, 1, 0, 2, -1, -1],
    [0, 1, 0, 1, -1, 0],
    [-1, 2, 0, 4, -2, -2]
])
C_2 = sp.Matrix([
    [0, 1, 0, 2, -1, -1],
    [0, 0, 0, -1, 0, 1],
    [-1, 1, 0, 3, -1, -2]
])
is_decoupling_system(cal_Tau(A, B, C_1))
is_decoupling_system(cal_Tau(A, B, C_2))
