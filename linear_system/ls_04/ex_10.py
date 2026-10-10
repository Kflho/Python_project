from utils.cal_matrix_parameters.cal_set_of_controllability_indices import(
    cal_set_of_controllability_indices
)
import sympy as sp
A = sp.Matrix([
    [-1, -1, 3, 4],
    [-2, -0, 2, 3],
    [-2, 1, -1, -1],
    [0, -1, 2, 3]
])
B = sp.Matrix([
    [1, 0],
    [1, -1],
    [1, -2],
    [0,1]
])
nu = list(cal_set_of_controllability_indices(A, B))
print("nu = ",nu)
