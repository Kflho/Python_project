import sympy as sp
from utils.cal_matrix_operation.cal_pole_placement.cal_state_feedback_pole_placement import (
    cal_state_feedback_pole_placement,
)
from utils.cal_matrix_parameters.cal_set_of_controllability_indices import (
    cal_set_of_controllability_indices,
)
from utils.cal_matrix_operation.cal_controllable_canonical_form import (
    cal_Q,
    cal_p,
    cal_Tau,
)
A = sp.Matrix([
    [0, 2, 0, 0],
    [0, 0, 1, 0],
    [-3, 1, 2, 3],
    [2, 1, 0, 0]
])
B = sp.Matrix([
    [0, 1],
    [0, 0],
    [1, 2],
    [0, 2]
]
)
poles = (-1 + sp.I, -1 - sp.I, -1 + sp.sqrt(2) * sp.I, -1 - sp.sqrt(2) * sp.I)
cal_state_feedback_pole_placement(A, B, poles)

nu = cal_set_of_controllability_indices(A, B)
Q = cal_Q(A, B, nu)
ps = sp.Matrix(cal_p(Q, nu))          # p_i 取 Q^-1 的第 nu_1+...+nu_i 行
Tau = sp.Matrix(cal_Tau(nu, A, B))
n = A.rows

K_formula = -Tau.inv() * sp.Matrix.vstack(
    ps[0, :] * (A + (1 + sp.I) * sp.eye(n)) * (A + (1 - sp.I) * sp.eye(n)),
    ps[1, :] * (A + (1 - sp.sqrt(2) * sp.I) * sp.eye(n)) * (A + (1 + sp.sqrt(2) * sp.I) * sp.eye(n)),
)
print("K_formula =", K_formula.applyfunc(sp.nsimplify))