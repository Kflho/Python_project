import numpy as np
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

# 4.4 极点配置公式，K 直接返回
K, info = cal_state_feedback_pole_placement(A, B, poles)
print("K =\n", np.array2string(K, precision=6))

# 另一种写法：用 (A - lambda_i I) 的乘积直接构造
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

# 两组 K 是否都命中同一组极点：比较闭环特征多项式（比直接比较 K 更稳）
Af = np.array(A.tolist(), dtype=float)
Bf = np.array(B.tolist(), dtype=float)
Kf = np.array(K_formula.tolist(), dtype=complex)
poly_want = np.poly([complex(p) for p in poles])
poly_K = np.poly(np.linalg.eigvals(Af + Bf @ K))
poly_Kf = np.poly(np.linalg.eigvals(Af + Bf @ Kf))
print("闭环特征多项式偏差  K  :", np.max(np.abs(poly_K.real - poly_want.real)))
print("闭环特征多项式偏差  Kf :", np.max(np.abs(poly_Kf.real - poly_want.real)))
print("||K - Kf|| =", np.linalg.norm(K - Kf), "（解不唯一，但极点相同）")
