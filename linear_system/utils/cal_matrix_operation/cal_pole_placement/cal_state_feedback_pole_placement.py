import sympy 
from utils.cal_matrix_parameters.cal_set_of_controllability_indices import (
    cal_set_of_controllability_indices,
)
from utils.cal_matrix_operation.cal_controllable_canonical_form import (
    cal_Q,
    cal_p,
    cal_T,
    cal_Tau
)
def cal_gamma_star(poles):
    """
    期望极点对应的首一特征多项式

        gamma(s) = (s - p_1)(s - p_2)...(s - p_n)
                 = s^n + gamma_(n-1) s^(n-1) + ... + gamma_1 s + gamma_0

    poles: 长度 n 的极点（实数、复数、sympy 表达式都行）
    返回: (1, n) 的 sympy 行向量 [gamma_0, gamma_1, ..., gamma_(n-1)]，系数从 0 到 n-1

    能控标准型里它是拿来填对角块最后一行的：

        A_c 第 i 块的末行 = -[gamma_0, gamma_1, ..., gamma_(nu_i-1)]
    """
    s = sympy.Symbol('s')
    poly = sympy.prod([s - sympy.sympify(p) for p in poles])
    coeffs = sympy.Poly(poly, s).all_coeffs()        # [1, gamma_(n-1), ..., gamma_1, gamma_0]
    return -sympy.Matrix([list(reversed(coeffs[1:]))])  # 去掉首项 1，再倒序成 0..n-1（行向量）
    
def cal_state_feedback_pole_placement(A, B, poles):
    print(A.eigenvals())
    m = B.cols
    nu = cal_set_of_controllability_indices(A, B)
    print("nu =",nu)
    gamma_star = cal_gamma_star(poles)
    print("gamma_star = ", gamma_star)
    Q = cal_Q(A, B, nu)
    print("Q =", Q)
    p = cal_p(Q, nu)
    print("p = ", p)
    T = cal_T(nu, A, B)
    print("T = ", T)
    Tau = cal_Tau(nu, A, B)
    ps = sympy.Matrix(p)                                  # cal_p 给的是 numpy 数组，转 sympy
    pa = sympy.Matrix.vstack(*[ps[i, :] * A ** nu[i] for i in range(m)])   # 第一个：p_i A^nu_i，m 行
    S = sympy.Matrix.vstack(ps[1:, :], gamma_star * sympy.Matrix(T))       # 第二个：p_2..p_m 再加最后一行 gamma_star*T，也是 m 行
    K = -sympy.Matrix(Tau).inv() * pa + sympy.Matrix(Tau).inv() * S
    print("K = ", K)

    