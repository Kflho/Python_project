"""状态反馈极点配置（课件 4.4 的极点配置公式）。

求 K 使 lambda(A + B*K) = poles，反馈取 u = K x，闭环矩阵为 A + B K。

    K = -Tau^{-1}(A,B) [p_1 A^{nu_1}; ...; p_m A^{nu_m}]
        + Tau^{-1}(A,B) [p_2; ...; p_m; gamma_star * T]

其中

    Q          = [Q_1 ... Q_m],  Q_i = [b_i, A b_i, ..., A^{nu_i-1} b_i]
    p_i        = Q^{-1} 的第 (nu_1 + ... + nu_i) 行
    Tau(A,B)   = [p_1 A^{nu_1-1} B; ...; p_m A^{nu_m-1} B]      (m x m，可逆)
    T          = [p_1; p_1 A; ...; p_1 A^{nu_1-1}; ...; p_m A^{nu_m-1}]   (n x n)
    gamma_star = -[gamma_0, gamma_1, ..., gamma_{n-1}]  (1 x n)
                 gamma(s) = prod(s - poles) = s^n + gamma_{n-1} s^{n-1} + ... + gamma_0

注意最后一项是 gamma_star * T（系数行向量乘坐标变换阵），不是 gamma_star 本身。
"""
import numpy as np
import sympy

from ...cal_matrix_parameters.cal_set_of_controllability_indices import (
    cal_set_of_controllability_indices,
)
from ..cal_controllable_canonical_form import cal_Q, cal_p, cal_T, cal_Tau

__all__ = ["cal_gamma_star", "cal_state_feedback_pole_placement"]


def cal_gamma_star(poles):
    """
    期望极点对应的首一特征多项式

        gamma(s) = (s - p_1)(s - p_2)...(s - p_n)
                 = s^n + gamma_(n-1) s^(n-1) + ... + gamma_1 s + gamma_0

    poles: 长度 n 的极点（实数、复数、sympy 表达式都行）
    返回: (1, n) 的 sympy 行向量 -[gamma_0, gamma_1, ..., gamma_(n-1)]，系数按 0..n-1 排

    能控标准型里它是拿来填对角块最后一行的：

        A_c 第 i 块的末行 = [gamma_0, gamma_1, ..., gamma_(nu_i-1)]
    """
    s = sympy.Symbol('s')
    poly = sympy.prod([s - sympy.sympify(p) for p in poles])
    coeffs = sympy.Poly(poly, s).all_coeffs()        # [1, gamma_(n-1), ..., gamma_1, gamma_0]
    return -sympy.Matrix([list(reversed(coeffs[1:]))])  # 去掉首项 1，再倒序成 0..n-1（行向量）


def _as_float(M):
    """sympy 矩阵 / numpy 数组 / 嵌套列表统一成 float 的 numpy 数组"""
    if isinstance(M, sympy.MatrixBase):
        return np.array(M.tolist(), dtype=float)
    return np.asarray(M, dtype=float)


def cal_state_feedback_pole_placement(A, B, poles, verbose=False):
    """
    状态反馈极点配置：求 K 使 lambda(A + B*K) = poles

    A : (n, n)
    B : (n, m)，第 i 列是 b_i（m 路输入）
    poles: 长度 n 的期望极点（共轭成对；重根不超过 rank(B) 即可）
    verbose: True 时打印 nu / Q / T / Tau / K

    返回
        K    : (m, n) numpy 数组，反馈 u = K x
        info : dict，含中间量 nu, Q, p, T, Tau, gamma_star, pa, S

    (A,B) 不能控时抛 ValueError（来自 cal_set_of_controllability_indices）。
    """
    A = _as_float(A)
    B = _as_float(B)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    n, m = A.shape[0], B.shape[1]

    nu = cal_set_of_controllability_indices(A, B)
    gamma_star = _as_float(cal_gamma_star(poles)).ravel()

    Q = cal_Q(A, B, nu)
    p = cal_p(Q, nu)
    T = cal_T(nu, A, B)
    Tau = cal_Tau(nu, A, B)

    # 第一个矩阵：[p_1 A^{nu_1}; ...; p_m A^{nu_m}]，m 行
    pa = np.vstack([p[i] @ np.linalg.matrix_power(A, nu[i]) for i in range(m)])
    # 第二个矩阵：[p_2; ...; p_m; gamma_star * T]，m 行
    S = np.vstack([p[1:, :], gamma_star @ T])

    K = np.linalg.inv(Tau) @ (S - pa)

    info = {
        "nu": list(nu),
        "Q": Q,
        "p": p,
        "T": T,
        "Tau": Tau,
        "gamma_star": gamma_star,
        "pa": pa,
        "S": S,
    }
    if verbose:
        print("nu =", nu)
        print("gamma_star =", gamma_star)
        print("Q =\n", Q)
        print("T =\n", T)
        print("Tau =\n", Tau)
        print("K =\n", K)
    return K, info
