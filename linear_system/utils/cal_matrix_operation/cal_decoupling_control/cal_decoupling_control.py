import sympy as sp

from utils.cal_matrix_operation.cal_decoupling_control.cal_r import (
    cal_r
)
from utils.cal_matrix_operation.cal_decoupling_control.cal_Tau import (
    cal_Tau
)
def cal_decoupling_control(A, B, C, gamma):
    Tau = cal_Tau(A, B, C)
    print("Tau = ",Tau)
    if is_decoupling_system(Tau):
        print("is decoupling system")
        Phi = sp.Matrix.vstack(*[C[i, :] * cal_gamma_A(gamma[i], A) for i in range(C.rows)])
        K = -Tau.inv() * Phi
        G = Tau.inv()                       # 控制律 u = K x + G v
        return K, G
    else: print("isn't decoupling system")


def is_decoupling_system(Tau):
    if Tau.rows != Tau.cols:                # 不是方阵（输出数 != 输入数）就没法静态解耦
        return False
    if Tau.det() != 0:
        return True
    return False


def cal_gamma_A(gamma_i, A, s=sp.Symbol("s")):
    """
    多项式代入矩阵：gamma(A) = c0*I + c1*A + c2*A^2 + ...

    gamma_i: sympy 多项式（默认变元 s，用别的字母就传 s=...）
    A      : (n, n) 矩阵
    返回   : (n, n) 矩阵 c0*I + c1*A + ... + cr*A^r
    """
    gamma_i = sp.sympify(gamma_i)
    poly = sp.Poly(gamma_i, s)
    if poly.degree() == 0 and gamma_i.free_symbols:
        raise ValueError(f"gamma 里没有变元 {s}，变元名不是 s 的话传 s=... 指定")
    A = sp.Matrix(A)
    val = sp.zeros(A.rows, A.cols)
    for k, c in enumerate(poly.all_coeffs()[::-1]):     # 系数取成低次 -> 高次
        val += c * A**k
    return val


    