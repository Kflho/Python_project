"""多输入系统的能控标准型（controllable canonical form）。

    A_c = T A T^-1,   B_c = T B

其中 T 由能控性矩阵 Q 的逆矩阵的特定行 p_i 生成：

    Q    = [b_1, A b_1, ..., A^(nu_1-1) b_1 | b_2, ..., A^(nu_m-1) b_m]
    p_i  = Q^-1 的第 (nu_1 + ... + nu_i) 行
    T    = [p_1; p_1 A; ...; p_1 A^(nu_1-1) | ... | p_m; ...; p_m A^(nu_m-1)]

变换后：

  * A_c 是分块形式，对角上的第 i 块是该块指标 nu_i 对应的能控标准型块
    （块内首行右移一位、末行放特征多项式系数）；非对角块一般不为零，
    取值由块间正交关系决定。
  * B_c 只有第 nu_1+...+nu_i 行非零，这些行组成的矩阵就是 Tau：

        Tau[i] = p_i A^(nu_i-1) B     -> B_c 的第 nu_1+...+nu_i 行

注意上面这条对应关系的前提：nu 必须是这组 (A, B) 真正的能控性指标。
若 nu 是随便给的（(A, B) 的指标其实是别的值），T 就不是合法的坐标变换，
此时 A_c 不成标准型、B_c 的非零行也不落在 s_i 上，而 B_c = T B、
A_c = T A T^-1 仍然无条件成立。所以 B_c 一律按 T B 算，Tau 单独算，
两者是否吻合留给调用方当检验用。
"""
import numpy as np
from numpy.linalg import inv

from .cal_Q_c import cal_Q_c

__all__ = ["cal_Q", "cal_p", "cal_T", "cal_Tau", "cal_A_c_and_B_c", "cal_A_new_and_B_new"]


def cal_Q(A, B, nu):
    """
    把 B 的每一列 b_i 的能控性矩阵 Q_c_i 从左到右拼接成 Q

        Q = [b_1, A b_1, ..., A^(nu_1-1) b_1 | b_2, A b_2, ..., A^(nu_2-1) b_2 | ...]

    A : (n, n)
    B : (n, m)，第 i 列是 b_i
    nu: 长度为 m 的能控性指标，nu[i] 是 b_i 贡献的列数
    返回: (n, n)，n = nu_1 + ... + nu_m
    """
    cols = []
    for i in range(B.shape[1]):
        cols.append(cal_Q_c(A, B[:, i], nu[i]))
    return np.hstack(cols)


def cal_p(Q, nu):
    """
    p_i 取 Q 的逆矩阵的第 (nu_1 + ... + nu_i) 行

    Q : (n, n)
    nu: 长度为 m 的能控性指标
    返回: (m, n)，第 i 行是 p_i
    """
    Q_inv = inv(Q)
    p = []
    s = 0
    for i in range(len(nu)):
        s += nu[i]
        p.append(Q_inv[s - 1, :])        # 行号 s 是 1-based，下标要减 1
    return np.vstack(p)


def cal_T(nu, A, B):
    """
    T_i = [p_i; p_i A; ...; p_i A^(nu_i-1)]，T 由各 T_i 垂直拼接

    nu: 长度为 m 的能控性指标
    A : (n, n)
    B : (n, m)
    返回: (n, n)
    """
    p = cal_p(cal_Q(A, B, nu), nu)
    blocks = []
    for i in range(len(nu)):
        rows = [p[i]]
        for _ in range(nu[i] - 1):
            rows.append(rows[-1] @ A)
        blocks.append(np.vstack(rows))
    return np.vstack(blocks)


def cal_Tau(nu, A, B):
    """
    Tau 的第 i 行是 p_i A^(nu_i-1) B，共 m 行

        Tau[i] = p_i A^(nu_i-1) B : (1, m)

    A : (n, n)
    B : (n, m)
    nu: 长度为 m 的能控性指标
    返回: (m, m)，第 i 行是 p_i A^(nu_i-1) B

    这些行就是 B_c 的非零行（前提是 nu 为这组 (A, B) 真正的能控性指标），
    见 cal_A_c_and_B_c。
    """
    p = cal_p(cal_Q(A, B, nu), nu)
    rows = []
    for i in range(len(nu)):
        rows.append(p[i] @ np.linalg.matrix_power(A, nu[i] - 1) @ B)
    return np.vstack(rows)


def cal_A_c_and_B_c(nu, A, B):
    """
    多输入系统的能控标准型 A_c = T A T^-1，B_c = T B

    nu: 长度为 m 的能控性指标
    A : (n, n)
    B : (n, m)
    返回: (A_c, B_c)
        A_c : (n, n)  对角块为能控标准型块
        B_c : (n, m)  第 nu_1+...+nu_i 行非零，就是 Tau 的各行
    """
    T = cal_T(nu, A, B)
    A_c = T @ A @ inv(T)
    B_c = T @ B
    return A_c, B_c


def cal_A_new_and_B_new(A, B, T):
    """相似变换 A_new = T A T^-1，B_new = T B"""
    T_inv = inv(T)
    A_new = T @ A @ T_inv
    B_new = T @ B
    return A_new, B_new
