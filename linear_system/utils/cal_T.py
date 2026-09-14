import numpy as np
from numpy.linalg import inv

from .cal_Q_c import cal_Q_c


def cal_Q(A, B, nu):
    """
    把 B 的每一列 b_i 的能控性矩阵 Q_c_i 从左到右拼接成 Q

        Q = [b_1, A b_1, ..., A^(nu_1-1) b_1 | b_2, A b_2, ..., A^(nu_2-1) b_2 | ...]

    A : (n, n)
    B : (n, m)，第 i 列是 b_i
    nu: 长度为 m 的能控性指标，nu[i] 是 b_i 贡献的列数
    返回: (n, nu_1 + ... + nu_m)
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
    Q = cal_Q(A, B, nu)
    p = cal_p(Q, nu)
    blocks = []
    for i in range(len(nu)):
        rows = [p[i]]
        for _ in range(nu[i] - 1):
            rows.append(rows[-1] @ A)
        blocks.append(np.vstack(rows))
    return np.vstack(blocks)


def cal_A_new_and_B_new(A, B, T):
    """相似变换 A_new = T A T^-1，B_new = T B"""
    T_inv = inv(T)
    A_new = T @ A @ T_inv
    B_new = T @ B
    return A_new, B_new
