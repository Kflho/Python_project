import numpy as np


def cal_Q_c(A, b, nu):
    """
    单个输入的能控性矩阵

        Q_c = [b, A b, A^2 b, ..., A^(nu-1) b]

    A : (n, n)
    b : (n, 1) 或 (n,)，第 i 个输入列 b_i
    nu: 该输入的能控性指标，即 Q_c 的列数
    返回: (n, nu)
    """
    b = np.asarray(b).reshape(-1, 1)
    cols = [b]
    for _ in range(nu - 1):
        cols.append(A @ cols[-1])
    return np.hstack(cols)
