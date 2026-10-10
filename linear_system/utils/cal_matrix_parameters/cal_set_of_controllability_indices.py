"""多输入对 (A, B) 的能控性指标集 nu = [nu_1, nu_2, ..., nu_m]。

B 的第 i 列 b_i 贡献 nu_i 列，使

    Q = [b_1, A b_1, ..., A^(nu_1-1) b_1 | ... | b_m, A b_m, ..., A^(nu_m-1) b_m]

成为 n x n 的非奇异矩阵（就是 cal_controllable_canonical_form 里的 Q），
于是 A_c = T A T^-1 的第 i 个对角块是 nu_i 阶能控标准型块，
B_c = T B 只有第 nu_1+...+nu_i 行非零，这些行组成 Tau。

求法（Luenberger 能控标准型里指标的定义）：把能控性矩阵

    Q_c = [B, A B, ..., A^(n-1) B] = [b_1, ..., b_m | A b_1, ..., A b_m | ...]

按从左到右的顺序逐列考察，凡是跟已经选出的列线性无关的就留下
（同一层 A^k B 内部按 b_1, ..., b_m 的顺序）。b_i 那一族 b_i, A b_i, A^2 b_i, ...
中被选中的列数就是 nu_i，而它们正好是 b_i, A b_i, ..., A^(nu_i-1) b_i 这个前缀。

逐层考察等价于：第 k 层选出的列数是

    #{i : nu_i >= k} = rank[B, ..., A^(k-1) B] - rank[B, ..., A^(k-2) B]

注意 nu_i 对应的是给定的 B 的第 i 列，所以 nu 本身不一定按大小排好
（例如 nu = [1, 2] 是可能的）；要得到递降的 nu_1 >= nu_2 >= ... >= nu_m，
得先把 B 的列按合适顺序重排再算。另外当 B 的列不满秩时会出现 nu_i = 0，
此时 sum(nu) 仍可能等于 n（系统能控），但能控标准型的构造要求每一块都非空，
即要求 B 列满秩。

(A, B) 不能控时 sum(nu) < n，不存在能控标准型，函数直接报错。
"""
import numpy as np

__all__ = ["cal_set_of_controllability_indices"]


def cal_set_of_controllability_indices(A, B):
    """
    能控性指标集 nu，见模块开头的说明

    A : (n, n)
    B : (n, m)，第 i 列是 b_i
    返回: 长度为 m 的列表 [nu_1, ..., nu_m]，第 i 项对应 B 的第 i 列，sum(nu) = n

    (A, B) 不能控时抛 ValueError（此时不存在能控标准型）。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    n, m = A.shape[0], B.shape[1]

    basis = np.zeros((n, 0))        # 已经选中的列
    nu = [0] * m
    layer = B                       # 第 k 层就是 A^k B
    for _ in range(n):              # 能控性矩阵 Q_c 一共 n 层
        for i in range(m):
            col = layer[:, [i]]
            if np.linalg.matrix_rank(np.hstack([basis, col])) > basis.shape[1]:
                basis = np.hstack([basis, col])
                nu[i] += 1
        if basis.shape[1] == n:     # 已经选满 n 个无关列，剩下的必然相关
            break
        layer = A @ layer

    if sum(nu) != n:
        raise ValueError(
            f"(A, B) 不能控：nu = {nu}，sum(nu) = {sum(nu)} != n = {n}，"
            "不存在能控标准型（Q 不可逆）"
        )
    return nu
