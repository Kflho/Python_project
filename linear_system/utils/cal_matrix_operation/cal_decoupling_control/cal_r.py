"""MIMO 线性系统各输出的相对阶（按传递函数矩阵的定义算）。

系统 ẋ = A x + B u,  y = C x，传递函数矩阵

    G(s) = C (sI - A)^-1 B          (p x m)

把第 (i,j) 项写成既约分式 G_ij(s) = n_ij(s) / d_ij(s)，它对第 j 个输入的相对阶是

    r_ij = deg d_ij - deg n_ij       （分母次数 - 分子次数，也就是 s = ∞ 处极点的阶）

第 i 个输出（第 i 行）的相对阶取这一行各通道的最小值：

    r_i = min_j r_ij

G_ij(s) 恒为 0（该通道完全解耦）时 r_ij = ∞，不参与取最小；整行都恒为 0 时 r_i 不存在。

顺带一个等价说法：c_i A^k b_j 就是 G_ij(s) 在 s = ∞ 处展开的马尔可夫参数，
第一个非零的指标加 1 恰好等于 deg d_ij - deg n_ij，所以从状态空间逐拍扫
c_i A^(k-1) B 也能得到同样的 r。两条路都不需要先知道 r，谈不上鸡生蛋的问题；
这里按传递函数矩阵的定义写。
"""
import sympy as sp

__all__ = ["cal_r"]


def cal_r(A, B, C):
    """
    各输出的相对阶 r_i = min_j ( deg d_ij - deg n_ij )

    A : (n, n)
    B : (n, m)
    C : (p, n)
    返回: 长度 p 的列表，第 i 项是 C 第 i 行那个输出的相对阶；
          该行与所有输入都解耦（整行恒为零）时该项为 None

    算出来的 r 正好能喂给 is_decoupling_system / cal_decoupling_control。
    """
    A, B, C = sp.Matrix(A), sp.Matrix(B), sp.Matrix(C)
    s = sp.Symbol("s")
    G = C * (s * sp.eye(A.rows) - A).inv() * B

    r = []
    for i in range(G.rows):
        r_row = []
        for j in range(G.cols):
            num, den = sp.fraction(sp.cancel(G[i, j]))      # 先约简，再取分子分母
            r_row.append(sp.degree(den, s) - sp.degree(num, s))
        r_min = min(r_row)                                   # 每行取最小值
        r.append(None if r_min == sp.oo else r_min)          # 整行为零 → 相对阶不存在
    return r
