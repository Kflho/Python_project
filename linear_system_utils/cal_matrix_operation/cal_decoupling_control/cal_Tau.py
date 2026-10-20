"""解耦（耦合）矩阵 Tau。

系统 ẋ = A x + B u,  y = C x，第 i 个输出的相对阶为 r_i 时

    Tau 的第 i 行 = c_i A^(r_i-1) B          （c_i 是 C 的第 i 行）

规定：r_i 为 0（或相对阶不存在，即 None）时该行取零行——这时该输出与输入
没有耦合，Tau 的行列式为 0，静态状态反馈判为不能解耦。
"""
import sympy as sp
from linear_system_utils.cal_matrix_operation.cal_decoupling_control.cal_r import(
    cal_r
)
__all__ = ["cal_Tau"]


def cal_Tau(A, B, C):
    """
    解耦（耦合）矩阵 Tau，第 i 行是 c_i A^(r_i-1) B

    A : (n, n)
    B : (n, m)
    C : (p, n)，第 i 行是 c_i
    r : 长度 p 的相对阶列表，r[i] 对应 C 的第 i 行；
        某一项为 0 或 None 时 Tau 的对应行取零行
    返回: (p, m) 的 sympy 矩阵
    """
    A, B, C = sp.Matrix(A), sp.Matrix(B), sp.Matrix(C)
    r = cal_r(A, B, C)
    print("r =",r)
    if len(r) != C.rows:
        raise ValueError(f"r 的长度 {len(r)} 与 C 的行数 {C.rows}（输出个数）不一致")
    m = B.cols

    rows = []
    for i in range(C.rows):                 # 一行对应一个输出
        if r[i] in (0, None):               # 相对阶为 0 / 不存在 → 零行
            rows.append(sp.zeros(1, m))
        else:
            rows.append(C[i, :] * A**(r[i] - 1) * B)
    return sp.Matrix.vstack(*rows)




