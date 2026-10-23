"""轨迹跟踪控制律（课件第 7 章）。

系统 ẋ = A x + B u,  y = C x，各输出的相对阶为 r_i，耦合矩阵

    Gamma(A,B,C) 的第 i 行 = c_i A^(r_i-1) B

非奇异、系统最小相位、y_i^* 至少 r_i 次可微时，定理 7.1.1 给出控制律（课件 (7.10)）

    u = Gamma^{-1}(A,B,C) [ -c_1 gamma_1(A) x + v_1^* ; ... ; -c_m gamma_m(A) x + v_m^* ]

其中 gamma_i(s) 是任意 r_i 次 Hurwitz 多项式

    gamma_i(s) = s^{r_i} + gamma_i,r_i-1 s^{r_i-1} + ... + gamma_i,0

v_i^* 是把期望输出的各阶导数（含 r_i 阶）按同一个多项式组合起来的

    v_i^* = y_i^{*(r_i)} + gamma_i,r_i-1 y_i^{*(r_i-1)} + ... + gamma_i,0 y_i^*

于是控制律写成"状态反馈 + 参考前馈"的两自由度形式

    u = K x + Gamma^{-1} v^*,   K = -Gamma^{-1} Phi,  Phi 的第 i 行 = c_i gamma_i(A)

这与第 6 章解耦控制的 u = K x + G v 是同一个式子，只是 v 换成了跟踪所需的 v^*。
本模块只给设计和参考信号，仿真闭环在练习脚本里做。
"""
import sympy as sp

from linear_system_utils.cal_matrix_operation.cal_decoupling_control.cal_r import (
    cal_r,
)

__all__ = [
    "cal_coupling_matrix",
    "cal_tracking_control",
    "cal_reference_vector",
    "cal_error_dynamics",
]


def cal_coupling_matrix(A, B, C, r=None):
    """耦合矩阵 Gamma 与各输出的相对阶 r。

    A : (n, n)，B : (n, m)，C : (p, n)
    r : 可选的相对阶列表；默认按 cal_r（传递函数矩阵 deg 分母 - deg 分子）算

    返回 (r, Gamma)：
        r     : 长度 p 的列表，r[i] 是第 i 个输出的相对阶
        Gamma : (p, m)，第 i 行 = c_i A^(r_i-1) B

    某个输出与所有输入都解耦（r_i 不存在）时抛 ValueError——这种输出没法用
    静态状态反馈跟踪，要先改输出。
    """
    A, B, C = sp.Matrix(A), sp.Matrix(B), sp.Matrix(C)
    if r is None:
        r = cal_r(A, B, C)
    r = list(r)
    if len(r) != C.rows:
        raise ValueError(f"r 的长度 {len(r)} 与输出个数 {C.rows} 不一致")
    if any(ri is None for ri in r):
        raise ValueError(f"存在相对阶不存在的输出（r = {r}），无法设计跟踪律")
    rows = [C[i, :] * A ** (r[i] - 1) * B for i in range(C.rows)]
    return r, sp.Matrix.vstack(*rows)


def _gamma_coeffs(gamma_i, r_i, s):
    """把 gamma_i(s) 写成 s^{r_i} + ... 的形式，返回低次到高次的系数表 [gamma_i0, ..., gamma_i,r_i-1]"""
    poly = sp.Poly(sp.sympify(gamma_i), s)
    if poly.degree() != r_i:
        raise ValueError(
            f"gamma = {gamma_i} 的次数是 {poly.degree()}，应等于相对阶 {r_i}"
        )
    coeffs = poly.all_coeffs()          # 高次 -> 低次
    if sp.simplify(coeffs[0] - 1) != 0:
        raise ValueError(f"gamma = {gamma_i} 不是首一多项式（首项应为 1）")
    return list(reversed(coeffs[1:]))   # 低次 -> 高次，去掉首项 1


def _is_hurwitz(gamma_i, s):
    """gamma_i 的根是否都在左半平面；符号系数判不出来时返回 None"""
    roots = sp.roots(sp.Poly(gamma_i, s), s)
    try:
        return bool(all(sp.re(rt) < 0 for rt in roots))
    except TypeError:
        return None


def cal_tracking_control(A, B, C, gamma, s=None):
    """跟踪控制律 u = K x + Gamma^{-1} v^*（课件 (7.10)）。

    A, B, C : 系统矩阵
    gamma   : 长度 p 的列表，第 i 项是 r_i 次首一 Hurwitz 多项式 gamma_i(s)
    s       : 多项式变元，默认 s

    返回 (K, Gamma, r, info)：
        K     : (m, n)，u = K x + Gamma^{-1} v^*
        Gamma : (p, m) 耦合矩阵
        r     : 相对阶列表
        info  : {Phi, gamma_A, s, stable}，Phi 的第 i 行是 c_i gamma_i(A)，
                stable 按 gamma_i 的根是否全在左半平面判断（系数带字母判不出来时为 None）

    Gamma 奇异（含 m != p）时抛 ValueError：静态状态反馈做不到这一组输出的
    独立跟踪，得先做输出变换或动态补偿。
    """
    A, B, C = sp.Matrix(A), sp.Matrix(B), sp.Matrix(C)
    s = s if s is not None else sp.Symbol("s")
    gamma = list(gamma)
    r, Gamma = cal_coupling_matrix(A, B, C)
    if len(gamma) != C.rows:
        raise ValueError(f"gamma 有 {len(gamma)} 个，应等于输出个数 {C.rows}")
    if Gamma.rows != Gamma.cols:
        raise ValueError(
            f"耦合矩阵是 {Gamma.rows}x{Gamma.cols}，输出数 {Gamma.rows} 与输入数 "
            f"{Gamma.cols} 不相等，静态状态反馈无法实现跟踪"
        )
    if sp.simplify(Gamma.det()) == 0:
        raise ValueError(f"耦合矩阵奇异（det = 0），Gamma = {Gamma}")

    Phi = sp.zeros(C.rows, A.cols)
    gamma_A = []
    for i in range(C.rows):
        coeffs = _gamma_coeffs(gamma[i], r[i], s)
        gi_A = A ** r[i]
        for k, ck in enumerate(coeffs):
            gi_A += ck * A ** k
        gamma_A.append(sp.expand(gi_A))
        Phi[i, :] = C[i, :] * gi_A

    K = sp.simplify(-Gamma.inv() * Phi)
    checks = [_is_hurwitz(gi, s) for gi in gamma]
    stable = None if any(ok is None for ok in checks) else all(checks)
    info = {
        "Phi": Phi,
        "gamma_A": gamma_A,
        "s": s,
        "stable": stable,
    }
    return K, Gamma, r, info


def cal_reference_vector(gamma, y_star, r, t, s=None):
    """参考前馈 v^*（课件 (7.8)）：

        v_i^* = y_i^{*(r_i)} + gamma_i,r_i-1 y_i^{*(r_i-1)} + ... + gamma_i,0 y_i^*

    gamma   : 长度 p 的 Hurwitz 多项式列表（与 cal_tracking_control 里同一组）
    y_star  : 长度 p 的期望输出表达式（sympy，变元 t）
    r       : 相对阶列表
    t       : 时间变元

    返回 (p, 1) 的 sympy 矩阵 v^*。
    """
    s = s if s is not None else sp.Symbol("s")
    t = sp.Symbol(t) if isinstance(t, str) else t
    y_star = [sp.sympify(y) for y in y_star]
    if not (len(y_star) == len(gamma) == len(r)):
        raise ValueError("y_star、gamma、r 的长度必须相同（都等于输出个数）")
    v = []
    for gi, yi, ri in zip(gamma, y_star, r):
        coeffs = _gamma_coeffs(gi, ri, s)
        vi = sp.diff(yi, t, ri)
        for k, ck in enumerate(coeffs):
            vi += ck * sp.diff(yi, t, k)
        v.append(sp.expand(vi))     # 只展开，不做 trig 化简（免得 sin(t+pi/4) 这种改写）
    return sp.Matrix(v)


def cal_error_dynamics(gamma, r, s=None):
    """跟踪误差 e_i = [e_i1, ..., e_i,r_i]^T 的闭环矩阵（课件 (7.6)）：

        e_i1' = e_i2, ..., e_i,r_i-1' = e_i,r_i,
        e_i,r_i' = -gamma_i0 e_i1 - ... - gamma_i,r_i-1 e_i,r_i

    gamma : 长度 p 的 Hurwitz 多项式列表；r : 相对阶列表

    返回 (sum(r), sum(r)) 的分块对角矩阵 A_e；它的特征值就是各 gamma_i(s) 的根，
    所以 A_e 是 Hurwitz 当且仅当 gamma_i 都 Hurwitz（跟踪收敛的充分条件）。
    """
    s = s if s is not None else sp.Symbol("s")
    blocks = []
    for gi, ri in zip(gamma, r):
        coeffs = _gamma_coeffs(gi, ri, s)
        blk = sp.zeros(ri, ri)
        for i in range(ri - 1):
            blk[i, i + 1] = 1
        for k, ck in enumerate(coeffs):
            blk[ri - 1, k] = -ck
        blocks.append(blk)
    A_e = sp.zeros(0, 0)
    for blk in blocks:
        A_e = sp.diag(A_e, blk) if A_e.rows else blk
    return A_e
