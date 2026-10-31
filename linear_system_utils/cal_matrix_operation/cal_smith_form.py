"""多输入能控系统的 Smith 规范型变换矩阵 T_r(s)。

对能控的 (A, B)（要求 B 列满秩），存在 (n+m) x (n+m) 的多项式矩阵 T_r(s)，使

    [A - s I_n,  B] T_r(s) = [I_n, 0],        det T_r(s) = 1,

记

    T_r(s) = [[P(s), N(s)],
              [Q(s), D(s)]],

也就是 PBH 判据里 [A - s I_n, B] 右乘一个幺模（行列式恒为 1）多项式矩阵之后
化成的 Smith 规范型；第 4 章的不变因子、特征结构配置要用到它。

构造（定理 2.7.1）分三步：

 1. 先做 Luenberger 能控规范型，拿到能控性指标集 mu_i、变换矩阵 T、耦合矩阵
    Gamma（就是本包 cal_controllable_canonical_form 的 Q / p / T / Tau）：

        Q = [b_1, A b_1, ..., A^(mu_1-1) b_1 | ... | b_m, ..., A^(mu_m-1) b_m]
        p_i = Q^-1 的第 (mu_1 + ... + mu_i) 行
        T = [p_1; p_1 A; ...; p_1 A^(mu_1-1) | ... | p_m; ...; p_m A^(mu_m-1)]
        Gamma[i] = p_i A^(mu_i-1) B          （m x m 单位上三角）

    规范型 A_c = T A T^-1 = A0 - B0 G，B_c = T B = B0 Gamma，其中

        A0 = diag_i [[0, I_(mu_i-1)], [0, 0]],   B0 = diag_i [0; ...; 0; 1]

    G 是 m x n 的块矩阵，第 (i, j) 块是 1 x mu_j 的行向量 alpha_ij

        G = [[alpha_11, ..., alpha_1m], ..., [alpha_m1, ..., alpha_mm]]

    求法（三种等价说法，任选一种）：
        G 的第 i 行块 = -(A_c 第 i 块块行的最后一行)   即读 A_c 里 A_ij 子块最后一行取反
        G 的第 i 行   = -(A0 - A_c) 的第 (mu_1+...+mu_i) 行
        G 的第 i 行   = -p_i A^(mu_i) T^-1
    注意 alpha 的次数从低到高排（alpha_ij0 配 s^0），和 A_ij 最后一行同序。
    第 4 章可用 G 反推不变因子：sI - (A0 - B0G) 的子块最后一行就是 -alpha。

 2. 由 mu_i 摆多项式矩阵（n x n 的 P0、n x m 的 N0、m x n 的 Q0、m x m 的 D0）：

        P0 = diag_i P0_i,  P0_i 的顶上一整行是 0，
             左下 (mu_i-1) x (mu_i-1) 块是 S_(mu_i-1)，最后一列是 0
             （mu_i = 1 时整块就是 1 x 1 的 0）
        S_k = [[1, 0, ..., 0],
               [s, 1, ..., 0],
               [...,        ],
               [s^(k-1), ..., s, 1]]       k x k 下三角 Toeplitz，S_k[i][j] = s^(i-j)

        Q0 = diag_i [s^(mu_i-1), ..., s, 1]          (1 x mu_i)
        N0 = diag_i [1; s; ...; s^(mu_i-1)]          (mu_i x 1)
        D0 = diag(s^(mu_1), ..., s^(mu_m))

    这四块满足 (A0 - s I) P0 + B0 Q0 = I、 (A0 - s I) N0 + B0 D0 = 0、
    det [[P0, N0], [Q0, D0]] = 1（容易逐块验证）。

 3. 拼

        T_r(s) = [[T^-1, 0], [Gamma^-1 G, Gamma^-1]]
                 [[P0, N0], [Q0, D0]]
                 [[T, 0], [0, I_m]]
               = [[P, N], [Q, D]],
        P = T^-1 P0 T,  N = T^-1 N0,  Q = Gamma^-1 (G P0 + Q0) T,
        D = Gamma^-1 (G N0 + D0).

注意 T_r 是 (n+m) x (n+m) 的方阵，不是 [A - sI, B] 的逆（那个不是方阵）；
课件把 T_r(s) 记作 R(s)。判据见 cal_smith_form_check。

用法（课程例 2.7.1）：A、B 见下，得到 6 x 6 的 T_r(s)
    T = [[1,0,0,-1],[0,-1,0,2],[-1,-1,1,2],[-1,2,0,-2]]
    Gamma = [[1,-1],[0,1]], G = [[-2,-2,-1,-1],[-3,0,0,-2]]

    import sympy as sp
    from linear_system_utils.cal_matrix_operation.cal_smith_form import (
        cal_smith_form, cal_smith_form_check)
    A = sp.Matrix([[-1, 0, 2, 1], [-1, 3, 3, -4], [0, 0, 2, 1], [-1, 1, 2, -1]])
    B = sp.Matrix([[0, 1], [0, 2], [1, 0], [0, 1]])
    res = cal_smith_form(A, B)
    cal_smith_form_check(A, B, res["T_r"])      # 两条判据都该是 True
"""
import numpy as np
import sympy as sp

from ..cal_matrix_parameters.cal_set_of_controllability_indices import (
    cal_set_of_controllability_indices,
)

__all__ = ["cal_smith_form", "cal_smith_form_check"]


def _to_sympy(M, name="M"):
    """numpy 数组 / 嵌套列表 / sympy 矩阵统一转成 sympy 矩阵（浮点按有理数收）"""
    try:
        M = sp.Matrix(M)
    except Exception as exc:                            # noqa: BLE001
        raise ValueError(f"{name} 转不成 sympy 矩阵：{exc}") from exc
    return M.applyfunc(lambda x: sp.nsimplify(x, rational=True) if x.is_Float else x)


def _block_diag(mats):
    """块对角拼接（按行高 / 列宽累加，空列表返回 0 x 0）"""
    rows = sum(m.rows for m in mats)
    cols = sum(m.cols for m in mats)
    out = sp.zeros(rows, cols)
    r = c = 0
    for m in mats:
        out[r:r + m.rows, c:c + m.cols] = m
        r += m.rows
        c += m.cols
    return out


def _S(k, s):
    """S_k：k x k 下三角 Toeplitz，S_k[i][j] = s^(i-j)（i >= j），其余为 0"""
    out = sp.zeros(k, k)
    for i in range(k):
        for j in range(i + 1):
            out[i, j] = s ** (i - j)
    return out


def _P0_block(k, s):
    """P0 的第 i 块：k x k，顶上一行全 0，左下 S_(k-1)，最后一列 0（k=1 时就是 [0]）"""
    if k == 1:
        return sp.Matrix([[0]])
    return sp.Matrix.vstack(
        sp.zeros(1, k),                                   # 顶上一整行 0
        sp.Matrix.hstack(_S(k - 1, s), sp.zeros(k - 1, 1)),   # 左下 S、最后一列 0
    )


def cal_smith_form(A, B, mu=None, s=None, verbose=False):
    """
    求 Smith 规范型的变换矩阵 T_r(s)，[A - sI_n, B] T_r(s) = [I_n, 0]、det T_r = 1

    A : (n, n)，能控
    B : (n, m)，列满秩（m 个输入）
    mu: 能控性指标集 [mu_1, ..., mu_m]，长度 m、sum(mu) = n；
        传 None 时按 B 的列序现算（B 的列序决定了 mu 的顺序，换列序结果跟着换）
    s : 多项式变量，默认用 sympy 的 s
    verbose: True 时打印 T、Gamma、G 和两条判据

    返回 dict：
        mu     [mu_1, ..., mu_m]
        Q, p   能控性矩阵 / 取行得到的 p_i
        T      Luenberger 变换矩阵，(n, n)
        Gamma  耦合矩阵，(m, m)，单位上三角
        G      alpha 块矩阵，(m, n)
        A_c    = T A T^-1，(n, n)
        B_c    = T B，(n, m)
        P0, N0, Q0, D0   第 2 步的多项式块
        T_r    Smith 规范型变换矩阵，(n+m, n+m) 多项式矩阵
    """
    A = _to_sympy(A, "A")
    B = _to_sympy(B, "B")
    s = sp.Symbol("s") if s is None else s

    n = A.rows
    m = B.cols
    if B.rows != n:
        raise ValueError(f"B 的行数 {B.rows} 应等于 n = {n}")

    if mu is None:
        mu = cal_set_of_controllability_indices(
            np.asarray(A.tolist(), dtype=float), np.asarray(B.tolist(), dtype=float))
    mu = list(mu)
    if len(mu) != m:
        raise ValueError(f"mu 的长度 {len(mu)} 应等于输入个数 m = {m}")
    if sum(mu) != n:
        raise ValueError(f"sum(mu) = {sum(mu)} 应等于 n = {n}（{mu} 不是这组 (A, B) 的指标集）")
    if min(mu) < 1:
        raise ValueError(f"mu = {mu} 里有 0，说明 B 的某列多余（B 不满列秩），做不了规范型")

    # ---- 第 1 步：Luenberger 规范型 ----
    cols = []
    for i in range(m):
        v = B[:, [i]]
        for _ in range(mu[i]):
            cols.append(v)
            v = A * v
    Q = sp.Matrix.hstack(*cols)
    if sp.simplify(Q.det()) == 0:
        raise ValueError("Q 奇异：要么 (A, B) 不能控，要么 mu 不是这组 (A, B) 的指标集")

    Q_inv = Q.inv()
    p, acc = [], 0
    for i in range(m):
        acc += mu[i]
        p.append(Q_inv[acc - 1, :])

    T = sp.Matrix.vstack(*[
        sp.Matrix.vstack(*[p[i] * A ** k for k in range(mu[i])]) for i in range(m)])
    Gamma = sp.Matrix.vstack(*[p[i] * A ** (mu[i] - 1) * B for i in range(m)])
    T_inv = T.inv()
    A_c = sp.expand(T * A * T_inv)
    B_c = sp.expand(T * B)

    A0 = _block_diag([
        sp.Matrix.vstack(sp.Matrix.hstack(sp.zeros(k - 1, 1), sp.eye(k - 1)), sp.zeros(1, k))
        for k in mu])
    B0 = _block_diag([sp.Matrix.vstack(sp.zeros(k - 1, 1), sp.ones(1, 1)) for k in mu])

    # ---- G：读 A_c 各块块行的最后一行取反（= -(A0 - A_c)[s_i] = -p_i A^mu_i T^-1）----
    G_rows, acc = [], 0
    for i in range(m):
        acc += mu[i]
        G_rows.append(-A_c[acc - 1, :])
    G = sp.Matrix.vstack(*G_rows)

    # 反查：A_c = A0 - B0 G、B_c = B0 Gamma 必须无条件成立，否则 mu 给错了
    if not sp.simplify(sp.expand(A0 - B0 * G) - A_c).is_zero_matrix:
        raise ValueError("A_c != A0 - B0 G：mu 不是这组 (A, B) 的真正指标集")
    if not sp.simplify(sp.expand(B0 * Gamma) - B_c).is_zero_matrix:
        raise ValueError("B_c != B0 Gamma：mu 不是这组 (A, B) 的真正指标集")

    # ---- 第 2 步：P0、N0、Q0、D0 ----
    P0 = _block_diag([_P0_block(k, s) for k in mu])
    N0 = _block_diag([sp.Matrix([s ** t for t in range(k)]) for k in mu])
    Q0 = _block_diag([sp.Matrix([[s ** t for t in range(k - 1, -1, -1)]]) for k in mu])
    D0 = sp.diag(*[s ** k for k in mu])

    # ---- 第 3 步：拼 ----
    middle = sp.Matrix.vstack(sp.Matrix.hstack(P0, N0), sp.Matrix.hstack(Q0, D0))
    left = sp.Matrix.vstack(
        sp.Matrix.hstack(T_inv, sp.zeros(n, m)),
        sp.Matrix.hstack(Gamma.inv() * G, Gamma.inv()))
    right = sp.Matrix.vstack(
        sp.Matrix.hstack(T, sp.zeros(n, m)),
        sp.Matrix.hstack(sp.zeros(m, n), sp.eye(m)))
    T_r = sp.expand(left * middle * right)

    res = {
        "mu": mu, "Q": Q, "p": p, "T": T, "Gamma": Gamma, "G": G,
        "A_c": A_c, "B_c": B_c, "T_inv": T_inv,
        "P0": P0, "N0": N0, "Q0": Q0, "D0": D0, "T_r": T_r,
    }

    if verbose:
        print(f"mu = {mu}")
        print("T =\n", T)
        print("Gamma =\n", Gamma)
        print("G =\n", G)
        print("T_r(s) =\n", T_r)
        chk = cal_smith_form_check(A, B, T_r, s=s)
        print("[A-sI B]T_r = [I 0]:", chk["first"], "  det T_r = 1:", chk["det"])

    return res


def cal_smith_form_check(A, B, T_r, s=None):
    """
    验算 T_r(s) 是不是这组 (A, B) 的 Smith 变换矩阵

    A : (n, n)
    B : (n, m)
    T_r : (n+m, n+m) 候选多项式矩阵
    s : 多项式变量，默认 sympy 的 s

    返回 dict：
        shape     T_r 是不是 (n+m, n+m)
        first     [A - sI, B] T_r == [I, 0]
        det       det T_r == 1
        residual  [A - sI, B] T_r - [I, 0]（化简后，用来定位）
        det_expr  det T_r - 1（化简后）
    """
    A = _to_sympy(A, "A")
    B = _to_sympy(B, "B")
    T_r = _to_sympy(T_r, "T_r")
    s = sp.Symbol("s") if s is None else s

    n = A.rows
    m = B.cols
    if B.rows != n:
        raise ValueError(f"B 的行数 {B.rows} 应等于 n = {n}")

    shape_ok = T_r.shape == (n + m, n + m)
    residual = sp.simplify(sp.expand(
        sp.Matrix.hstack(A - s * sp.eye(n), B) * T_r
        - sp.Matrix.hstack(sp.eye(n), sp.zeros(n, m))))
    det_expr = sp.simplify(sp.expand(T_r.det()) - 1)

    return {
        "shape": shape_ok,
        "first": residual.is_zero_matrix is True,
        "det": det_expr == 0,
        "residual": residual,
        "det_expr": det_expr,
    }
