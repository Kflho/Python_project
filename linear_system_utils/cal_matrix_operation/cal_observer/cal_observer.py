"""观测器设计（课件第 9 章）：全维、降维、函数观测器。

统一约定（与课件 9.1/9.2/9.3 一致，系统 x' = A x + B u, y = C x）：

    全维：x̂' = F x̂ + G y + H u,   F = A + L C,  G = -L,  H = B
          误差 e = x - x̂ 满足 e' = F e，所以 F 必须是 Hurwitz。
          观测器增益 L 由对偶极点配置给出：L = K^T，
          K 使 A^T + C^T K 的特征值为期望极点（"分离原理"那一半）。

    降维：z' = F z + G y + H u,  x̂ = M z + N y，其中

          F  = A22 + L21 A12
          G  = A21 + L21 A11 - F L21
          H  = B2  + L21 B1
          M  = C0 (C0^T C0)^{-1}
          N  = C^T (C C^T)^{-1} - M L21

          （[A11 A12; A21 A22] 是用 T0 把系统按 C T0^{-1} = [I_p, 0] 分块后
          的矩阵，T0 见 cal_reduced_order_observer。）

    函数：只重构 f = R x，不重构整个状态
            z' = F z + G y + H u
            f̂ = M z + N y
          条件是（课件 9.3 的 (9.32)-(9.34)）
            H = T B,   T A - F T = G C,   M T + N C = R
          (F, G) 由设计者挑（F Hurwitz、可控），T 由 Sylvester 方程解出。

三种观测器都只做"设计量"，闭环、仿真、与解析解对照在各实验目录里做。
"""
import numpy as np

from ..cal_pole_placement.cal_state_feedback_pole_placement import (
    cal_state_feedback_pole_placement,
)

__all__ = [
    "cal_observer_matrix",
    "cal_full_order_observer",
    "cal_reduced_order_observer",
    "cal_function_observer",
    "cal_observer_closed_loop",
]


def cal_observer_matrix(A, C):
    """能观性矩阵 [C; C A; ...; C A^{n-1}]，形状 (n*p, n)。"""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = A.shape[0]
    blocks = [C]
    for _ in range(n - 1):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def _null_space(M, tol=1e-9):
    """M 的零空间（列向量），SVD 实现，不依赖 scipy。"""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    _, S, Vt = np.linalg.svd(M)
    s_max = S[0] if S.size else 0.0
    r = int(np.sum(S > tol * max(M.shape) * max(s_max, 1.0)))
    return Vt[r:].T


def _poles_of(M):
    return np.linalg.eigvals(np.atleast_2d(np.asarray(M, dtype=float)))


def _is_hurwitz(M):
    return bool(np.all(np.real(_poles_of(M)) < 0))


def _poles_hit(A_cl, poles, tol=1e-6):
    """A_cl 的特征值集合是否就是 poles（按复数排序后逐个比）"""
    want = np.sort_complex(np.atleast_1d(np.asarray(poles, dtype=complex)))
    got = np.sort_complex(_poles_of(A_cl))
    return got.size == want.size and bool(np.all(np.abs(got - want) < tol))


def _dual_state_gain(A_d, B_d, poles):
    """对偶系统 (A_d, B_d) 的状态反馈增益 K：lambda(A_d + B_d K) = poles。

    标准做法是多输入极点配置公式；但它要求 B 的列满秩、能控性指标都非零，
    降维观测器里 q = n-p 很小、p 比 q 还大（如 p=2、q=1）时这套指标会退化成
    nu = [1, 0]，公式里的 Q 不再是方阵。于是加一条退路：挑一个输入组合 v，
    先化成单输入 (A_d, B_d v) 能控，再对单输入做极点配置，K = v k^T。

    两种做法都会拿特征值回代校验，配不准就换下一个 v。
    """
    A_d = np.atleast_2d(np.asarray(A_d, dtype=float))
    B_d = np.atleast_2d(np.asarray(B_d, dtype=float))
    n = A_d.shape[0]
    m = B_d.shape[1]

    # 先试原公式（m <= n 时才有可能不退化成 nu_i = 0）
    if m <= n:
        try:
            K, _ = cal_state_feedback_pole_placement(A_d, B_d, poles)
            K = np.atleast_2d(np.asarray(K, dtype=float))
            if _poles_hit(A_d + B_d @ K, poles):
                return K
        except (ValueError, np.linalg.LinAlgError):
            pass

    # 退路：单输入化
    candidates = [np.eye(m)[:, [i]] for i in range(m)]
    candidates.append(np.ones((m, 1)))
    candidates.append(np.arange(1.0, m + 1).reshape(-1, 1))
    for v in candidates:
        b = B_d @ v
        if np.linalg.matrix_rank(b) == 0:
            continue
        Qc = np.hstack([np.linalg.matrix_power(A_d, k) @ b for k in range(n)])
        if np.linalg.matrix_rank(Qc) < n:
            continue
        try:
            k, _ = cal_state_feedback_pole_placement(A_d, b, poles)   # (1, n)
        except (ValueError, np.linalg.LinAlgError):
            continue
        K = v @ np.atleast_2d(np.asarray(k, dtype=float))
        if _poles_hit(A_d + B_d @ K, poles):
            return K
    raise ValueError(
        f"极点配置失败：v 试过 {len(candidates)} 个，都做不到 lambda(A + B K) = {poles}"
    )


def cal_full_order_observer(A, B, C, poles):
    """全维状态观测器：x̂' = F x̂ + G y + H u。

    A : (n, n)   B : (n, m)   C : (p, n)
    poles : 长度 n 的观测器极点（建议比状态反馈闭环极点快 2~5 倍）

    返回 (L, F, G, H, info)：
        L  : (n, p) 观测器增益，F = A + L C
        F, G, H : 观测器方程里的三个矩阵（G = -L, H = B）
        info: {obs_rank, observable, K_dual, poles, stable, residual}

    (A, C) 不能观时抛 ValueError（来自极点配置的能控性检查）。
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = A.shape[0]
    poles = np.atleast_1d(np.asarray(poles, dtype=complex))
    if poles.size != n:
        raise ValueError(f"观测器极点应有 {n} 个，实际 {poles.size} 个")
    if cal_observer_matrix(A, C).shape[0] < n or \
            np.linalg.matrix_rank(cal_observer_matrix(A, C)) < n:
        raise ValueError("(A, C) 不能观，无法设计全维观测器")

    # 对偶：A^T + C^T K 的极点 = A + K^T C 的极点，所以 L = K^T
    K_dual, _ = cal_state_feedback_pole_placement(A.T, C.T, poles)
    K_dual = np.atleast_2d(np.asarray(K_dual, dtype=float))
    L = K_dual.T

    F = A + L @ C
    G = -L
    H = B.copy()

    info = {
        "obs_rank": int(np.linalg.matrix_rank(cal_observer_matrix(A, C))),
        "observable": True,
        "K_dual": K_dual,
        "L": L,
        "poles": _poles_of(F),
        "stable": _is_hurwitz(F),
        "residual": float(np.max(np.abs(F - (A + L @ C)))),
    }
    return L, F, G, H, info


def cal_reduced_order_observer(A, B, C, poles, C0=None, tol=1e-9):
    """降维（Luenberger）状态观测器：z' = F z + G y + H u, x̂ = M z + N y。

    A : (n, n)   B : (n, m)   C : (p, n)，C 行满秩
    poles : 长度 n-p 的误差极点（F = A22 + L21 A12 的特征值）
    C0 : (n, n-p) 可选，满足 C C0 = 0；默认用 SVD 求 C 的零空间

    返回 (F, G, H, M, N, info)。info 里有：
        T0, T0_inv   坐标变换（x̃ = T0 x，C T0^{-1} = [I_p, 0]）
        Aij, B1, B2  分块矩阵
        L21          降维观测器的设计自由量
        poles, stable
        residual     重构恒等式  M L21 + N - C^T (C C^T)^{-1}  的最大绝对值
                     （说明 z + L21 y 就是 x̃2 的估计，稳态无差）

    注意：模型里只用 F、G、H、M、N，T0 只是推导用的中间量。
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n, m = A.shape[0], B.shape[1]
    p = C.shape[0]
    q = n - p
    if q <= 0:
        raise ValueError("p >= n：输出已经给出全部状态，不需要降维观测器")
    poles = np.atleast_1d(np.asarray(poles, dtype=complex))
    if poles.size != q:
        raise ValueError(f"降维观测器应有 {q} 个极点，实际 {poles.size} 个")

    # ---- C0 与坐标变换 T0（课件 (9.16)(9.17)）----------------------------
    if C0 is None:
        C0 = _null_space(C, tol=tol)
        if C0.shape[1] != q:
            raise ValueError("C 不是行满秩，无法按 (9.17) 构造 T0")
    C0 = np.atleast_2d(np.asarray(C0, dtype=float))
    if np.max(np.abs(C @ C0)) > 1e-8:
        raise ValueError("C0 必须满足 C C0 = 0")

    T0_inv = np.hstack([C.T @ np.linalg.inv(C @ C.T),
                        C0 @ np.linalg.inv(C0.T @ C0)])
    T0 = np.linalg.inv(T0_inv)

    # ---- 分块（课件 (9.20)(9.21)）--------------------------------------
    AT = T0 @ A @ T0_inv
    BT = T0 @ B
    A11, A12 = AT[:p, :p], AT[:p, p:]
    A21, A22 = AT[p:, :p], AT[p:, p:]
    B1, B2 = BT[:p, :], BT[p:, :]

    # ---- L21：让 A22 + L21 A12 的极点为期望值（对偶极点配置）------------
    K_r = _dual_state_gain(A22.T, A12.T, poles)
    L21 = K_r.T

    F = A22 + L21 @ A12
    G = A21 + L21 @ A11 - F @ L21
    H = B2 + L21 @ B1
    M = C0 @ np.linalg.inv(C0.T @ C0)
    N = C.T @ np.linalg.inv(C @ C.T) - M @ L21

    resid = M @ L21 + N - C.T @ np.linalg.inv(C @ C.T)
    info = {
        "T0": T0, "T0_inv": T0_inv, "C0": C0,
        "A11": A11, "A12": A12, "A21": A21, "A22": A22,
        "B1": B1, "B2": B2,
        "L21": L21, "q": q, "p": p,
        "poles": _poles_of(F), "stable": _is_hurwitz(F),
        "residual": float(np.max(np.abs(resid))),
    }
    return F, G, H, M, N, info


def cal_function_observer(A, B, C, R, F, G, tol=1e-9):
    """函数观测器：只重构 f = R x，输出 f̂ = M z + N y。

    A : (n, n)   B : (n, m)   C : (p, n)
    R : (r, n) 观测函数（状态反馈时 R = K；重构全部状态时 R = I_n）
    F : (q, q) Hurwitz，q 是观测器阶数（一般取 q >= n - p 最省事）
    G : (q, p)

    流程（课件 9.3 + 9.4）：
        1) Sylvester 方程  T A - F T = G C  解出 T (q, n)
        2) H = T B   （(9.32)）
        3) [M N] = R [T; C]^+   （(9.46)，要求 [T; C] 行满秩）

    返回 (M, N, T, H, info)：
        info = {rank_Phi, residual_sylvester, residual_output, poles, stable}
        residual_output = max|M T + N C - R|，应为 ~1e-13
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    F = np.atleast_2d(np.asarray(F, dtype=float))
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    n = A.shape[0]
    q = F.shape[0]
    p = C.shape[0]

    # ---- T A - F T = G C  ->  (A^T ⊗ I_q - I_n ⊗ F) vec(T) = vec(G C) ----
    Mk = np.kron(A.T, np.eye(q)) - np.kron(np.eye(n), F)
    rhs = (G @ C).reshape(-1, order="F")
    sol, *_ = np.linalg.lstsq(Mk, rhs, rcond=None)
    T = sol.reshape(q, n, order="F")
    resid_syl = np.max(np.abs(T @ A - F @ T - G @ C))
    if resid_syl > 1e-6:
        raise ValueError(
            f"Sylvester 方程无解（残差 {resid_syl:.2e}）：F 的特征值别和 A 的重合"
        )

    H = T @ B

    # ---- [M N] [T; C] = R ----------------------------------------------
    Phi = np.vstack([T, C])
    rank_phi = int(np.linalg.matrix_rank(Phi, tol=tol))
    MN = R @ np.linalg.pinv(Phi)
    M, N = MN[:, :q], MN[:, q:]
    resid_out = np.max(np.abs(M @ T + N @ C - R))

    info = {
        "T": T, "H": H, "q": q, "rank_Phi": rank_phi, "need_rank": q + p,
        "residual_sylvester": float(resid_syl),
        "residual_output": float(resid_out),
        "poles": _poles_of(F), "stable": _is_hurwitz(F),
    }
    return M, N, T, H, info


def cal_observer_closed_loop(A, B, C, K, F, G, H, M, N):
    """观测器 + 状态反馈 u = K x̂ 的闭环矩阵（[x; z] 坐标）。

    u = K (M z + N y) = K M z + K N C x，观测器 z' = F z + G y + H u。
    所以（z 阶数 q）

        [x']   [ A + B K N C        B K M      ] [x]
        [z'] = [ G C + H K N C      F + H K M  ] [z]

    返回 (A_cl, info)：info 里是闭环极点/稳定性，以及分离原理的复核
    （闭环极点 = eig(A+BK) ∪ eig(F)，数值上差 ~1e-12）。
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    F = np.atleast_2d(np.asarray(F, dtype=float))
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    H = np.asarray(H, dtype=float)
    if H.ndim == 1:
        H = H.reshape(-1, 1)
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    N = np.asarray(N, dtype=float)
    if N.ndim == 1:
        N = N.reshape(-1, 1)

    KC = K @ N @ C          # (m, n)
    KM = K @ M              # (m, q)
    A_cl = np.block([
        [A + B @ KC, B @ KM],
        [G @ C + H @ KC, F + H @ KM],
    ])
    poles = _poles_of(A_cl)
    poles_state = _poles_of(A + B @ K)
    poles_obs = _poles_of(F)
    info = {
        "A_closed": A_cl,
        "poles": poles,
        "stable": bool(np.all(np.real(poles) < 0)),
        "poles_state": poles_state,
        "poles_observer": poles_obs,
    }
    return A_cl, info
