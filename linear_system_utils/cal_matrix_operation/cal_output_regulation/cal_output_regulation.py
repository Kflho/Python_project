"""输出调节（output regulation）：调节器方程与静态状态反馈增益。

问题（参考输入和干扰都由同一个"外系统"产生）：

    x' = A x + B u + E d        d = P w     （干扰，可省）
    y  = C x
    e  = y - r                  r = Q w     （参考，可省）
    w' = S w

取静态状态反馈 u = K x + K_v w 能渐近跟踪 r 并抑制 d，当且仅当存在
(Pi, Gamma) 满足调节器方程

    Pi S = A Pi + B Gamma + E P      （稳态：x = Pi w，u = Gamma w）
    C  Pi = Q                        （稳态输出正好等于参考）

这时的稳态是 x = Pi w。令 K_v = Gamma - K Pi，把 e = C x - Q w 写成
e = C (x - Pi w)，而 xi = x - Pi w 满足

    xi' = (A + B K) xi + (E P + B Gamma ... 全消掉) = (A + B K) xi

（代进去就发现外系统项被方程抵消了），所以只要 K 让 A + B K 稳定，
任意初值下 e -> 0。

本模块只负责"解方程 / 给增益 / 验残差"，K 从外面给（极点配置、LQR 都行），
这样既能配静态状态反馈（实验三），也能配内模动态补偿器的状态反馈（实验四）。
"""
import numpy as np

__all__ = [
    "cal_regulator_equations",
    "cal_output_regulation_gain",
    "cal_regulation_solvability",
]


def cal_regulator_equations(A, B, C, S, Q=None, E=None, P=None, tol=1e-9):
    """解调节器方程，返回 (Pi, Gamma, info)。

        Pi S = A Pi + B Gamma + E P
        C  Pi = Q

    A : (n, n)      B : (n, m)      C : (p, n)
    S : (ns, ns)    外系统（参考/干扰的生成器），w' = S w
    Q : (p, ns)     参考输出系数 r = Q w，默认全 0（只抑制干扰）
    E : (n, nd)     干扰输入矩阵，默认 None（无干扰）
    P : (nd, ns)    干扰的外系统系数 d = P w，默认 None

    Pi : (n, ns)    稳态解 x = Pi w
    Gamma : (m, ns) 稳态控制 u = Gamma w

    info 里：
        residual        两块方程的合并残差（最大绝对值），数值上应 ~1e-13
        rank            方程组的秩
        unique          True 表示解唯一（秩 = 未知数个数）
        n_unknown       n*ns + m*ns

    m = p 时方程组是方阵；m != p 时用最小二乘，看 residual 判断是否真解。
    解不出来（外系统频率落在零点上）时 residual 会明显偏大，函数不报错、
    由调用方检查 —— 判据见 cal_regulation_solvability。
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    S = np.atleast_2d(np.asarray(S, dtype=float))

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    ns = S.shape[0]

    Q = np.zeros((p, ns)) if Q is None else np.atleast_2d(np.asarray(Q, dtype=float))
    EP = np.zeros((n, ns))
    if E is not None and P is not None:
        EP = np.atleast_2d(np.asarray(E, dtype=float)) @ np.atleast_2d(np.asarray(P, dtype=float))

    # 未知量按列优先排：vec(Pi)（n*ns 个）后接 vec(Gamma)（m*ns 个）
    M = np.zeros((n * ns + p * ns, n * ns + m * ns))
    rhs = np.zeros(M.shape[0])
    i_pi = lambda i, j: i + j * n            # Pi[i, j]
    i_ga = lambda k, j: n * ns + k + j * m   # Gamma[k, j]

    row = 0
    for j in range(ns):                       # 逐列写 Pi S - A Pi - B Gamma = E P
        for i in range(n):
            for k in range(ns):
                M[row, i_pi(i, k)] += S[k, j]
            for l in range(n):
                M[row, i_pi(l, j)] -= A[i, l]
            for k in range(m):
                M[row, i_ga(k, j)] -= B[i, k]
            rhs[row] = EP[i, j]
            row += 1
    for j in range(ns):                       # 再写 C Pi = Q
        for i in range(p):
            for l in range(n):
                M[row, i_pi(l, j)] += C[i, l]
            rhs[row] = Q[i, j]
            row += 1

    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    Pi = sol[:n * ns].reshape(ns, n).T        # 列优先还原
    Gamma = sol[n * ns:].reshape(ns, m).T
    res = M @ sol - rhs
    rank = int(np.linalg.matrix_rank(M, tol=tol))

    info = {
        "residual": float(np.max(np.abs(res))) if res.size else 0.0,
        "rank": rank,
        "n_unknown": M.shape[1],
        "unique": bool(rank == M.shape[1]),
    }
    return Pi, Gamma, info


def cal_output_regulation_gain(A, B, C, S, K, Q=None, E=None, P=None):
    """给稳定化增益 K，返回静态输出调节增益 (K_v, Pi, Gamma, info)。

        u = K x + K_v w,    K_v = Gamma - K Pi

    K : (m, n)，需要使 A + B K 稳定（u = K x 的约定，与 A - B K 相反）。

    info 在 cal_regulator_equations 的基础上追加：
        residual_closed_loop    Pi S - (A+BK) Pi - B K_v - E P 的最大绝对值
        residual_output        C Pi - Q 的最大绝对值
        stable                 闭环 A + B K 是否稳定
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))

    Pi, Gamma, info = cal_regulator_equations(A, B, C, S, Q=Q, E=E, P=P)
    Kv = Gamma - K @ Pi

    EP = np.zeros_like(Pi)
    if E is not None and P is not None:
        EP = np.atleast_2d(np.asarray(E, dtype=float)) @ np.atleast_2d(np.asarray(P, dtype=float))
    Qm = np.zeros((C.shape[0], Pi.shape[1])) if Q is None else np.atleast_2d(np.asarray(Q, dtype=float))

    Acl = A + B @ K
    info["residual_closed_loop"] = float(np.max(np.abs(Pi @ np.asarray(S, dtype=float)
                                                      - Acl @ Pi - B @ Kv - EP)))
    info["residual_output"] = float(np.max(np.abs(C @ Pi - Qm)))
    info["stable"] = bool(np.all(np.real(np.linalg.eigvals(Acl)) < 0))
    info["closed_loop_poles"] = np.linalg.eigvals(Acl).tolist()
    return Kv, Pi, Gamma, info


def cal_regulation_solvability(A, B, C, S, tol=1e-9):
    """能调节性判据：外系统的每个特征值都不能是 (A,B,C) 的传输零点。

    对每个 lambda in eig(S) 检查

        rank([A - lambda I,  B;  C,  0]) == n + p

    不满足说明该频率上"输入推不动 / 输出看不见"的模态存在，静态反馈无解
    （内模动态补偿同样无解，得改模型）。返回 dict：

        ok      全部特征值都通过
        lam     特征值列表（复数）
        rank    对应秩
        need    需要的秩 n + p
        fail    没通过的频率
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    S = np.atleast_2d(np.asarray(S, dtype=float))

    n = A.shape[0]
    p = C.shape[0]
    lam = np.linalg.eigvals(S)
    ranks = []
    for L in lam:
        M = np.block([[A - L * np.eye(n), B],
                      [C, np.zeros((p, B.shape[1]))]])
        ranks.append(int(np.linalg.matrix_rank(M, tol=tol)))
    need = n + p
    bad = [complex(L) for L, r in zip(lam, ranks) if r < need]
    return {
        "ok": len(bad) == 0,
        "lam": lam.tolist(),
        "rank": ranks,
        "need": need,
        "fail": bad,
    }
