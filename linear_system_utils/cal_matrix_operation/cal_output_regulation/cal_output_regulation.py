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
    "cal_internal_model",
    "cal_internal_model_closed_loop",
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


def cal_internal_model(A, B, C, S, Bc=None):
    """把外系统 S 的特征多项式做成内模，拼出"对象 + 内模"的增广系统。

    内模（补偿器的动态部分，e = r - y = r - C x）：

        eta' = A_c eta + B_c e

    A_c 取 S 的特征多项式的**相变量形（友）矩阵**，B_c 默认 = [0; ...; 0; 1]。
    对 S = [0 2; -2 0]（r = 0.1cos2t）特征多项式 s^2 + 4，得到

        A_c = [0 1; -4 0]，B_c = [0; 1]

    正好就是"两个积分器 + 反馈 -4"的振荡器 —— 实验四图里那个结构。

    增广系统 z = [x; eta]，u 是控制输入、r 是外部参考：

        z' = A_aug z + B_aug u + [0; B_c] r
        A_aug = [[A, 0], [-B_c C, A_c]],   B_aug = [[B], [0]]
        y = C_aug z,   C_aug = [C, 0]

    对 (A_aug, B_aug) 做极点配置得 K_aug = [K_x, K_eta]，控制律取

        u = K_x x + K_eta eta

    闭环稳定则 e -> 0（内模原理）。和静态输出调节的区别：**不需要 w 可测**，
    参考的模态由内模自己"长出来"，所以还能顺带抑制同频率的干扰。

    A (n,n)   B (n,m)   C (p,n)   S (ns,ns)   Bc (nc,p) 可选
    m = p 时最省事；p > 1 请显式给 Bc（每个输出通道各配一组内模）。

    返回 (A_aug, B_aug, C_aug, info)：
        info = {Ac, Bc, coeffs, nc, ctrb_rank, controllable}
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    S = np.atleast_2d(np.asarray(S, dtype=float))

    n, m = A.shape[0], B.shape[1]
    p = C.shape[0]

    coeffs = np.poly(np.linalg.eigvals(S))        # [1, a1, ..., a_nc]
    nc = coeffs.size - 1
    if nc == 0:
        raise ValueError("S 是 1x1 且特征值为 0 时没有可用的内模，请直接用积分器")

    Ac = np.zeros((nc, nc))
    if nc > 1:
        Ac[:-1, 1:] = np.eye(nc - 1)
    Ac[-1, :] = -coeffs[::-1][:-1]                # [-a_nc, ..., -a_1]

    if Bc is None:
        if p != 1:
            raise ValueError("p > 1 时请显式给 Bc（每个输出通道配一组内模）")
        Bc = np.zeros((nc, 1))
        Bc[-1, 0] = 1.0
    Bc = np.atleast_2d(np.asarray(Bc, dtype=float))

    A_aug = np.block([[A, np.zeros((n, nc))],
                      [-Bc @ C, Ac]])
    B_aug = np.vstack([B, np.zeros((nc, m))])
    C_aug = np.hstack([C, np.zeros((p, nc))])

    # 增广系统能控性（内模原理要求 (A,B) 在 S 的特征值处能控）
    nn = n + nc
    cols = [B_aug]
    for _ in range(nn - 1):
        cols.append(A_aug @ cols[-1])
    rank = int(np.linalg.matrix_rank(np.hstack(cols)))

    info = {
        "Ac": Ac, "Bc": Bc, "coeffs": coeffs, "nc": nc,
        "ctrb_rank": rank, "controllable": bool(rank == nn), "n_aug": nn,
    }
    return A_aug, B_aug, C_aug, info


def cal_internal_model_closed_loop(A, B, C, S, K_aug, Bc=None):
    """给内模增广系统的增益 K_aug = [K_x, K_eta]，返回 (A_cl, info)。

    A_cl 是 z = [x; eta] 的闭环矩阵（参考 r 当外部输入处理），info 里追加
    poles / stable / A_aug / B_aug / Ac / Bc / nc。

    稳定性只说明 z -> 0 的齐次部分是稳的；对 r 的跟踪性由内模保证，
    仿真验证见 experiments/src/ex_04/。
    """
    A_aug, B_aug, C_aug, info = cal_internal_model(A, B, C, S, Bc=Bc)
    K = np.atleast_2d(np.asarray(K_aug, dtype=float))
    if K.shape[1] != A_aug.shape[0]:
        raise ValueError(
            f"K_aug 应有 {A_aug.shape[0]} 列（[K_x, K_eta]），实际 {K.shape[1]} 列"
        )
    A_cl = A_aug + B_aug @ K
    poles = np.linalg.eigvals(A_cl)
    info.update({
        "A_aug": A_aug, "B_aug": B_aug, "C_aug": C_aug,
        "A_closed": A_cl, "poles": poles,
        "stable": bool(np.all(np.real(poles) < 0)),
    })
    return A_cl, info

