"""输出调节（output regulation）：调节器方程组与静态状态反馈增益。

问题（参考与干扰由同一个外系统产生，符号按课程笔记）：

    x' = A x + B u + E v
    v' = S v
    e  = C x + D u + J v

v 是外系统状态，参考和干扰都是它的分量；E 把外信号送进对象，J 把参考写进误差。

取静态状态反馈 u = K_x x + K_v v，能内稳定且 e -> 0，当且仅当 (A, B) 能稳、且存在
(X, U) 满足**调节器方程组**

    X S = A X + B U + E        （稳态轨迹 x = X v，稳态控制 u = U v）
    0   = C X + D U + J        （稳态时误差为 0）

这时取 K_x 使 A + B K_x 为 Hurwitz 矩阵，K_v = U - K_x X；令 xi = x - X v，则

    xi' = (A + B K_x) xi,      e = C x + D u + J v = (C + D K_x) xi

（外系统项被方程组消掉），所以只要 A + B K_x 稳定，e 必趋于 0。

本模块只负责"解方程 / 给增益 / 验残差"，K_x 从外面给（极点配置、LQR 都行），
这样既能配静态状态反馈（实验三），也能配内模动态补偿器的状态反馈（实验四）。

方程组可解的判据（**不变零点条件**）见 cal_regulation_solvability：

    rank[[A - sI_n, B], [C, D]] = n + p,   s 取遍 S 的特征值
"""
import numpy as np

__all__ = [
    "cal_regulator_equations",
    "cal_output_regulation_gain",
    "cal_regulation_solvability",
    "cal_minimal_polynomial",
    "cal_internal_model_blocks",
    "cal_internal_model",
    "cal_internal_model_closed_loop",
]


def cal_regulator_equations(A, B, C, S, E=None, D=None, J=None, tol=1e-9):
    """解调节器方程组，返回 (X, U, info)。

        X S = A X + B U + E
        0   = C X + D U + J

    A : (n, n)      B : (n, m)      C : (p, n)      D : (p, m)
    S : (ns, ns)    外系统，v' = S v
    E : (n, ns)     外信号输入，默认 None（无干扰、参考也不直接进对象）
    J : (p, ns)     误差系数（参考取负就写负号），默认 None（参考为 0）

    X : (n, ns)     稳态解 x = X v
    U : (m, ns)     稳态控制 u = U v

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

    E = np.zeros((n, ns)) if E is None else np.atleast_2d(np.asarray(E, dtype=float))
    D = np.zeros((p, m)) if D is None else np.atleast_2d(np.asarray(D, dtype=float))
    J = np.zeros((p, ns)) if J is None else np.atleast_2d(np.asarray(J, dtype=float))

    # 未知量按列优先排：vec(X)（n*ns 个）后接 vec(U)（m*ns 个）
    M = np.zeros((n * ns + p * ns, n * ns + m * ns))
    rhs = np.zeros(M.shape[0])
    i_x = lambda i, j: i + j * n              # X[i, j]
    i_u = lambda k, j: n * ns + k + j * m     # U[k, j]

    row = 0
    for j in range(ns):                       # 逐列写 X S - A X - B U = E
        for i in range(n):
            for k in range(ns):
                M[row, i_x(i, k)] += S[k, j]
            for l in range(n):
                M[row, i_x(l, j)] -= A[i, l]
            for k in range(m):
                M[row, i_u(k, j)] -= B[i, k]
            rhs[row] = E[i, j]
            row += 1
    for j in range(ns):                       # 再写 C X + D U = -J
        for i in range(p):
            for l in range(n):
                M[row, i_x(l, j)] += C[i, l]
            for k in range(m):
                M[row, i_u(k, j)] += D[i, k]
            rhs[row] = -J[i, j]
            row += 1

    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    X = sol[:n * ns].reshape(ns, n).T         # 列优先还原
    U = sol[n * ns:].reshape(ns, m).T
    res = M @ sol - rhs
    rank = int(np.linalg.matrix_rank(M, tol=tol))

    info = {
        "residual": float(np.max(np.abs(res))) if res.size else 0.0,
        "rank": rank,
        "n_unknown": M.shape[1],
        "unique": bool(rank == M.shape[1]),
    }
    return X, U, info


def cal_output_regulation_gain(A, B, C, S, K, E=None, D=None, J=None):
    """给稳定化增益 K_x，返回静态输出调节增益 (K_v, X, U, info)。

        u = K_x x + K_v v,    K_v = U - K_x X

    K_x : (m, n)，需要使 A + B K_x 稳定（u = K x 的约定，与 A - B K 相反）。

    info 在 cal_regulator_equations 的基础上追加：
        residual_closed_loop   X S - (A + B K_x) X - B K_v - E 的最大绝对值
        residual_output        C X + D (K_x X + K_v) + J 的最大绝对值（e 的稳态项）
        stable                 A + B K_x 是否稳定
        closed_loop_poles      A + B K_x 的特征值
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))

    X, U, info = cal_regulator_equations(A, B, C, S, E=E, D=D, J=J)
    K_v = U - K @ X

    S = np.atleast_2d(np.asarray(S, dtype=float))
    p = C.shape[0]
    E_m = np.zeros_like(X) if E is None else np.atleast_2d(np.asarray(E, dtype=float))
    D_m = np.zeros((p, B.shape[1])) if D is None else np.atleast_2d(np.asarray(D, dtype=float))
    J_m = np.zeros((p, S.shape[0])) if J is None else np.atleast_2d(np.asarray(J, dtype=float))

    A_cl = A + B @ K
    info["residual_closed_loop"] = float(np.max(np.abs(X @ S - A_cl @ X - B @ K_v - E_m)))
    info["residual_output"] = float(np.max(np.abs(C @ X + D_m @ (K @ X + K_v) + J_m)))
    info["stable"] = bool(np.all(np.real(np.linalg.eigvals(A_cl)) < 0))
    info["closed_loop_poles"] = np.linalg.eigvals(A_cl).tolist()
    return K_v, X, U, info


def cal_regulation_solvability(A, B, C, S, D=None, tol=1e-9):
    """能调节性判据（不变零点条件）：外系统的每个特征值都不能是 (A,B,C,D) 的传输零点。

    对每个 lambda in eig(S) 检查

        rank([A - lambda I,  B;  C,  D]) == n + p

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
    m = B.shape[1]
    p = C.shape[0]
    D = np.zeros((p, m)) if D is None else np.atleast_2d(np.asarray(D, dtype=float))

    lam = np.linalg.eigvals(S)
    ranks = []
    for L in lam:
        M = np.block([[A - L * np.eye(n), B],
                      [C, D]])
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


def cal_minimal_polynomial(S, tol=1e-9):
    """数值求外系统 S 的最小多项式（首一、降幂系数），返回 (coeffs, d)。

    做法：把 I, S, S^2, ... 拉直成向量逐个入栈，找第一次线性相关的位置。
    设最小的 d 使 {vec(S^0), ..., vec(S^d)} 相关，则

        S^d = c_0 I + c_1 S + ... + c_{d-1} S^{d-1}
        min poly = s^d - c_{d-1} s^{d-1} - ... - c_1 s - c_0

    coeffs = [1, a_1, ..., a_d]，即 s^d + a_1 s^{d-1} + ... + a_d。

    例：S = diag(0, [0 2; -2 0])（常值干扰 + r=cos2t）-> 最小多项式 s^3 + 4s，
    coeffs = [1, 0, 4, 0]，d = 3（特征多项式也是 3 次，这里两者重合）。

    S (ns,ns)  tol 判线性相关的相对容差
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    n = S.shape[0]
    vecs = [np.eye(n).ravel()]
    M = np.eye(n)
    for d in range(1, n + 1):
        M = M @ S
        v = M.ravel()
        V = np.vstack(vecs)                      # (d, n^2)，行是 vec(S^i)
        sol, *_ = np.linalg.lstsq(V.T, v, rcond=None)
        if np.linalg.norm(V.T @ sol - v) <= tol * max(1.0, np.linalg.norm(v)):
            sol = np.where(np.abs(sol) < tol, 0.0, sol)
            coeffs = np.concatenate(([1.0], -sol[::-1]))
            return coeffs, d
        vecs.append(v)
    # 到这里说明数值上判不出来，退回特征多项式
    return np.poly(np.linalg.eigvals(S)), n


def cal_internal_model_blocks(S, p, tol=1e-9):
    """多输出内模：每个输出通道各配一组，Ac = kron(I_p, Ac0)，Bc = kron(I_p, Bc0)。

    笔记（输出调节·动态状态反馈）里内模是分块对角：

        Omega_i = 友矩阵（最小多项式 s^k0 + ... ），Pi_i = [0; ...; 0; 1]

    k0 = deg(min poly(S)) 是最小多项式的次数（不是特征多项式——S 有重根/分块时
    两者不同），l_i = k0 对每个输出都一样，于是 l = p * k0 阶。

    S (ns,ns)   p 输出维数（误差通道数）
    返回 (Ac, Bc, info)：
        Ac (p*k0, p*k0)  Bc (p*k0, p)
        info = {Ac0, Bc0, coeffs, k0, nc}
    拼增广系统时把它们传给 cal_internal_model(A, B, C, S, Ac=Ac, Bc=Bc)。
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    coeffs, k0 = cal_minimal_polynomial(S, tol=tol)

    Ac0 = np.zeros((k0, k0))
    if k0 > 1:
        Ac0[:-1, 1:] = np.eye(k0 - 1)
    Ac0[-1, :] = -coeffs[::-1][:-1]              # [-a_k0, ..., -a_1]
    Bc0 = np.zeros((k0, 1))
    Bc0[-1, 0] = 1.0

    Ac = np.kron(np.eye(p), Ac0)
    Bc = np.kron(np.eye(p), Bc0)
    info = {"Ac0": Ac0, "Bc0": Bc0, "coeffs": coeffs, "k0": k0, "nc": p * k0}
    return Ac, Bc, info


def cal_internal_model(A, B, C, S, Bc=None, Ac=None):
    """把外系统 S 的特征多项式做成内模，拼出"对象 + 内模"的增广系统。

    内模（补偿器的动态部分，e = r - y = r - C x）：

        eta' = A_c eta + B_c e

    A_c 默认取 S 的特征多项式的**相变量形（友）矩阵**，B_c 默认 = [0; ...; 0; 1]。
    对 S = [0 2; -2 0]（r = 0.1cos2t）特征多项式 s^2 + 4，得到

        A_c = [0 1; -4 0]，B_c = [0; 1]

    但同一个传递函数有不同实现（积分链的方向、反馈取自哪一级），手搭模型时
    状态顺序由接线决定，所以 Ac / Bc 都可以直接传进来覆盖默认值（例如实验四
    手搭模型用的是 A_c = [0 -4; 1 0]、B_c = [1; 0]）。两种实现特征多项式相同，
    配出来的 K_eta 只是跟着状态顺序变。

    **多输出（p > 1）时每个输出通道各要一组内模**，阶数 l = p * k0（k0 是 S 的
    最小多项式次数，见 cal_internal_model_blocks）。这时 A_c / B_c 显式传进来，
    增广阶数按 A_c 的行数算，不再受 S 特征多项式次数限制：

        A_c = kron(I_p, A_c0)，B_c = kron(I_p, B_c0)

    增广系统 z = [x; eta]，u 是控制输入、r 是外部参考（误差取 e = y - r = C x - r）：

        z' = A_aug z + B_aug u + [0; -B_c] r
        A_aug = [[A, 0], [B_c C, A_c]],    B_aug = [[B], [0]]
        y = C_aug z,   C_aug = [C, 0]

    （若模型里误差取的是 r - y，把 Bc 传成 -Bc 即可，等价于 A_aug 的第二块取 -Bc C。）

    对 (A_aug, B_aug) 做极点配置得 K_aug = [K_x, K_eta]，控制律取

        u = K_x x + K_eta eta

    闭环稳定则 e -> 0（内模原理）。和静态输出调节的区别：**不需要 v 可测**，
    参考的模态由内模自己"长出来"，所以还能顺带抑制同频率的干扰。

    A (n,n)   B (n,m)   C (p,n)   S (ns,ns)
    Bc (nc,p) 可选，默认 [0;...;0;1]（仅 p = 1 时可省）
    Ac (nc,nc) 可选，默认按 S 的特征多项式构友矩阵；给了就以它的行数定 nc

    返回 (A_aug, B_aug, C_aug, info)：
        info = {Ac, Bc, coeffs, nc, ctrb_rank, controllable, n_aug}
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    S = np.atleast_2d(np.asarray(S, dtype=float))

    n, m = A.shape[0], B.shape[1]
    p = C.shape[0]

    coeffs = np.poly(np.linalg.eigvals(S))        # [1, a1, ..., a_ns]
    if Ac is None:
        nc = coeffs.size - 1
        if nc == 0:
            raise ValueError("S 是 1x1 且特征值为 0 时没有可用的内模，请直接用积分器")
        Ac = np.zeros((nc, nc))
        if nc > 1:
            Ac[:-1, 1:] = np.eye(nc - 1)
        Ac[-1, :] = -coeffs[::-1][:-1]            # [-a_nc, ..., -a_1]
    else:
        Ac = np.atleast_2d(np.asarray(Ac, dtype=float))
        nc = Ac.shape[0]
        if Ac.shape[1] != nc:
            raise ValueError(f"Ac 应为方阵，实际 {Ac.shape}")

    if Bc is None:
        if p != 1:
            raise ValueError("p > 1 时请显式给 Bc（每个输出通道配一组内模）")
        Bc = np.zeros((nc, 1))
        Bc[-1, 0] = 1.0
    Bc = np.atleast_2d(np.asarray(Bc, dtype=float))
    if Bc.shape[0] != nc:
        raise ValueError(f"Bc 应有 {nc} 行，实际 {Bc.shape[0]}")
    if Bc.shape[1] != p:
        raise ValueError(f"Bc 应有 {p} 列（输出通道数），实际 {Bc.shape[1]}")

    A_aug = np.block([[A, np.zeros((n, nc))],
                      [Bc @ C, Ac]])
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


def cal_internal_model_closed_loop(A, B, C, S, K_aug, Bc=None, Ac=None):
    """给内模增广系统的增益 K_aug = [K_x, K_eta]，返回 (A_cl, info)。

    A_cl 是 z = [x; eta] 的闭环矩阵（参考 r 当外部输入处理），info 里追加
    poles / stable / A_aug / B_aug / C_aug / Ac / Bc / nc。

    稳定性只说明 z -> 0 的齐次部分是稳的；对 r 的跟踪性由内模保证，
    仿真验证见 experiments/src/ex_04/。
    """
    A_aug, B_aug, C_aug, info = cal_internal_model(A, B, C, S, Bc=Bc, Ac=Ac)
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
