"""观测器设计的符号版（sympy）：极点配置、降维观测器、函数观测器、已知 F 反求系数。

数值版在 cal_observer.py（numpy，适合课件的算例），这里保留符号矩阵，
用于极点/参数还是字母的题（如第 9 章 9.8、9.9）以及需要看闭式解的时候。

约定与 numeric 版、课件 9.1-9.3 完全一致：

    全维：x̂' = F x̂ + G y + H u,  F = A + L C,  G = -L,  H = B
    降维：z' = F z + G y + H u,   x̂ = M z + N y
          F = A22 + L21 A12
          G = A21 + L21 A11 - F L21
          H = B2 + L21 B1
          M = C0 (C0^T C0)^{-1}
          N = C^T (C C^T)^{-1} - M L21
    函数：z' = F z + G y + H u,   f̂ = M z + N y,  f = R x
          H = T B,  T A - F T = G C,  M T + N C = R
"""
import sympy as sp

__all__ = [
    "cal_pole_placement_symbolic",
    "cal_reduced_order_observer_symbolic",
    "cal_function_observer_symbolic",
    "cal_observer_from_T",
]


def _as_matrix(M):
    """numpy / 列表 / sympy 矩阵统一成 sympy 矩阵"""
    if isinstance(M, sp.MatrixBase):
        return sp.Matrix(M)
    return sp.Matrix(M)


def _verify_poles(A_cl, wants, s):
    """charpoly(A_cl) 是否等于 prod(s - p)，即极点是否配到位"""
    phi = sp.expand(sp.prod([s - sp.sympify(p) for p in wants]))
    return sp.simplify(sp.expand(A_cl.charpoly(s).as_expr() - phi)) == 0


def cal_pole_placement_symbolic(A, B, poles, s=None):
    """状态反馈 K 使 lambda(A + B K) = poles（符号版，Ackermann 公式 + 单输入化）。

    A : (n, n)，B : (n, m)；poles : 长度 n 的期望极点（可为字母/复数）

    多输入时先挑一个输入组合 v，把 (A, B) 化成单输入 (A, B v) 能控，
    再对单输入用 Ackermann 公式，最后 K = v k^T——极点配置的解本就不唯一，
    这只是一个方便的代表解。

    返回 (K, info)：K 为 (m, n)，info = {v, b, Qc, phi}。
    B 的列怎么组合都不满秩时抛 ValueError（说明 (A, B) 不能控）。

    注意：判断 Q_c 是否可逆用的是"det 是否化简为 0"，带字母参数时这是
    "一般不退化"的判断，不是严格证明。
    """
    A, B = _as_matrix(A), _as_matrix(B)
    s = s if s is not None else sp.Symbol("s")
    n, m = A.shape[0], B.shape[1]
    poles = [sp.sympify(p) for p in poles]
    if len(poles) != n:
        raise ValueError(f"极点应有 {n} 个，实际 {len(poles)} 个")
    phi = sp.expand(sp.prod([s - p for p in poles]))
    coeffs = sp.Poly(phi, s).all_coeffs()          # 高次 -> 低次，首项 1
    phi_A = sp.zeros(n, n)
    for k, ck in enumerate(reversed(coeffs)):      # 低次 -> 高次
        phi_A += ck * A ** k

    candidates = [sp.Matrix([1 if i == j else 0 for i in range(m)]).reshape(m, 1)
                  for j in range(m)]
    candidates.append(sp.ones(m, 1))
    candidates.append(sp.Matrix(list(range(1, m + 1))).reshape(m, 1))
    tried = []
    for v in candidates:
        if any(sp.simplify(v - w) == sp.zeros(m, 1) for w in tried):
            continue
        tried.append(v)
        b = B * v
        if b.is_zero_matrix:
            continue
        Qc = sp.Matrix.hstack(*[A ** k * b for k in range(n)])
        if sp.simplify(Qc.det()) == 0:
            continue
        e_n = sp.Matrix([[1 if i == n - 1 else 0 for i in range(n)]])
        k = -(e_n * Qc.inv() * phi_A)              # 1 x n，A + b k^T 的极点为 poles
        K = v * k                                  # m x n
        if _verify_poles(A + B * K, poles, s):
            info = {"v": v, "b": b, "Qc": Qc, "phi": phi}
            return sp.simplify(K), info
    raise ValueError("找不到使 (A, B v) 能控的输入组合 v，(A, B) 不能控")


def cal_reduced_order_observer_symbolic(A, B, C, poles, C0=None, s=None):
    """降维状态观测器（符号版）：z' = F z + G y + H u，x̂ = M z + N y。

    A : (n, n)，B : (n, m)，C : (p, n)（行满秩）
    poles : 长度 n-p 的误差极点
    C0 : 可选，(n, n-p)，满足 C C0 = 0；默认取 C 的零空间

    返回 (F, G, H, M, N, info)，info 里是 T0、分块 Aij、B1、B2、L21、极点校验。

    设计步骤同课件 (9.16)-(9.23)：先构造 T0 把系统变成 C T0^{-1} = [I_p, 0]，
    再对 (A22, A12) 做对偶极点配置得 L21，最后按 (9.22) 算 F、G、H、M、N。
    """
    A, B, C = _as_matrix(A), _as_matrix(B), _as_matrix(C)
    s = s if s is not None else sp.Symbol("s")
    n = A.rows
    p = C.rows
    q = n - p
    if q <= 0:
        raise ValueError("p >= n：输出已给出全部状态，不需要降维观测器")
    poles = [sp.sympify(pl) for pl in poles]
    if len(poles) != q:
        raise ValueError(f"降维观测器应有 {q} 个极点，实际 {len(poles)} 个")

    # ---- C0 与坐标变换 T0（(9.16)(9.17)）--------------------------------
    if C0 is None:
        basis = C.nullspace()
        if len(basis) != q:
            raise ValueError("C 不是行满秩，null(C) 的维数不是 n-p")
        C0 = sp.Matrix.hstack(*basis)
    C0 = _as_matrix(C0)
    if sp.simplify(C * C0) != sp.zeros(p, C0.cols):
        raise ValueError("C0 必须满足 C C0 = 0")
    T0_inv = sp.Matrix.hstack(
        C.T * (C * C.T).inv(),
        C0 * (C0.T * C0).inv(),
    )
    T0 = T0_inv.inv()

    # ---- 分块（(9.20)(9.21)）-------------------------------------------
    AT = sp.simplify(T0 * A * T0_inv)
    BT = sp.simplify(T0 * B)
    A11, A12 = AT[:p, :p], AT[:p, p:]
    A21, A22 = AT[p:, :p], AT[p:, p:]
    B1, B2 = BT[:p, :], BT[p:, :]

    # ---- L21：对偶极点配置，A22^T + A12^T L21^T 的极点为 poles ----------
    L21_T, place_info = cal_pole_placement_symbolic(A22.T, A12.T, poles, s=s)
    L21 = sp.simplify(L21_T.T)                       # (n-p) x p

    F = sp.simplify(A22 + L21 * A12)
    G = sp.simplify(A21 + L21 * A11 - F * L21)
    H = sp.simplify(B2 + L21 * B1)
    M = sp.simplify(C0 * (C0.T * C0).inv())
    N = sp.simplify(C.T * (C * C.T).inv() - M * L21)

    info = {
        "T0": T0, "T0_inv": T0_inv, "C0": C0,
        "A11": A11, "A12": A12, "A21": A21, "A22": A22,
        "B1": B1, "B2": B2, "L21": L21, "q": q, "p": p,
        "place_info": place_info,
        "poles_ok": _verify_poles(F, poles, s),
        "recon_ok": sp.simplify(M * L21 + N - C.T * (C * C.T).inv()) == sp.zeros(n, p),
    }
    return F, G, H, M, N, info


def cal_function_observer_symbolic(A, B, C, R, F, G, s=None):
    """函数观测器（符号版）：给定 F、G，解出 T、H、M、N。

    A : (n, n)，B : (n, m)，C : (p, n)
    R : (r, n) 观测函数 f = R x
    F : (q, q) Hurwitz；G : (q, p)

    流程（课件 9.3）：
        1) Sylvester 方程 T A - F T = G C 解出 T（唯一，当 lambda(F)∩lambda(A)=空）
        2) H = T B
        3) [M N] = R [T; C]^{-1}（要求 q+p = n 且 [T; C] 非奇异）

    返回 (T, H, M, N, info)；info 含 Phi、Phi_det、两个残差和 charpoly_F。
    """
    A, B, C = _as_matrix(A), _as_matrix(B), _as_matrix(C)
    R = _as_matrix(R)
    F = _as_matrix(F)
    G = _as_matrix(G)
    s = s if s is not None else sp.Symbol("s")
    n = A.rows
    p = C.rows
    q = F.rows
    if G.shape != (q, p):
        raise ValueError(f"G 的形状应为 {(q, p)}，实际 {G.shape}")

    # ---- Sylvester：T A - F T = G C ------------------------------------
    syms = sp.symbols(f"t0:{q * n}")
    T = sp.Matrix(q, n, syms)
    eqs = list(T * A - F * T - G * C)
    sol = sp.linsolve(eqs, syms)
    if sol is sp.EmptySet:
        raise ValueError("Sylvester 方程无解：检查 lambda(F) 与 lambda(A) 是否相交")
    vals = dict(zip(syms, list(sol)[0]))
    # 解里若还带未知量（参数化解），把剩下的自由未知量取 0 得一组合理解
    free = set().union(*[v.free_symbols for v in vals.values()]) & set(syms)
    T = sp.Matrix(q, n, [sp.simplify(vals[v].subs({f: 0 for f in free})) for v in syms])
    syl_res = sp.simplify(T * A - F * T - G * C)

    H = sp.simplify(T * B)
    Phi = sp.Matrix.vstack(T, C)
    if Phi.rows != Phi.cols:
        raise ValueError(
            f"[T; C] 是 {Phi.rows}x{Phi.cols}，本函数只处理 q+p = n 的情形"
        )
    det_phi = sp.simplify(Phi.det())
    if det_phi == 0:
        raise ValueError("[T; C] 奇异：换一组 (F, G)")
    MN = sp.simplify(R * Phi.inv())
    M, N = MN[:, :q], MN[:, q:]

    info = {
        "T": T, "Phi": Phi, "Phi_det": det_phi, "q": q, "p": p,
        "residual_sylvester": syl_res,
        "residual_output": sp.simplify(M * T + N * C - R),
        "charpoly_F": F.charpoly(s).as_expr(),
    }
    return T, H, M, N, info


def cal_observer_from_T(A, B, C, F, T, R=None, use_output=True, s=None):
    """已知 F 和 T，反求观测器系数 G、H、M、N（课件 (9.32)-(9.34) 的逆用）。

    A, B, C : 系统矩阵；F : (q, q)；T : (q, n)（设计者给定或解出来的）
    R       : (r, n) 观测函数，默认为 I_n（要求重构整个状态）
    use_output : True  按 f̂ = M z + N y 的形式，[M N] = R [T; C]^{-1}（需 q+p = n）
                 False 按 x̂ = M z 的形式（N = 0），M = R T^{-1}（需 q = n）

    返回 (G, H, M, N, info)，info 里是可解性检查、Sylvester 残差、极点校验。

    可解性：G 存在当且仅当 (T A - F T) 作用在 null(C) 上为零，此时
        G = (T A - F T) C^T (C C^T)^{-1}
    就是 (9.33) 对 G 的最小二乘解（C 行满秩时是精确解）。
    """
    A, B, C = _as_matrix(A), _as_matrix(B), _as_matrix(C)
    F = _as_matrix(F)
    T = _as_matrix(T)
    s = s if s is not None else sp.Symbol("s")
    n = A.rows
    p = C.rows
    q = F.rows
    R = sp.eye(n) if R is None else _as_matrix(R)

    Delta = sp.simplify(T * A - F * T)
    G = sp.simplify(Delta * C.T * (C * C.T).inv())   # G = Delta * C^+
    compat = sp.simplify(Delta - G * C)              # 应恒为零矩阵
    H = sp.simplify(T * B)

    if use_output:
        Phi = sp.Matrix.vstack(T, C)
        if Phi.rows != Phi.cols:
            raise ValueError(f"[T; C] 是 {Phi.rows}x{Phi.cols}，需 q+p = n 才能解出 M、N")
        if sp.simplify(Phi.det()) == 0:
            raise ValueError("[T; C] 奇异，解不出 M、N")
        MN = sp.simplify(R * Phi.inv())
        M, N = MN[:, :q], MN[:, q:]
    else:
        if q != n:
            raise ValueError(f"x̂ = M z 的形式要求 q = n = {n}，实际 q = {q}")
        if sp.simplify(T.det()) == 0:
            raise ValueError("T 奇异，解不出 M")
        M = sp.simplify(R * T.inv())
        N = sp.zeros(R.rows, p)

    info = {
        "Phi": sp.Matrix.vstack(T, C), "delta": Delta, "compat_ok": compat == sp.zeros(q, n),
        "T": T, "F": F, "q": q, "p": p,
        "residual_output": sp.simplify(M * T + N * C - R),
        "charpoly_F": F.charpoly(s).as_expr(),
    }
    return G, H, M, N, info
