"""代数 Riccati 方程（Algebraic Riccati Equation, ARE）的求解。

被控对象：

    ẋ = A x + B u

有限时间二次型性能指标（L 是终值加权，Q 是状态加权，R 是输入加权）：

    J = xᵀ(t_f) L x(t_f) + ∫₀^{t_f} (xᵀ Q x + uᵀ R u) dt

对应的矩阵微分 Riccati 方程以 L 为终值条件：

    -Ṗ = P A + Aᵀ P - P B R⁻¹ Bᵀ P + Q,      P(t_f) = L

无穷时间（t_f → ∞）时终值项消失，L 取 0，微分方程趋于稳态，得到代数
Riccati 方程

    P A + Aᵀ P - P B R⁻¹ Bᵀ P + Q = 0

它的唯一对称正定解 P（也就是使闭环 A - B R⁻¹ Bᵀ P 渐近稳定的那个解）给出
最优控制

    u* = -K x,      K = R⁻¹ Bᵀ P

所以本模块里 L 只在有限时间的微分 Riccati 方程里才有用（见同目录的
cal_differential_Riccati_function），代数的（无穷时间）方程用不到它：
传 None 或 0 都行，只是为了和微分方程的函数保持同一个签名。

关于 J 里的 1/2：方程里的 Q、R 就是 J 里的 Q、R（1/2 只是把整个积分乘了个
常数，不改变方程的写法）。所以课本上写 J = ∫(xᵀQx+uᵀRu)dt 还是
J = ½∫(xᵀQx+uᵀRu)dt，把对应的 Q、R 原样传进来就行。

求解思路（符号法，方程直接精确解）：

 1. 待定对称矩阵 P：上三角 n(n+1)/2 个元素取未知量 p_ij，下三角镜像；
 2. 把矩阵方程展开成 n(n+1)/2 个多项式方程（逐项通分，把 R⁻¹ 带来的分母
    乘掉，前提是 det R ≠ 0，正定的 R 自然满足）；
 3. sympy.solve 精确求解，再按「元素实值 → P 正定 → 闭环稳定」挑出唯一解。

n 比较大（n ≥ 4）或系数是浮点数时符号解法会很慢，这时用本模块的数值解法
cal_algebra_Riccati_numeric（Hamilton 矩阵特征向量法，纯 numpy，解同一个方程）。

用法（A = [[0,1],[0,0]]、B = [[0],[1]]、Q = I、R = 1 的双积分器）：

    import sympy as sp
    from utils.cal_matrix_operation.cal_optimal_control.cal_algebra_Riccati_function import (
        cal_algebra_Riccati_function,
        cal_optimal_feedback_gain,
    )

    A = sp.Matrix([[0, 1], [0, 0]])
    B = sp.Matrix([[0], [1]])
    P = cal_algebra_Riccati_function(A, B, R=sp.eye(1))     # L=0、Q=I 都是默认值
    # P = [[sqrt(3), 1], [1, sqrt(3)]]
    K = cal_optimal_feedback_gain(P, B, sp.eye(1))          # K = [[1, sqrt(3)]]
"""
import numpy as np
import sympy as sp

__all__ = [
    "cal_algebra_Riccati_function",
    "cal_optimal_feedback_gain",
    "cal_algebra_Riccati_numeric",
]


# ---------------------------------------------------------------------------
# 内部小工具
# ---------------------------------------------------------------------------

def _to_sympy(M, n=None, name="M", rationalize=True):
    """
    嵌套列表 / numpy 数组 / sympy 矩阵统一转成 sympy 矩阵

    M 为 None 时返回 n 阶单位阵（L、Q、R 的默认值就是这么来的）；
    rationalize=True 把浮点元素 nsimplify 成有理数，好让后面 sympy 精确求解。
    """
    if M is None:
        if n is None:
            raise ValueError(f"{name} 取默认单位阵时要知道阶数")
        return sp.eye(n)
    M = sp.Matrix(M)
    if rationalize:
        M = M.applyfunc(lambda x: sp.nsimplify(x, rational=True) if x.is_Float else x)
    return M


def _symmetrize(M, name, verbose=True):
    """方阵取对称部分 (M + Mᵀ)/2；本来就对称的原样返回"""
    diff = sp.simplify(M - M.T)
    if diff.is_zero_matrix:
        return M
    if verbose and diff.is_zero_matrix is False:
        print(f"注意：{name} 不是对称矩阵，按对称部分 ({name} + {name}^T)/2 计算")
    return sp.simplify((M + M.T) / 2)


def _symmetric_matrix_with_symbols(n, prefix="p"):
    """
    生成 n 阶对称符号矩阵 P 和它的未知量

    上三角元素 p_ij (i <= j) 是独立的未知量（待定时取实未知量），下三角镜像：
    这样未知量个数正好是 n(n+1)/2，解出来的 P 天然对称。
    """
    symbols = {}
    for i in range(n):
        for j in range(i, n):
            symbols[(i, j)] = sp.Symbol(f"{prefix}_{i + 1}{j + 1}", real=True)
    P = sp.Matrix(n, n, lambda i, j: symbols[(min(i, j), max(i, j))])
    return P, list(symbols.values())


_PROBE_VALUES = (sp.Rational(7, 5), sp.Rational(3, 5))     # 参数探测用的两个正值


def _matrix_without_symbols(M):
    """不含自由符号时转成 complex 的 numpy 数组；含符号或转不动就返回 None"""
    if M.free_symbols:
        return None
    try:
        return np.array(sp.N(M, 20).tolist(), dtype=complex)
    except (TypeError, ValueError):
        return None


def _probe_frames(A, B, R_inv, P):
    """
    把 P 里的自由参数换成具体正数，得到一组纯数值的 (A, B, R⁻¹, P)

    R、Q 正定是 LQR 的前提，所以参数（比如 R = r）按「取正值」探测是合理的；
    每个参数都在 _PROBE_VALUES 里取两个值各探一次，两次结论一致才算数。
    """
    params = P.free_symbols
    if not params:
        return [(A, B, R_inv, P)]
    frames = []
    for v in _PROBE_VALUES:
        subs = {s: (-v if s.is_negative else v) for s in params}
        frames.append((A.subs(subs), B.subs(subs), R_inv.subs(subs), P.subs(subs)))
    return frames


def _classify(A, B, R_inv, P, exclude=(), tol=1e-9):
    """
    判断候选解 P 的三个性质，返回 (real, pd, stable, probed)

        real  : 元素是否实值（False 表示确定不是实值，None 表示判不出）
        pd    : P 是否正定
        stable: 闭环 A - B R⁻¹BᵀP 是否渐近稳定（特征值实部全负）
        probed: 是否用了「参数取正」的数值探测（P 里含参数时才可能为 True）

    exclude 是方程还没定下来的待定元（解欠定时会出现），这时三个性质都判不了。

    解出来是具体数时直接数值判断；含参数（例如 R = r）时 sympy 往往判不动，
    就把参数换成具体正数再判断，两次探测结论一致才给是/否，否则返回 None。
    """
    if P.free_symbols & set(exclude):
        return None, None, None, True
    if any(x.has(sp.I) for x in P):
        return False, False, False, False

    reals, pds, stables = [], [], []
    for A_k, B_k, R_inv_k, P_k in _probe_frames(A, B, R_inv, P):
        M = _matrix_without_symbols(P_k)
        if M is None:
            continue
        reals.append(bool(np.all(np.abs(M.imag) <= tol)))
        if not reals[-1]:                           # 复解直接否掉，不再判正定和稳定性
            pds.append(False)
            stables.append(False)
            continue
        M = M.real
        pds.append(bool(np.all(np.linalg.eigvalsh((M + M.T) / 2) > tol)))
        A_np, B_np, R_np = (_matrix_without_symbols(X) for X in (A_k, B_k, R_inv_k))
        if A_np is None or B_np is None or R_np is None:
            stables.append(None)
            continue
        A_cl = A_np.real - B_np.real @ R_np.real @ B_np.real.T @ M
        stables.append(bool(np.all(np.linalg.eigvals(A_cl).real < -tol)))

    def _agree(values):
        """几组探测结果一致才给结论，不一致就是判不出（None）"""
        values = [v for v in values if v is not None]
        if not values:
            return None
        if all(values):
            return True
        if not any(values):
            return False
        return None

    return _agree(reals), _agree(pds), _agree(stables), bool(P.free_symbols)


def _fmt(flag):
    """True / False / None 打印成「是 / 否 / 判不出」"""
    return {True: "是", False: "否", None: "判不出"}[flag]


def _pprint(expr):
    """
    二维排版打印表达式：根号和分式是画出来的，嵌套根号比一行式好认得多

    控制台编码撑不住就依次往后退（Unicode 画法 → ASCII 画法 → 什么都不打），
    所以 Windows 的 GBK 终端也不会抛 UnicodeEncodeError，最多是画得朴素一点。
    """
    for use_unicode in (True, False):
        try:
            print(sp.pretty(expr, use_unicode=use_unicode))
            return
        except UnicodeEncodeError:
            continue


def _same_matrix(M1, M2):
    """两个矩阵是否相等（化简后差为零矩阵）"""
    return sp.simplify(M1 - M2).is_zero_matrix is True


def _hint_numeric(n, msg):
    """符号解法失败时的报错文案，顺手指个路"""
    return (
        f"{msg}；如果确认方程本身没问题，多半是 n = {n} 太大、符号解法太慢，"
        "可以改用同文件里的数值解法 cal_algebra_Riccati_numeric"
    )


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def cal_optimal_feedback_gain(P, B, R):
    """
    最优反馈增益 K = R⁻¹ Bᵀ P

    P : (n, n) 代数 Riccati 方程的解
    B : (n, m)
    R : (m, m)，可逆
    返回: (m, n) 的 sympy 矩阵，最优控制 u* = -K x
    """
    P, B, R = _to_sympy(P), _to_sympy(B), _to_sympy(R)
    return sp.simplify(R.inv() * B.T * P)


def cal_algebra_Riccati_function(A, B, L=None, Q=None, R=None, verbose=True):
    """
    符号求解代数（无穷时间）Riccati 方程 P A + AᵀP - P B R⁻¹BᵀP + Q = 0

    A : (n, n) 系统矩阵
    B : (n, m) 输入矩阵
    L : (n, n) 终值加权矩阵，只属于有限时间的微分 Riccati 方程；
        无穷时间取 0，这里只是为了和 cal_differential_Riccati_function
        保持同一套参数，传 None 或 0 都行（传非零值会被忽略）
    Q : (n, n) 状态加权，默认单位阵；要求对称（一般还要求正定或半正定）
    R : (m, m) 输入加权，默认单位阵；要求对称、可逆（一般还要求正定）
    verbose: 是否打印求解过程、候选解、增益 K 和验算结果

    返回: (n, n) 的 sympy 矩阵 P（对称正定，且使 A - B R⁻¹BᵀP 渐近稳定）

    找不到唯一合适的解时抛 ValueError，并把 sympy 给出的候选解及其
    「是否实值 / 是否正定 / 闭环是否稳定」打印出来，方便自己判断。
    """
    A = _to_sympy(A)
    B = _to_sympy(B)
    n, m = A.rows, B.cols

    if A.rows != A.cols:
        raise ValueError(f"A 必须是方阵，现在形状是 {A.shape}")
    if B.rows != n:
        raise ValueError(f"B 的行数 {B.rows} 与 A 的阶数 {n} 不一致")

    # 无穷时间的终值加权 L = 0：传 None、标量 0 或零矩阵都当 0 处理
    if L is None or (not hasattr(L, "__len__") and sp.sympify(L).is_zero):
        L = sp.zeros(n, n)
    else:
        L = _to_sympy(L)
        if L.rows == 1 and L.cols == 1 and L[0, 0].is_zero:
            L = sp.zeros(n, n)
    Q = _to_sympy(Q, n=n)                           # 默认 Q = I_n（状态加权）
    R = _to_sympy(R, n=m)                           # 默认 R = I_m（输入加权）

    if L.shape != (n, n):
        raise ValueError(f"L 的形状 {L.shape} 与 A 的阶数 {n} 不匹配")
    if Q.shape != (n, n):
        raise ValueError(f"Q 的形状 {Q.shape} 与 A 的阶数 {n} 不匹配（Q 是状态加权）")
    if R.shape != (m, m):
        raise ValueError(f"R 的形状 {R.shape} 与 B 的列数 {m} 不匹配")

    if not sp.simplify(L).is_zero_matrix and verbose:
        print("注意：代数（无穷时间）Riccati 方程里没有终值加权 L，按 L = 0 处理；"
              "L 只用在有限时间的微分 Riccati 方程里")

    Q = _symmetrize(Q, "Q", verbose)
    R = _symmetrize(R, "R", verbose)

    if R.det().is_zero:
        raise ValueError("R 不可逆（det R = 0），代数 Riccati 方程没有意义")
    try:
        R_inv = R.inv()
    except Exception as exc:                        # R 里含参数、行列式判不出来时
        raise ValueError(f"R 求逆失败：{exc}")

    # 1) 待定对称矩阵 P，元素 p_ij 为未知量
    P_sym, unknowns = _symmetric_matrix_with_symbols(n)

    # 2) 展开成多项式方程：P A + AᵀP - P B R⁻¹BᵀP + Q = 0，res 对称，取上三角就够
    res = P_sym * A + A.T * P_sym - P_sym * B * R_inv * B.T * P_sym + Q
    eqs = [sp.fraction(sp.cancel(res[i, j]))[0] for i in range(n) for j in range(i, n)]

    if verbose:
        print(f"待定对称矩阵 P 的未知量 {len(unknowns)} 个：{unknowns}")
        print(f"代数 Riccati 方程展开成 {len(eqs)} 个多项式方程，交给 sympy.solve 精确求解")

    # 3) 精确求解
    try:
        solutions = sp.solve(eqs, unknowns, dict=True)
    except Exception as exc:
        raise ValueError(_hint_numeric(n, f"sympy 求解代数 Riccati 方程失败：{exc}"))
    if not solutions:
        raise ValueError(_hint_numeric(n, "代数 Riccati 方程没有解"))

    # 4) 每个解代回 P，标上「实值 / 正定 / 闭环稳定」
    candidates = []
    for sol in solutions:
        P_sol = sp.simplify(P_sym.subs(sol))
        if any(_same_matrix(P_sol, c[0]) for c in candidates):
            continue                                # sympy 偶尔给出重复解
        free = bool(P_sol.free_symbols & set(unknowns))     # 解欠定，还带着待定元
        real, pd, stable, probed = _classify(A, B, R_inv, P_sol, exclude=unknowns)
        if free and verbose:
            print("有解含自由参数（方程欠定，解不唯一），先留作候选")
        candidates.append((P_sol, real, pd, stable, free, probed))

    if verbose:
        print(f"sympy 给出 {len(candidates)} 个不同的候选解：")
        for k, (_, real, pd, stable, free, probed) in enumerate(candidates, 1):
            print(
                f"  候选 {k}: 实值={_fmt(real)}，正定={_fmt(pd)}，"
                f"闭环稳定={_fmt(stable)}，含自由参数={'是' if free else '否'}"
                + ("（含参数，按参数取正探测判断）" if probed else "")
            )

    # 5) 挑最像「唯一对称正定解」的那个：不含待定元 > 实值 > 正定 > 闭环稳定
    def rank(c):
        _, real, pd, stable, free, _ = c
        return (
            0 if free else 1,
            1 if real is True else 0,
            2 if pd is True else (1 if pd is None else 0),
            2 if stable is True else (1 if stable is None else 0),
        )

    best = max(candidates, key=rank)
    P_sol, real, pd, stable, free, probed = best

    if real is False:
        raise ValueError(
            f"只找到复解（{len(candidates)} 个候选都不实值），检查 Q、R 是否正定、"
            "(A, B) 是否可稳、(A, Q) 是否可检测"
        )
    note = "（含参数，按参数取正探测判断）" if probed else ""
    if sum(1 for c in candidates if rank(c) == rank(best)) > 1 and verbose:
        print("警告：有多个同样合格的解，这里返回的是第一个，请自行核对")
    if free and verbose:
        print("警告：返回的解里还带着自由参数，方程欠定（一般说明 Q 不正定或系统不可检测）")
    if pd is not True and verbose:
        print(f"警告：这个解的正定性{_fmt(pd)}{note}，正常 LQR 里 Q 正定、(A, Q) 可检测时应为正定解")
    if stable is not True and verbose:
        print(f"警告：这个解的闭环稳定性{_fmt(stable)}{note}，正常 LQR 里 (A, B) 可稳时应有唯一稳定解")

    # 6) 验算 + 顺手给出增益 K、闭环极点和最优指标
    if verbose:
        check = sp.simplify(P_sol * A + A.T * P_sol - P_sol * B * R_inv * B.T * P_sol + Q)
        print("P =", P_sol)                         # 一行式，方便复制粘贴
        _pprint(P_sol)                              # 二维排版，根号分式看得清
        print(
            "代回验算 P A + A^T P - P B R^-1 B^T P + Q =",
            "O（零矩阵，方程成立）" if check.is_zero_matrix else check,
        )
        K = cal_optimal_feedback_gain(P_sol, B, R)
        print("最优反馈增益 K = R^-1 B^T P =", K)
        _pprint(K)
        print("最优控制 u* = -K x")
        A_cl = sp.simplify(A - B * K)
        print("闭环系统矩阵 A - B K =", A_cl)
        _pprint(A_cl)
        print("闭环极点 =", A_cl.eigenvals())
        _pprint(A_cl.eigenvals())

    return P_sol


def cal_algebra_Riccati_numeric(A, B, L=None, Q=None, R=None):
    """
    Hamilton 矩阵特征向量法数值求解同一个代数 Riccati 方程，返回 numpy 的 P

        H = [[A, -B R⁻¹Bᵀ], [-Q, -Aᵀ]]

    它的特征值关于虚轴对称：取实部为负的那 n 个特征值对应的特征向量，按上下两半
    拼成 [V1; V2]（V1 为 n x n 可逆），则 P = V2 V1⁻¹ 就是对称正定解
    （实现里再对称化一次，压掉数值误差）。

    和 cal_algebra_Riccati_function 解的是同一个方程，区别是这里全程浮点运算：
    n 较大（符号解法太慢）或系数本来就是测量数据时用它更合适。
    要求 H 在虚轴上没有特征值，否则稳定子空间选不出来（方程通常没有稳定解）。

    A : (n, n)
    B : (n, m)
    L : 终值加权，无穷时间取 0，这里只是和符号版保持同一套参数（传了也不用）
    Q : (n, n) 状态加权，None 表示单位阵
    R : (m, m) 输入加权，None 表示单位阵
    返回: (n, n) 的 numpy 数组 P
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n, m = A.shape[0], B.shape[1]

    def _np_or_eye(M, k, name):
        if M is None:
            return np.eye(k)
        M = np.asarray(M, dtype=float)
        if M.ndim == 1:
            M = M.reshape(-1, 1)
        return M

    Q = _np_or_eye(Q, n, "Q")
    R = _np_or_eye(R, m, "R")

    if A.shape != (n, n):
        raise ValueError(f"A 必须是方阵，现在形状是 {A.shape}")
    if B.shape[0] != n:
        raise ValueError(f"B 的行数 {B.shape[0]} 与 A 的阶数 {n} 不一致")
    if Q.shape != (n, n):
        raise ValueError(f"Q 的形状 {Q.shape} 与 A 的阶数 {n} 不匹配（Q 是状态加权）")
    if R.shape != (m, m):
        raise ValueError(f"R 的形状 {R.shape} 与 B 的列数 {m} 不匹配")

    R = (R + R.T) / 2
    Q = (Q + Q.T) / 2

    H = np.block([
        [A, -B @ np.linalg.inv(R) @ B.T],
        [-Q, -A.T],
    ])

    w, V = np.linalg.eig(H)
    idx = np.argsort(w.real)[:n]                    # 取实部最小的 n 个，就是稳定子空间
    if np.any(w.real[idx] > 1e-9):
        raise ValueError(
            "Hamilton 矩阵没有 n 个实部为负的特征值（虚轴上有特征值），"
            "取不出 n 维稳定子空间，这时方程可能没有唯一正定解"
        )
    V1, V2 = V[:n, idx], V[n:, idx]
    P = np.real(V2 @ np.linalg.inv(V1))
    return (P + P.T) / 2                            # 对称化，压掉数值误差
