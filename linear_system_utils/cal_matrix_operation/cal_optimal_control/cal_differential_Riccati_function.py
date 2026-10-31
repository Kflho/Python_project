"""微分 Riccati 方程（有限时间 LQR 的 P(t)）。

给定 (A, B, Q, R) 与终端权 L，解

    -dP/dt = P A + A^T P - P B R^-1 B^T P + Q,     P(T) = L,

反向积分实现：令 tau = T - t、X(tau) = P(T - tau)，则

    dX/dtau = X A + A^T X - X B R^-1 B^T X + Q,    X(0) = L,

从 tau = 0 前向积分到 tau = T，再按 t = T - tau 翻回来。每一步把 X 对称化
（X -> (X + X^T)/2）：DRE 的解必对称，而数值积分会让对称性慢慢漂。

用法（有限时间 LQR）：
    res = cal_differential_Riccati_function(A, B, L, Q, R, T=5.0)
    P0 = res["P0"]                       # t = 0 处的 P，最优代价 J = x0^T P0 x0
    P_at = lambda t: res["P"][int(np.argmin(np.abs(res["t"] - t)))]

T -> inf 时 P(t) 趋于代数 Riccati 方程的正定解，可用
cal_algebra_Riccati_function 对照（见 exercises/src/ls_05/ex_2.py）。
"""
import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["cal_differential_Riccati_function"]


def _sym(M):
    """对称化：(M + M^T)/2，DRE 的解必对称"""
    return 0.5 * (M + M.T)


def _as_mat(M, rows, cols, name):
    """统一成 (rows, cols) 浮点矩阵；一维按列向量收"""
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        if rows == 1:
            M = M.reshape(1, -1)
        else:
            M = M.reshape(-1, 1)
    M = np.atleast_2d(M)
    if M.shape != (rows, cols):
        raise ValueError(f"{name} 的形状 {M.shape} 与 ({rows}, {cols}) 不符")
    return M


def cal_differential_Riccati_function(A, B, L, Q, R, T, t_eval=None,
                                      rtol=1e-11, atol=1e-13, symmetrize=True,
                                      method="RK45", rhs=None):
    """
    解微分 Riccati 方程 -P' = PA + A^T P - P B R^-1 B^T P + Q，P(T) = L

    A : (n, n)
    B : (n, m)，m 个输入；单输入传 (n, 1) 或长度 n 的向量
    L : (n, n)，终端权 L = P(T)
    Q : (n, n)，半正定状态权
    R : (m, m)，正定控制权
    T : 终端时刻（> 0）
    t_eval : 要取的时刻（升序，落在 [0, T] 内）；None 时用求解器自己的网格
    rtol, atol : 传给 solve_ivp 的容差
    symmetrize : 每步对称化（默认 True；关掉可以看数值积分的对称性漂移）
    method : solve_ivp 的方法
    rhs : 换掉右端项（用 dX/dtau 自己的函数，比如加正则项），签名 rhs(tau, X) -> dX

    返回 dict：
        t     t 升序
        P     与 t 对应的 P(t)，list of (n, n)
        P0    t = 0 处的 P（J = x0^T P0 x0）
        PT    t = T 处的 P，应等于 L
        K     K(t) = -R^-1 B^T P(t)，list of (m, n)
        ok    自检：PT 与 L 的最大偏差、对称性偏差、P 的最小特征值
        sol   原始 OdeResult、n、m
    """
    A = np.asarray(A, dtype=float)
    A = A.reshape(A.shape[0], A.shape[0])
    n = A.shape[0]
    B = np.asarray(B, dtype=float)
    B = B.reshape(n, -1)
    m = B.shape[1]

    L = _as_mat(L, n, n, "L")
    Q = _as_mat(Q, n, n, "Q")
    R = _as_mat(R, m, m, "R")
    if T <= 0:
        raise ValueError(f"T 应为正数，收到 {T}")

    R_inv = np.linalg.inv(R)
    BRB = B @ R_inv @ B.T

    if rhs is None:
        def rhs(_tau, x):
            X = x.reshape(n, n)
            dX = X @ A + A.T @ X - X @ BRB @ X + Q
            return _sym(dX).reshape(-1) if symmetrize else dX.reshape(-1)

    kwargs = {}
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        if np.any(t_eval < -1e-12) or np.any(t_eval > T * (1 + 1e-12)):
            raise ValueError(f"t_eval 应落在 [0, T] = [0, {T}] 内")
        # tau = T - t：t 升序 -> tau 降序，solve_ivp 要求升序，所以翻转一次
        kwargs["t_eval"] = (T - t_eval)[::-1]

    sol = solve_ivp(rhs, (0.0, T), L.reshape(-1), method=method,
                    rtol=rtol, atol=atol, **kwargs)
    if sol.status != 0:
        raise RuntimeError(f"微分 Riccati 积分失败：{sol.message}")

    X = [_sym(sol.y[:, k].reshape(n, n)) if symmetrize else sol.y[:, k].reshape(n, n)
         for k in range(sol.y.shape[1])]
    t = T - sol.t
    order = np.argsort(t)                     # 反积分得到 t 降序，翻成升序
    t = t[order]
    P = [X[k] for k in order]
    K = [-R_inv @ B.T @ Pk for Pk in P]

    return {
        "t": t, "P": P, "P0": P[0], "PT": P[-1], "K": K,
        "ok": {
            "PT_minus_L": float(np.max(np.abs(P[-1] - L))),
            "symmetry_dev": float(max(np.max(np.abs(Pk - Pk.T)) for Pk in P)),
            "min_eig_P": float(min(np.min(np.linalg.eigvalsh(Pk)) for Pk in P)),
            "T": float(T),
        },
        "sol": sol, "n": n, "m": m,
    }
