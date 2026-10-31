"""cal_differential_Riccati_function 的回归测试。

判据有两条，都不依赖被测代码：
1. 标量情形 -p' = 2ap - p^2 + 1、p(T) = 0 有闭式解
       p(0) = tanh(mu T) / (mu - a tanh(mu T)),   mu = sqrt(a^2 + 1),
   对 a = 1、T = 3 比对；
2. T -> inf 时数值解趋于代数 Riccati 方程的正定解（双积分器，Q = diag(1,0)、R = 1 时
   P_inf = [[sqrt(2), 1], [1, sqrt(2)]]），并检查 P 对称、正定、P(T) = L。

跑法（不装 pytest 也能跑）：
    .\\.venv\\Scripts\\python.exe tests\\test_differential_riccati.py
"""
import numpy as np

from linear_system_utils.cal_matrix_operation.cal_optimal_control.cal_differential_Riccati_function import (
    cal_differential_Riccati_function,
)


def test_scalar_closed_form():
    a, T = 1.0, 3.0
    mu = np.sqrt(a * a + 1.0)
    p0_want = np.tanh(mu * T) / (mu - a * np.tanh(mu * T))

    res = cal_differential_Riccati_function(
        np.array([[a]]), np.array([[1.0]]), np.array([[0.0]]),
        np.array([[1.0]]), np.array([[1.0]]), T=T, t_eval=[0.0, T])
    p0 = float(res["P0"][0, 0])

    assert abs(p0 - p0_want) < 1e-9, f"p(0) = {p0}，闭式 {p0_want}"
    assert res["ok"]["PT_minus_L"] < 1e-12, res["ok"]
    # 终端条件 p(T) = 0
    assert abs(float(res["PT"][0, 0])) < 1e-12, res["PT"]


def test_tendsto_algebraic_riccati():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([1.0, 0.0])
    R = np.array([[1.0]])
    L = np.eye(2)
    P_inf = np.array([[np.sqrt(2.0), 1.0], [1.0, np.sqrt(2.0)]])

    err_prev = np.inf
    for T in (1.0, 5.0, 20.0):
        res = cal_differential_Riccati_function(A, B, L, Q, R, T=T)
        P0 = res["P0"]
        assert np.max(np.abs(P0 - P0.T)) < 1e-12, "P0 不对称"
        assert np.min(np.linalg.eigvalsh(P0)) > 0, "P0 不正定"
        err = float(np.max(np.abs(P0 - P_inf)))
        assert err < err_prev, f"T = {T} 时没有更接近代数 Riccati 解"
        err_prev = err
    assert err_prev < 1e-6, f"T = 20 时仍差 {err_prev}，没收敛到代数 Riccati 解"


def test_rejects_bad_time_grid():
    A = np.eye(2)
    args = (A, np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    for T in (0.0, -1.0):
        try:
            cal_differential_Riccati_function(*args, T=T)
        except ValueError:
            pass
        else:
            raise AssertionError(f"T = {T} 应该报 ValueError")
    try:
        cal_differential_Riccati_function(*args, T=1.0, t_eval=[-0.5, 0.5])
    except ValueError:
        pass
    else:
        raise AssertionError("t_eval 跑出 [0, T] 应该报 ValueError")


if __name__ == "__main__":
    for fn in (test_scalar_closed_form, test_tendsto_algebraic_riccati,
               test_rejects_bad_time_grid):
        fn()
        print(f"[ok] {fn.__name__}")
    print("微分 Riccati 回归测试全部通过")
