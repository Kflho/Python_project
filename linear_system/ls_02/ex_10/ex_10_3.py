"""把 ex_10 的例题接到 utils 里的函数上跑一遍。

内容和 ex_10_1.py 一样（同一组 A、b_1、b_2，nu_1=3、nu_2=1），
只是 Q、p、T 不再手写，全部交给函数算，最后和手算结果对照。
"""
import pathlib
import sys

import numpy as np

# utils 是包，linear_system 是它的根目录，加进去后无论从哪个目录运行都能 import
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from utils.cal_T import cal_A_new_and_B_new, cal_Q, cal_T, cal_p

nu_1 = 3
nu_2 = 1
nu = [nu_1, nu_2]

A = np.array([[-1, -1, 3, 4],
              [-2, 0, 2, 3],
              [-2, 1, -1, -1],
              [0, -1, 2, 3]], dtype=float)

b_1 = np.array([[1], [1], [1], [0]], dtype=float)
b_2 = np.array([[0], [-1], [-2], [1]], dtype=float)
B = np.hstack([b_1, b_2])

np.set_printoptions(precision=4, suppress=True)

# ---------- 调函数 ----------
Q = cal_Q(A, B, nu)
p = cal_p(Q, nu)
T = cal_T(nu, A, B)
A_new, B_new = cal_A_new_and_B_new(A, B, T)

print("Q =\n", Q)
print("Q_inv =\n", np.linalg.inv(Q))
print("p =\n", p)
print("T =\n", T)
print("A_new =\n", A_new)
print("B_new =\n", B_new)

# ---------- 和手算结果对照 ----------
Q_hand = np.hstack([b_1, A @ b_1, A @ A @ b_1, b_2])          # ex_10_1.py 里的 Q
T_hand = np.array([[1, -1, 0, -1],
                   [1, 0, -1, -2],
                   [1, 0, 0, -1],
                   [0, -1, 1, 2]], dtype=float)               # ex_10_2.py 里的 T
Tau = np.vstack([p[0] @ A @ A @ B, p[1] @ B])                  # ex_10_2.py 里的 Tau

checks = {
    "Q 与手算一致": np.allclose(Q, Q_hand),
    "p_1 == Q_inv[nu_1-1, :]": np.allclose(p[0], np.linalg.inv(Q)[nu_1 - 1, :]),
    "p_2 == Q_inv[nu_1+nu_2-1, :]": np.allclose(p[1], np.linalg.inv(Q)[nu_1 + nu_2 - 1, :]),
    "T == 手算 T": np.allclose(T, T_hand),
    "T 可逆": np.allclose(T @ np.linalg.inv(T), np.eye(4)),
    "Tau == B_new 的第 3、4 行": np.allclose(Tau, B_new[[nu_1 - 1, nu_1 + nu_2 - 1], :]),
    "A_new 是能控标准型": np.allclose(A_new, [[0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [1, -2, 0, -1],
                                              [0, 0, 0, 1]]),
}

print()
for name, ok in checks.items():
    print(f"  {name}: {ok}")
print("\n全部通过" if all(checks.values()) else "\n有对不上的地方")
