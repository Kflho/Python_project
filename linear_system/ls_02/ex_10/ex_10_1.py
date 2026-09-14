import numpy as np
import sympy as sp
from numpy.linalg import inv
nu_1=3
nu_2=1

A = np.array([[-1, -1, 3, 4], 
              [-2, 0, 2, 3], 
              [-2, 1, -1, -1], 
              [0, -1, 2, 3]])

b_1 = np.array([[1], [1], [1], [0]])
b_2 = np.array([[0], [-1], [-2], [1]])
B = np.hstack([b_1, b_2])

# 矩阵乘法用 @
Ab1 = A @ b_1
A2b1 = A @ A @ b_1   # 等价于 (A^2) b_1

# 水平拼接成矩阵
Q_1 = np.hstack([b_1, Ab1, A2b1])   # 形状 (4, 3)
Q_2 = b_2                           # 形状 (4, 1)

Q = np.hstack([Q_1, Q_2])           # 形状 (4, 4)
print(Q)
Q_inv=inv(Q)
print(Q_inv)
p_1=Q_inv[nu_1-1, :]
print(p_1)

p_2=Q_inv[nu_1+nu_2-1, :]
print(p_2)
T_1 = np.vstack([p_1,p_1 @ A, p_1 @ A @ A])
T_2 = np.vstack([p_2])
T = np.vstack([T_1, T_2])
print(T)
T_inv = inv(T)
A_new = T @ A @ T_inv
B_new = T @ B
print(A_new)
print(B_new)

