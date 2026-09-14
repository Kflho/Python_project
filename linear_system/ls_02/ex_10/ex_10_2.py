import sympy as sp
from sympy import BlockMatrix

s = sp.symbols('s')

# ---------- 基本矩阵 ----------
G = sp.Matrix([[-1, 2, 0, 1],
               [0, 0, 0, -1]])          # 2x4

A = sp.Matrix([[-1, -1, 3, 4],
               [-2, 0, 2, 3],
               [-2, 1, -1, -1],
               [0, -1, 2, 3]])          # 4x4

b_1 = sp.Matrix([[1], [1], [1], [0]])   # 4x1
b_2 = sp.Matrix([[0], [-1], [-2], [1]]) # 4x1
B = sp.Matrix.hstack(b_1, b_2)               # 4x2

# 行向量要用 [[...]]
p_1 = sp.Matrix([[1, -1, 0, -1]])       # 1x4
p_2 = sp.Matrix([[0, -1, 1, 2]])        # 1x4

Tau = sp.Matrix.vstack(p_1 * A * A * B,
                 p_2 * B)              # 2x2
print(Tau)

T = sp.Matrix([[1, -1, 0, -1],
               [1, 0, -1, -2],
               [1, 0, 0, -1],
               [0, -1, 1, 2]])          # 4x4

# ---------- 多项式矩阵（根据你的实际定义修改）----------
# 这里只是语法示例，具体内容请你按数学公式填写
# D_0, Q_0, N_0 的维度必须与分块矩阵匹配

# 假设 D_0 是 2x2
D_0 = sp.diag(s**3, s)                  # 2x2 对角矩阵

# 假设 Q_0 是 2x4（示例，请根据实际修改）
Q_0 = sp.Matrix([[s**2, s, 1, 0],
                 [0, 1, s, 1]])         # 2x4

# 假设 N_0 是 4x2（示例）
N_0 = sp.Matrix([[1, 0],
                 [s, 0],
                 [s**2, 0],
                 [0, 1]])               # 4x2

# 假设 P_0 是 4x4（你原来写了 P_0 但未定义，这里用零矩阵占位）
P_0 = sp.zeros(4, 4)                    # 请替换为真实矩阵

# ---------- 分块矩阵 ----------
# 第一个分块矩阵：6x6
# [ T.inv()      , 0(4x2) ]
# [ Tau.inv()*G  , Tau.inv() ]
block1 = BlockMatrix([
    [T.inv(), sp.zeros(4, 2)],
    [Tau.inv() * G, Tau.inv()]
])

# 第二个分块矩阵：6x6
# [ P_0 , N_0 ]
# [ Q_0 , D_0 ]
block2 = BlockMatrix([
    [P_0, N_0],
    [Q_0, D_0]
])

# 第三个分块矩阵：6x6
# [ T , 0(4x2) ]
# [ 0(2x4) , eye(2) ]
block3 = BlockMatrix([
    [T, sp.zeros(4, 2)],
    [sp.zeros(2, 4), sp.eye(2)]
])

# 相乘
T_new = block1 * block2 * block3
print(T_new.as_explicit())