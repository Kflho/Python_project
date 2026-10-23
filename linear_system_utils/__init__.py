"""linear_system_utils：线性系统课程练习用到的可复用计算函数。

按功能分两个子包：

    cal_matrix_parameters   能控性矩阵、能控性指标集（判据类）
    cal_matrix_operation    能控规范型、极点配置、解耦控制、最优控制、轨迹跟踪、观测器（算法类）

定位：只放"算一遍就复用"的算法，不放任何练习脚本。练习在
`D:\\data\\local\\study\\postgraduate\\linear_system\\projects\\exercises`，
本包以 editable 方式装进 `Python\\projects\\.venv`，练习脚本直接
`from linear_system_utils... import ...` 即可，不需要 sys.path 技巧。
"""

__version__ = "0.1.0"
