"""轨迹跟踪：耦合矩阵 Gamma、控制律 u = K x + Gamma^{-1} v^*、参考前馈 v^*。"""

from .cal_tracking_control import (
    cal_coupling_matrix,
    cal_error_dynamics,
    cal_reference_vector,
    cal_tracking_control,
)

__all__ = [
    "cal_coupling_matrix",
    "cal_tracking_control",
    "cal_reference_vector",
    "cal_error_dynamics",
]
