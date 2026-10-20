"""输出调节：调节器方程 Pi S = A Pi + B Gamma + E P、C Pi = Q，静态反馈增益 K_v。"""

from .cal_output_regulation import (
    cal_output_regulation_gain,
    cal_regulation_solvability,
    cal_regulator_equations,
)

__all__ = [
    "cal_regulator_equations",
    "cal_output_regulation_gain",
    "cal_regulation_solvability",
]
