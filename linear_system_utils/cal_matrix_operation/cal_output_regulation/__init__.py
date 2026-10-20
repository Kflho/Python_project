"""输出调节：调节器方程 Pi S = A Pi + B Gamma + E P、C Pi = Q；静态增益 K_v；内模增广系统。"""

from .cal_output_regulation import (
    cal_internal_model,
    cal_internal_model_closed_loop,
    cal_output_regulation_gain,
    cal_regulation_solvability,
    cal_regulator_equations,
)

__all__ = [
    "cal_regulator_equations",
    "cal_output_regulation_gain",
    "cal_regulation_solvability",
    "cal_internal_model",
    "cal_internal_model_closed_loop",
]
