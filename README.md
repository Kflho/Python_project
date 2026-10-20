# Python 练习 + 线性系统算法库

这个仓库只放两件东西：

| 目录 | 是什么 |
|---|---|
| `python/` | Python 语言练习（`py_01`、`py_02`），跟着教程走，和线性系统无关 |
| `linear_system_utils/` | `linear_system_utils` 包：线性系统课程练习/实验复用的算法库 |

课程练习本身不在这个仓库，在 `D:\data\local\study\postgraduate\linear_system\projects\exercises`
（实验在隔壁 `experiments/`）。这里只提供它们 import 的算法和跑它们的环境。

## linear_system_utils

```
linear_system_utils/
  cal_matrix_parameters/     能控性矩阵、能控性指标集（判据类）
  cal_matrix_operation/      算法类
    cal_controllable_canonical_form.py   能控规范型：Q, p, T, Tau
    cal_pole_placement/                  极点配置 K
    cal_decoupling_control/              解耦控制 K, G、解耦矩阵 Tau
    cal_optimal_control/                 代数 / 微分 Riccati
```

约定：放"算一遍就复用"的算法，不放任何练习脚本；新算法先在这里实现，练习和实验只调用。
（`cal_smith_form.py`、`cal_pole_placement_formula.py`、`cal_differential_Riccati_function.py` 目前是
占位空文件，留待按章节补。）

## 环境

```powershell
# 仓库根已建好 .venv，包以 editable 方式装在里面；依赖变了重跑：
.\.venv\Scripts\python.exe -m pip install -e ".[exercises]"
```

`[exercises]` 额外带 scipy（个别练习解 Lyapunov 方程要用）。装好后在任何目录都能：

```powershell
.\.venv\Scripts\python.exe -c "from linear_system_utils.cal_matrix_parameters.cal_controllablility_matrix import cal_Q_c; print('ok')"
```

## 打包

`pyproject.toml` 只打包 `linear_system_utils*`，`python/` 下的练习脚本和 `.venv` 都不参与。
