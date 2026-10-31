"""cal_smith_form 的回归测试：课程例 2.7.1 的课件答案 + 三种形状的判据 + 该报错的错法。

金标准是课程例 2.7.1（第 2 章 2.7 节）：课件直接给了 T、Gamma、G 和最终 6x6 的
[P N; Q D]，这里由它反推 (A, B)，再把库算出来的 T、Gamma、G、T_r 逐元素对照。
判据是

    [A - sI_n, B] T_r(s) = [I_n, 0],     det T_r(s) = 1

跑法（不装 pytest 也能跑，跑完有断言、失败就非零退出）：

    & 'D:\\data\\local\\software\\programming\\Python\\projects\\.venv\\Scripts\\python.exe' `
      'D:\\data\\local\\software\\programming\\Python\\projects\\tests\\test_cal_smith_form.py'
"""
import sympy as sp

from linear_system_utils.cal_matrix_operation.cal_smith_form import (
    cal_smith_form,
    cal_smith_form_check,
)

s = sp.symbols("s")


def _course_example():
    """课程例 2.7.1：(A, B) 由课件给的 T、Gamma、G 反推"""
    T_kb = sp.Matrix([[1, 0, 0, -1], [0, -1, 0, 2], [-1, -1, 1, 2], [-1, 2, 0, -2]])
    Gam_kb = sp.Matrix([[1, -1], [0, 1]])
    G_kb = sp.Matrix([[-2, -2, -1, -1], [-3, 0, 0, -2]])
    A0 = sp.diag(sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), sp.Matrix([[0]]))
    B0 = sp.diag(sp.Matrix([[0], [0], [1]]), sp.Matrix([[1]]))
    A_kb = sp.expand(T_kb.inv() * (A0 - B0 * G_kb) * T_kb)
    B_kb = sp.expand(T_kb.inv() * (B0 * Gam_kb))
    # 课件印的最终答案（OCR 校对过：2s+2、3s+2、s^2+s+2、2s+1、s^3-s^2-2s-5、-s(s-3)）
    Tr_kb = sp.Matrix([
        [2, 0, 0, -2, 2 * s + 2, 1],
        [3, 0, 0, -3, 3 * s + 2, 2],
        [s + 1, -1, 0, 1 - s, s ** 2 + s + 2, 1],
        [2, 0, 0, -2, 2 * s + 1, 1],
        [s ** 2 - s - 4, 2 - s, 1, -s ** 2 + 3 * s, s ** 3 - s ** 2 - 2 * s - 5, s - 3],
        [-1, 2, 0, -2, -3, s - 2],
    ])
    return A_kb, B_kb, T_kb, Gam_kb, G_kb, Tr_kb


def test_course_example_matches_slides():
    A_kb, B_kb, T_kb, Gam_kb, G_kb, Tr_kb = _course_example()
    res = cal_smith_form(A_kb, B_kb)

    assert res["mu"] == [3, 1], res["mu"]
    assert sp.simplify(res["T"] - T_kb).is_zero_matrix, "T 与课件不一致"
    assert sp.simplify(res["Gamma"] - Gam_kb).is_zero_matrix, "Gamma 与课件不一致"
    assert sp.simplify(res["G"] - G_kb).is_zero_matrix, "G 与课件不一致"
    assert sp.simplify(res["T_r"] - Tr_kb).is_zero_matrix, "T_r 与课件答案逐元素不一致"


def _other_cases():
    return {
        "习题例题 mu=(3,1)": (
            sp.Matrix([[-1, -1, 3, 4], [-2, 0, 2, 3], [-2, 1, -1, -1], [0, -1, 2, 3]]),
            sp.Matrix([[1, 0], [1, -1], [1, -2], [0, 1]]),
            [3, 1],
        ),
        "SISO n=4": (
            sp.Matrix([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-2, -3, -1, -4]]),
            sp.Matrix([[0], [0], [0], [1]]),
            [4],
        ),
        "m=3 n=6": (
            sp.Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [-1, -2, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, -2, -3, 0]]),
            sp.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]]),
            [1, 2, 3],
        ),
    }


def test_other_shapes_pass_both_criteria():
    """n x n / SISO / m=3 三种形状：形状对、[A-sI B]T_r == [I 0]、det T_r == 1"""
    for name, (A, B, mu) in _other_cases().items():
        r = cal_smith_form(A, B)
        assert r["mu"] == mu, f"{name}: mu = {r['mu']}，应为 {mu}"
        chk = cal_smith_form_check(A, B, r["T_r"])
        assert chk["shape"], f"{name}: T_r 不是 (n+m) x (n+m)"
        assert chk["first"], f"{name}: [A-sI B]T_r != [I 0]，残差 {chk['residual']}"
        assert chk["det"], f"{name}: det T_r != 1，偏差 {chk['det_expr']}"


def test_rejects_wrong_index_and_uncontrollable():
    """指标集给错 / 系统不能控，必须抛 ValueError（不能默默算出一堆错东西）"""
    A2, B2, _ = _other_cases()["习题例题 mu=(3,1)"]
    with_error = [
        ("mu 给错", dict(A=A2, B=B2, mu=[2, 2])),
        ("不能控", dict(A=sp.Matrix([[1, 0], [0, 1]]), B=sp.Matrix([[1], [0]]))),
    ]
    for name, kwargs in with_error:
        try:
            cal_smith_form(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name}: 应该抛 ValueError，却算出来了")


if __name__ == "__main__":
    for fn in (test_course_example_matches_slides,
               test_other_shapes_pass_both_criteria,
               test_rejects_wrong_index_and_uncontrollable):
        fn()
        print(f"[ok] {fn.__name__}")
    print("cal_smith_form 回归测试全部通过")
