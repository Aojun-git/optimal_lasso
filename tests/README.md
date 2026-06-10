# tests 目录说明

## 1. 测试与正式实验的区别

测试用于快速回答“代码是否仍然满足约定”，正式实验用于回答“哪个算法更好”。

| 对比项 | 测试 | 正式实验 |
|---|---|---|
| 入口 | `python -m unittest discover -s tests -v` | `experiments/` 下的运行脚本 |
| 数据规模 | 很小，通常几十行、几十列 | 计划规定的完整数据或推荐子集 |
| 迭代次数 | 少，只需触发主要代码路径 | 足够收敛并形成实验结论 |
| 输出位置 | 临时目录，测试结束后删除 | `outputs/figures/`、`tables/`、`logs/` |
| 主要目的 | 找接口、形状、计算和文件输出错误 | 比较收敛速度、精度、稀疏度和时间 |

因此，不要在测试里跑 1000 次迭代、E2006 全量数据或生成最终展示图片。

## 2. 当前测试文件

### `test_core.py`

检查：

- `lasso_objective()` 的数值。
- `grad_smooth()` 的梯度。
- `soft_threshold()` 的软阈值结果。
- `record_history()` 是否正确追加记录。
- `validate_solver_result()` 是否接受正确格式、拒绝错误格式。

### `test_data.py`

检查：

- 固定随机种子能否生成完全相同的合成数据。
- 无噪声时是否严格满足 `b = A @ x_star`。
- `support` 是否与 `x_star` 的非零位置一致。
- `.npz` 保存后能否无损读回。
- Diabetes 是否得到固定的 264/89/89 训练、验证、测试切分。

### `test_metrics_results.py`

检查支持集 Precision、Recall、F1、测试 MSE 和结果 CSV 的固定字段。

### `test_plotting.py`

使用 Matplotlib 的非交互后端生成一张临时图片，确认绘图函数能够运行并写出
非空文件。它不检查图片是否适合最终 PPT，最终图仍需人工查看。

### `test_pipeline.py`

使用两步更新的 `mock_solver` 运行整个公共流程：

```text
小型合成数据
→ 运行模拟求解器
→ validate_solver_result
→ 统一指标
→ history CSV
→ 结果 CSV
→ 三张收敛图
```

这个测试证明模块之间能连接起来，不证明 `mock_solver` 是正确的 LASSO 算法。
此外还会用三个 `mu` 检查 `run_mu_sweep()` 是否生成三行汇总结果、三份 history
和一张参数曲线。

## 3. 新算法实现后应增加什么测试

例如新增 `src/lasso_demo/algorithms/module_b/ista.py` 后，建议新增
`tests/test_ista.py`，至少检查：

1. 返回值通过 `validate_solver_result()`。
2. `x` 中没有 `NaN` 或无穷值。
3. history 的 iteration 和 time 单调不减。
4. 用很小的稳定步长运行几步后，最终目标值不高于初始目标值。

示例：

```python
import unittest

import numpy as np

from lasso_demo.algorithms.module_b.ista import ista
from lasso_demo.core import validate_solver_result
from lasso_demo.data import make_synthetic_lasso


class IstaTests(unittest.TestCase):
    def test_small_problem(self):
        dataset = make_synthetic_lasso(m=20, n=30, k=3, seed=0)
        result = ista(
            dataset.A_train,
            dataset.b_train,
            0.05,
            {"max_iter": 5, "x_star": dataset.x_star},
        )

        validate_solver_result(result, dataset.A_train.shape[1])
        self.assertTrue(np.all(np.isfinite(result["x"])))
        self.assertLessEqual(
            result["history"]["objective"][-1],
            result["history"]["objective"][0],
        )
```

不要对所有算法都强制“每一步目标函数严格下降”。例如 FISTA 的目标值可能出现
局部非单调，次梯度法也不保证每一步下降。测试条件必须符合对应算法理论。

## 4. `validate_solver_result()` 的使用边界

它只验证统一接口：

- 有 `x` 和 `history`。
- `x` 长度与特征数相同。
- 必需 history 字段存在、非空且等长。

它不会验证：

- 算法公式是否正确。
- 目标值是否收敛。
- 最终解是否接近最优解。
- CPU 时间比较是否公平。

这些内容需要算法自己的数值测试和正式实验共同验证。

运行单个测试文件：

```powershell
python -m unittest discover -s tests -p "test_core.py" -v
```

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```
