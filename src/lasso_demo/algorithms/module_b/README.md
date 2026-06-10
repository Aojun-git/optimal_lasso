# 模块 B：Subgradient / ISTA / FISTA

在本目录新增：

- `subgradient.py`
- `ista.py`
- `fista.py`

每个求解器必须使用 `solver_template.py` 中的参数形式，并返回
`{"x": x, "history": history}`。使用
`lasso_demo.core.create_history()` 创建记录，用
`lasso_demo.core.record_history()` 追加每次迭代。

实现后：

1. 在开发脚本中直接调用一次算法，再调用
   `validate_solver_result(result, A.shape[1])`。
2. 在 `tests/` 中为该算法新增测试，固定检查返回接口和基本数值性质。
3. 注册到 `run_experiment()` 后，正式实验会自动再次检查接口。

`validate_solver_result()` 应由调用方使用，不需要写进算法函数内部。完整示例见
项目根目录 README 的“5.1”及 `tests/README.md`。

完成算法后，还可将它注册到 `experiments/run_mu_path.py`，由
`run_mu_sweep()` 批量比较不同 `mu`。`mu` 网格应优先通过
`make_mu_grid(A, b)` 根据当前数据生成。

算法核心必须手写迭代，不能调用 `sklearn.linear_model.Lasso`、
`scipy.optimize.minimize` 或 `cvxpy`。
