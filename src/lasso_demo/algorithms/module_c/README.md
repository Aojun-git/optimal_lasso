# 模块 C：Coordinate Descent / ADMM / 最终绘图

在本目录新增：

- `coordinate_descent.py`
- `admm.py`

求解器接口和 history 规范与模块 B 相同。ADMM 将原始残差和对偶残差
传给 `record_history()` 的 `primal_residual`、`dual_residual` 参数。

每个算法实现后，应在 `tests/` 中增加小规模接口测试，并调用
`validate_solver_result(result, A.shape[1])`。正式注册到
`run_experiment()` 后，流水线会自动执行同一检查，不需要在算法函数内部重复调用。
完整示例见项目根目录 README 的“5.1”及 `tests/README.md`。

完成算法后，还可将它注册到 `experiments/run_mu_path.py`，通过
`run_mu_sweep()` 批量比较不同 `mu`。

通用收敛图已经在 `lasso_demo.plotting` 和
`lasso_demo.pipeline.run_experiment()` 中实现。本模块只需补充 ADMM
残差图、最终系数对比图及展示所需的组合排版。
