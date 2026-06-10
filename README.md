# LASSO 多算法优化课程展示

项目目标是手写并比较：

- Subgradient
- ISTA
- FISTA
- Coordinate Descent
- ADMM

统一求解：

\[
\min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1
\]

## 当前状态

模块 A 已完成：

- 合成数据生成、Diabetes 预处理和 E2006 子集预处理入口。
- 公共数学函数、统一指标、绘图和 CSV 导出。
- solver 返回格式及接口校验。
- 单个 `mu` 的算法比较流程。
- 多个 `mu` 的网格扫描流程。
- 模块 B、C 的算法目录和测试框架。

五个正式 solver 已全部实现（详见下方"模块 B / C 完成说明"）。

```text
src/lasso_demo/algorithms/
├── module_b/       # Subgradient、ISTA、FISTA
└── module_c/       # Coordinate Descent、ADMM
```

算法核心必须手写迭代，不能调用现成 LASSO 或通用优化求解器代替。

## 开发流程

```mermaid
flowchart TD
    A["实现 solver"] --> B["增加小规模测试"]
    B --> C["运行 tests"]
    C --> D{"测试通过？"}
    D -- "否" --> A
    D -- "是" --> E["注册 solver"]
    E --> F["逐个加载实验数据集"]
    F --> G["make_mu_grid 生成 mu 网格"]
    G --> H["run_mu_sweep 扫描 mu"]
    H --> I["选择当前数据集的 mu"]
    I --> J["run_experiment 比较多个 solver"]
    J --> K["从 outputs 获取图表和日志"]
```

后续开发同学需要完成的工作就是：

1. 实现 solver。
2. 在小数据上测试。
3. 在多个数据集上扫描 `mu`。
4. 固定当前数据集的 `mu`，比较五种算法。
5. 汇总 `outputs/` 中的结果。

## 安装与数据

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### 使用 Git LFS 拉取 E2006 数据

处理后的 `e2006_subset.npz` 约 145 MiB，通过 Git LFS 存储。首次使用仓库前需要
安装并初始化 Git LFS。

Windows 可以安装 [Git for Windows](https://gitforwindows.org/)，其中通常已经
包含 Git LFS。确认安装：

```powershell
git lfs version
```

首次克隆推荐使用：

```powershell
git lfs install
git clone https://github.com/Aojun-git/optimal_lasso.git
cd optimal_lasso
git lfs pull
```

如果仓库已经克隆，但 `e2006_subset.npz` 只有几行 LFS 指针文本，进入项目目录后
执行：

```powershell
git lfs install
git lfs pull
```

验证大文件是否已经拉取：

```powershell
git lfs ls-files
Get-Item data\processed\e2006\e2006_subset.npz | Select-Object Name,Length
```

正常文件大小约为 152 MB；如果只有约 130 字节，说明仍是 LFS 指针，需要重新执行
`git lfs pull`。

已生成的数据位于：

```text
data/processed/
├── synthetic/
│   ├── synthetic_small.npz          # 开发测试
│   ├── synthetic_demo.npz           # 合成主实验
│   ├── synthetic_corr_*.npz         # 相关性实验
│   ├── synthetic_noise_*.npz        # 噪声实验
│   └── synthetic_stress.npz         # 规模拓展
└── diabetes/
    └── diabetes.npz                 # 训练/验证/测试
```

仓库已经提供处理后的 E2006 子集：

```text
data/processed/e2006/e2006_subset.npz
```

原始文件体积较大，不上传到 GitHub。只有需要重新生成 E2006 子集时才下载：

- [E2006.train.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2)
- [E2006.test.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.test.bz2)

共约 221.7 MiB，放入 `data/raw/e2006/`，然后运行：

```powershell
python -m experiments.prepare_data --prepare-e2006
```

如需重新生成所有非 E2006 数据，可运行：

```powershell
python -m experiments.prepare_data
```

## 实现 Solver

统一接口：

```python
def solver(A, b, mu, config):
    ...
    return {"x": x, "history": history}
```

建议使用：

```python
from lasso_demo.core import create_history, record_history
```

基本结构：

```python
def solver(A, b, mu, config):
    x = np.zeros(A.shape[1])
    history = create_history()
    start = time.perf_counter()

    for k in range(config.get("max_iter", 1000)):
        # 实现一次算法更新

        record_history(
            history,
            iteration=k,
            elapsed_time=time.perf_counter() - start,
            A=A,
            b=b,
            x=x,
            mu=mu,
            x_star=config.get("x_star"),
        )

    return {"x": x, "history": history}
```

history 必须包含：

```text
iteration, time, objective, sparsity, error
```

ADMM 还应记录 `primal_residual` 和 `dual_residual`。

## 测试 Solver

每个新算法都应在 `tests/` 中增加一个小规模测试：

```python
dataset = make_synthetic_lasso(m=20, n=30, k=3, seed=0)
result = solver(
    dataset.A_train,
    dataset.b_train,
    mu=0.05,
    config={"max_iter": 5, "x_star": dataset.x_star},
)

validate_solver_result(result, dataset.A_train.shape[1])
```

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

`validate_solver_result()` 检查 `x` 和 history 的格式。正式调用
`run_mu_sweep()` 或 `run_experiment()` 时也会自动检查。

更详细的测试说明见 [tests/README.md](tests/README.md)。

## 完整实验

完整实验不是只跑模板中的一个数据集，而是循环多个数据集和多个 `mu`。

推荐将所有 solver 注册为：

```python
solvers = {
    "Subgradient": subgradient,
    "ISTA": ista,
    "FISTA": fista,
    "Coordinate Descent": coordinate_descent,
    "ADMM": admm,
}
```

每个数据集都按以下流程运行：

```python
from lasso_demo.core import make_mu_grid
from lasso_demo.data import load_dataset
from lasso_demo.pipeline import run_experiment, run_mu_sweep

for path in dataset_paths:
    dataset = load_dataset(path)

    mu_values = make_mu_grid(
        dataset.A_train,
        dataset.b_train,
        n_values=10,
        min_ratio=1e-3,
        max_ratio=1.0,
    )

    run_mu_sweep(
        dataset,
        solvers,
        mu_values,
        solver_configs=solver_configs,
    )

    selected_mu = ...  # 根据扫描结果选择

    run_experiment(
        dataset,
        solvers,
        mu=selected_mu,
        solver_configs=solver_configs,
    )
```

`make_mu_grid()` 会根据当前数据计算合适的 `mu` 区间，不需要手写固定范围。

- 合成数据根据恢复误差、支持集 F1 和稀疏度选择 `mu`。
- Diabetes/E2006 根据验证 MSE 和稀疏度选择 `mu`。
- 测试集只用于参数确定后的最终结果。

参考入口：

```text
experiments/run_mu_path.py    # 扫描 mu
experiments/run_template.py   # 固定 mu 比较 solver
```

## 需要完成的实验

| 实验 | 数据集 | 比较内容 |
|---|---|---|
| Solver 主比较 | `synthetic_demo` | 五种算法的收敛、恢复误差、稀疏度和时间 |
| `mu` 比较 | `synthetic_demo`、Diabetes | 多个 `mu` 下的误差和非零系数数量 |
| 相关性实验 | `synthetic_demo`、`synthetic_corr_*` | 不同特征相关性下的结果 |
| 噪声实验 | `synthetic_demo`、`synthetic_noise_*` | 不同噪声下的恢复结果 |
| ADMM 参数实验 | `synthetic_demo` | 不同 `rho` 下的原始/对偶残差 |
| 真实数据实验 | Diabetes | 验证 MSE、测试 MSE、稀疏度和时间 |
| 拓展实验 | `synthetic_stress`、E2006 | 大规模数据上的性能 |

最终至少汇总：

- objective vs iteration。
- objective vs CPU time。
- relative objective gap。
- 真实系数与恢复系数。
- `mu` vs 误差和非零系数数量。
- ADMM 原始残差与对偶残差。
- 合成数据和 Diabetes 的结果表。

## 结果位置

```text
outputs/
├── figures/      # 实验图片
├── tables/       # 汇总 CSV
└── logs/         # 每次运行的 history CSV
```

`run_experiment()` 输出：

- `<dataset>_results.csv`
- `<dataset>_<solver>_history.csv`
- objective-iteration 图
- objective-time 图
- relative-gap 图

`run_mu_sweep()` 输出：

- `<dataset>_mu_sweep_results.csv`
- 每个 solver 的 `mu` 曲线
- 每个 solver 与 `mu` 组合的 history

## 项目目录

```text
lasso_show/
├── data/                       # 原始数据与处理后的数据
├── experiments/
│   ├── prepare_data.py         # 生成和预处理数据
│   ├── run_mu_path.py          # 扫描 mu
│   ├── run_template.py         # 固定 mu 比较 solver
│   └── run_all.py              # 一键运行全部实验
├── outputs/
│   ├── figures/                # 实验图片
│   ├── tables/                 # 汇总 CSV
│   └── logs/                   # 迭代 history
├── report/                     # 课程报告
├── slides/                     # 展示 PPT
├── src/lasso_demo/
│   ├── core.py                 # 数学工具、history、mu grid
│   ├── data.py                 # 数据生成和加载
│   ├── metrics.py              # 统一指标
│   ├── results.py              # CSV 导出
│   ├── plotting.py             # 通用绘图
│   ├── pipeline.py             # 完整运行流程
│   └── algorithms/
│       ├── module_b/           # Subgradient、ISTA、FISTA
│       └── module_c/           # Coordinate Descent、ADMM
├── tests/                      # 框架和 solver 测试
├── lasso_plan.html             # 项目计划
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 模块 B / C 完成说明

> 以下内容由模块 B、C 开发同学补充，不改动模块 A 的原始文档。

### 当前状态更新

所有五个 solver 已实现完毕：

| 模块 | 算法 | 文件 | 状态 |
|---|---|---|---|
| B | Subgradient | `module_b/subgradient.py` | ✅ 已完成 |
| B | ISTA | `module_b/ista.py` | ✅ 已完成 |
| B | FISTA | `module_b/fista.py` | ✅ 已完成 |
| C | Coordinate Descent | `module_c/coordinate_descent.py` | ✅ 已完成 |
| C | ADMM | `module_c/admm.py` | ✅ 已完成 |

### 一键运行全部实验

```powershell
# 激活环境
.\.venv\Scripts\Activate.ps1

# 运行全部实验（合成数据、Diabetes、E2006、相关性、噪声、ADMM残差、mu路径）
python -m experiments.run_all
```

运行完成后，`outputs/figures/` 和 `outputs/tables/` 中会生成全部图片和结果表格。

macOS / Linux 用户将 `.\.venv\Scripts\Activate.ps1` 替换为 `source .venv/bin/activate`。

### 生成的实验图片

| 编号 | 文件 | 内容 |
|---|---|---|
| Fig.1 | `Fig1_L1_L2_geometry.png` | L1 菱形 vs L2 圆形几何稀疏性示意 |
| Fig.2 | `Fig2_coefficient_comparison.png` | 真实系数 vs 5 种算法恢复系数 |
| Fig.3 | `Fig3_objective_iteration.png` | 目标函数值 vs 迭代次数 |
| Fig.4 | `Fig4_relative_gap.png` | 相对目标差半对数曲线 |
| Fig.5 | `Fig5_objective_time.png` | 目标函数值 vs CPU 时间 |
| Fig.6a | `Fig6a_synthetic_mu_path.png` | 合成数据 μ 路径（非零数 & 测试误差） |
| Fig.6b | `Fig6b_diabetes_mu_path.png` | Diabetes μ 路径 |
| Fig.7 | `Fig7_ADMM_residuals.png` | ADMM 原始/对偶残差（默认 rho） |
| Fig.7* | `Fig7_ADMM_rho_0.1.png` 等 | ADMM 不同 rho 残差对比 |
| Fig.8 | `Fig8_e2006_summary.png` | E2006 真实数据算法对比 |

### 生成的结果表格

| 文件 | 数据集 | 内容 |
|---|---|---|
| `synthetic_demo_results.csv` | 合成主实验 | 5 算法收敛、恢复误差、稀疏度 |
| `diabetes_results.csv` | Diabetes | 5 算法验证/测试 MSE、时间 |
| `e2006_results.csv` | E2006-tfidf | FISTA/CD/ADMM 大规模对比 |
| `synthetic_corr_050_results.csv` | 相关性 0.5 | 5 算法恢复结果 |
| `synthetic_corr_095_results.csv` | 相关性 0.95 | 高相关性下恢复更难 |
| `synthetic_noise_050_results.csv` | 噪声 σ=0.05 | 5 算法恢复结果 |
| `synthetic_noise_100_results.csv` | 噪声 σ=0.10 | 高噪声下恢复更难 |

### 实验主要结论

1. **近端方法显著优于次梯度法**：Subgradient 不产生稀疏解（nnz≈299/300），支持集 F1 仅 0.125；ISTA/FISTA/CD/ADMM 均能达到 F1≈0.655。
2. **FISTA 加速效果明显**：在目标值相同的前提下，FISTA 的 CPU 时间约为 ISTA 的 40%，迭代收敛速度更快。
3. **CD 实现直观且稳定**：手写 for 循环逐坐标更新，KKT 违背 < 6e-5，结果与 FISTA 完全一致。
4. **ADMM 小规模表现好**：在合成数据和 Diabetes 上，ADMM 与其他算法目标值差异 < 1.5e-5。支持自适应 rho 和 Woodbury 恒等式加速。
5. **ADMM 大规模收敛受 ρ 影响**：在 E2006（2400×5000）上，300 次迭代 ADMM 目标值（236）高于 FISTA/CD（194），这是 ADMM 的已知特性，可作为报告讨论点。
6. **高相关性和高噪声降低恢复质量**：相关性 0.95 → F1 从 0.655 降至 0.489；噪声 σ=0.10 → F1 从 0.655 降至 0.341。
7. **μ 越大解越稀疏**：Fig.6 清晰展示了 μ 与非零系数数量的反比关系。

### 算法实现要点

#### Coordinate Descent（`module_c/coordinate_descent.py`）

- 手写 `for j in range(n)` 逐坐标更新，不调用高级求解器
- 残差增量更新：加回旧贡献 → 计算 ρ_j → 软阈值 → 扣除新贡献
- 预计算列范数平方 `col_norm_sq`，避免重复计算
- 收敛判据：相对目标值改善 < `tol`

#### ADMM（`module_c/admm.py`）

- 变量分裂 `x = z`，x 更新用 Cholesky 分解（不使用 `np.linalg.inv`）
- z 更新用软阈值 `S_{μ/ρ}(x + u)`
- 双残差追踪：原始残差 `‖x - z‖`、对偶残差 `ρ‖z^k - z^{k-1}‖`
- **Woodbury 恒等式**：当 m < n 时，将 n×n 系统转化为 m×m 系统（E2006 上提速约 10 倍）
- **自适应 rho**（Boyd et al. 2010）：每 10 步根据原始/对偶残差比自动调整 ρ

### 注意事项

- E2006 数据通过 Git LFS 存储，需先安装 `git-lfs` 并执行 `git lfs pull`（详见上方"使用 Git LFS 拉取 E2006 数据"章节）
- 模块 B 的算法文件可由模块 B 同学替换，不影响模块 C 的代码和绘图
- 若需重新生成数据，运行 `python -m experiments.prepare_data`
