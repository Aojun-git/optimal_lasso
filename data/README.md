# 数据目录

- `raw/e2006/`：手动下载的 E2006-tfidf 原始压缩文件。
- `processed/synthetic/`：模块 A 生成的主实验、相关性、噪声和压力测试数据。
- `processed/diabetes/`：从 scikit-learn 加载并按训练/验证/测试切分的 Diabetes。
- `processed/e2006/`：从 E2006 原始文件筛选样本和特征后的缓存。

所有 `.npz` 文件都通过 `lasso_demo.data.load_dataset()` 加载，返回统一的
`DatasetBundle`。
