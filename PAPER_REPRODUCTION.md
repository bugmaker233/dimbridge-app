# DimBridge 论文算法复现说明

本分支在保留原有前后端功能和旧 `/api/predicate/regression` 接口的前提下，新增两条独立算法路径：

- `POST /api/predicate/paper-regression`
- `POST /api/predicate/rpi`

前端 `Predicate Mode` 可以在旧回归、论文回归和 RPI 之间切换。所有模式仍使用同一套投影框选、Predicate View、SPLOM、相关性分析、出现频率统计和导出流程。

## Predicate Regression

实现对应 [DimBridge 论文](https://arxiv.org/abs/2404.07386)第 4.2.1 节：

1. 使用公式 5 的可微代理函数。
2. 使用公式 6 的 BCE 与 `L1` 稀疏项，`a_j -> 0` 时删除覆盖全数据范围的因子。
3. 多框选和曲线框选使用公式 8-10，同时计算 `a` 与 `mu` 的平方 `L2` 平滑项。
4. 输出区间由 `mu_j +/- 1 / |a_j|` 得到，再映射回原始数据尺度。
5. `qualities.f1` 是论文代理函数在 0.5 阈值下的 F1；`predicate_f1` 是最终轴对齐区间合取规则的 F1。

论文未指定优化器、初始化、`b` 和各 `gamma` 的固定值。当前实现采用公开 Jupyter 代码中的标准化、选择区域质心/标准差初始化和带 Nesterov 动量的 SGD；`b` 默认为 4，所有参数均可通过 API 或前端修改。类别平衡 BCE 默认开启，以匹配作者公开实现对小框选区域的处理。

## Recursive Predicate Induction

实现依据 [PIXAL 论文](https://arxiv.org/abs/2205.11004)第 5.1 节和 DimBridge 第 4.2 节：

1. 每个数值因子先对唯一观测值排序，相邻观测值的中点作为分割点。
2. 直接枚举分割点构成的所有非平凡连续区间，不再使用分位数分箱。区间生成对应作者仓库的 [`notebooks/interval-tree.ipynb`](https://github.com/tiga1231/dim-bridge/blob/main/notebooks/interval-tree.ipynb) 中 `all_intervals` 的思路。
3. 对每个已扩展状态，都评估每个未使用因子的全部连续区间；首层保留 F1 大于 0 的区间，后续层只继续 F1 严格提高的合取分支。
4. 每个因子中导致相同样本集的区间只保留一个，子句顺序不同但内容相同的合取状态也会合并。
5. 搜索后删除“去掉后 F1 不降”的冗余子句，再按因子集去重，保留每个因子组合中 F1 最高的区间规则。
6. 候选解按 F1、因子数和支持度排序，所以可以同时返回多组替代因子集。

PIXAL 原始算法使用 Bayes factor；DimBridge 明确将评分替换为用户框选标签上的 F1。作者仓库中的 interval-tree notebook 提供了全区间生成和单一最佳区间的贪心原型，但没有集成多解 RPI，且原型使用的纯度损失不是 F1。因此，本实现的区间生成与官方 notebook 一致，多分支递归和 F1 评分则是依据 PIXAL/DimBridge 论文的可测试重建，不应表述为官方 RPI 源码的逐行移植。

默认每次框选最多生成 20,000 个状态，每个因子保留排名最高的 5 个不同子集区间继续搜索，每层保留 100 个分支，最终返回 20 个因子集候选。每次扩展仍会对所有精确区间评分，这两个分支参数只决定哪些高分区间继续向下组合。响应中的 `diagnostics.brushes[*]` 会分别标记每因子区间剪枝、状态截断、beam 剪枝、深度限制和返回数限制。`search_complete=true` 表示本次搜索没有受前四类递归限制影响；`max_solutions` 只限制输出，不改变该标记。

设置以下字段为 `null` 可以取消对应限制；前端中输入 `0` 会转换为 `null`：

- `max_intervals_per_factor`: 保留每个因子的全部精确区间分支。
- `beam_width`: 保留每层的全部递归分支。
- `max_states`: 不限制生成状态数。
- `max_solutions`: 返回全部因子集候选，该参数不影响搜索过程。
- `max_depth`: 递归到所有可用因子维度。

同时取消 `max_intervals_per_factor`、`beam_width`、`max_states` 和 `max_depth` 限制时，才是当前候选定义下的全递归。该模式会出现组合爆炸，适合在服务器上离线试验。

## 兼容返回结构

- `predicates`: 每次框选的首选解，继续驱动原有联动视图。
- `qualities`: 首选解质量。
- `candidate_solutions`: RPI 的全部保留候选，按框选分组。
- `diagnostics`: 优化参数、损失项或递归搜索统计。
- `algorithm`: 实际执行的算法标识。

## 本地验证

```bash
PYTHONPATH=backend /opt/anaconda3/envs/dimbridge-app/bin/python \
  -m unittest discover -s backend/tests -v

cd frontend
npm run build
```
