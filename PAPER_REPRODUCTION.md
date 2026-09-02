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

1. 每个数值因子按分位数分箱，所有箱区间构成单子句基础谓词。
2. 所有基础谓词并行成为递归起点。
3. 每个状态与其他维度的基础谓词进行合取。
4. 只有 F1 严格提高的子状态继续递归。
5. 子句顺序不同但内容相同的合取规则按无序签名合并。
6. 无法继续提高的状态成为候选解，最终按 F1、因子数和支持度排序。

PIXAL 原始算法使用 Bayes factor；DimBridge 明确将评分替换为用户框选标签上的 F1。论文没有发布 RPI 官方源码，也没有规定数值分箱方式，因此这里是按论文描述进行的可测试重建，不是逐行源码移植。

默认每次框选最多评估 20,000 个状态并返回 20 个候选。响应中的 `diagnostics.brushes[*].truncated` 表示是否触发搜索预算。设置以下字段为 `null` 可以取消对应限制：

- `max_states`: 不限制评估状态数。
- `max_solutions`: 返回全部终止候选。
- `max_depth`: 递归到所有可用因子维度。

取消限制可能产生组合爆炸，RPI 本身在 DimBridge 论文中也被定位为离线算法。

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
