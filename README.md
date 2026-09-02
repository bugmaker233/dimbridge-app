# DimBridge Standalone App

DimBridge Standalone App 是从原 Jupyter 小组件迁移出的前后端可视化分析系统。项目保留原有 Projection View、Predicate View、SPLOM View 的交互链路，并通过 FastAPI 后端提供数据上传、降维投影和谓词解释计算能力。

## 功能概览

- 支持上传 CSV、XLSX、XLS 数据文件。
- 支持选择数值型 factor columns 和 target/color column。
- 支持 PCA、t-SNE、UMAP、Supervised UMAP 二维降维投影。
- 支持投影视图缩放、视图重置、框选、对比刷选和曲线刷选。
- 支持 Data Extent 和 Predicate Regression 两种谓词解释模式。
- 支持投影视图、谓词区间视图和 SPLOM 矩阵视图联动。
- 前端使用 React + Vite，后端使用 FastAPI + pandas + scikit-learn + umap-learn + PyTorch。

## 目录结构

```text
dimbridge-app/
  backend/
    app/
      api/                 FastAPI 路由
      core/                谓词回归核心计算
      services/            数据集、投影、预处理服务
      main.py              后端应用入口
      schemas.py           API 请求/响应模型
    requirements.txt       后端 Python 依赖
    README.md              后端接口说明
  frontend/
    src/
      components/          React 页面组件
      legacy/              从 Jupyter widget 迁移的可视化逻辑
      pages/               工作台页面
      services/            前端 API 调用封装
      styles/              页面样式
    package.json           前端依赖和脚本
    vite.config.js         Vite 开发服务器和代理配置
```

## 环境要求

后端建议环境：

```text
Python >= 3.10
pip >= 23
```

前端建议环境：

```text
Node.js >= 18
npm >= 9
```

当前项目在已有 `visulization` conda 环境中验证过。迁移到新机器时，可以使用 conda、venv 或系统 Python，只要安装 `backend/requirements.txt` 即可。

## 后端安装与启动

进入后端目录：

```bash
cd dimbridge-app/backend
```

创建并激活 Python 环境。使用 conda 示例：

```bash
conda create -n dimbridge-app python=3.10
conda activate dimbridge-app
```

如果继续使用已有环境，例如 `visulization`：

```bash
conda activate visulization
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动后端：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/api/health
```

根路径返回：

```json
{"message":"DimBridge backend is running."}
```

这是正常现象，接口文档应访问 `/docs`。

## 前端安装与启动

进入前端目录：

```bash
cd dimbridge-app/frontend
```

安装依赖：

```bash
npm install
```

启动开发服务器：

```bash
npm run dev
```

默认前端地址：

```text
http://127.0.0.1:5173
```

开发模式下，`frontend/vite.config.js` 会把 `/api` 请求代理到：

```text
http://localhost:8000
```

因此本地开发时需要同时启动后端 `8000` 端口和前端 `5173` 端口。

## 生产构建

前端构建：

```bash
cd dimbridge-app/frontend
npm run build
```

构建产物输出到：

```text
dimbridge-app/frontend/dist/
```

本地预览构建产物：

```bash
npm run preview -- --host 0.0.0.0 --port 4173
```

部署到服务器时建议采用同源代理方式：

```text
浏览器 -> Nginx/网关
  /      -> frontend/dist
  /api   -> FastAPI backend:8000
```

如果前端和后端不是同源部署，需要在 `backend/app/main.py` 的 CORS 配置中加入前端域名。

## 数据要求

上传文件要求：

- 文件格式：`.csv`、`.xlsx`、`.xls`
- 数据不能为空。
- 至少包含一个数值型字段作为 factor column。
- 降维计算会自动过滤 factor/target 中无法转为有效数值的行。
- Dataset 目前保存在后端进程内存中，后端重启后已上传数据会丢失。

## 降维方法与参数

前端提供四种投影模式：

```text
PCA
t-SNE
UMAP
Supervised UMAP
```

参数说明：

| 方法 | 前端参数 | 说明 |
| --- | --- | --- |
| PCA | Standardize | 固定输出二维坐标，适合快速线性降维。 |
| t-SNE | perplexity, learning_rate, max_iterations, Standardize | 适合观察局部聚类结构，计算时间通常比 PCA 更长。 |
| UMAP | n_neighbors, min_dist, Standardize | 无监督 UMAP，适合通用二维流形可视化。 |
| Supervised UMAP | n_neighbors, min_dist, target_weight, Standardize | 使用 target column 作为监督信号，target column 必填且当前要求为数值型。 |

其中 target column 在 PCA、t-SNE、无监督 UMAP 下只用于颜色编码；在 Supervised UMAP 下同时参与投影计算。

## 主要 API

健康检查：

```text
GET /api/health
```

上传数据集：

```text
POST /api/dataset/upload
```

获取数据集摘要：

```text
GET /api/dataset/{dataset_id}
```

运行降维投影：

```text
POST /api/projection/run
```

兼容旧 UMAP 接口：

```text
POST /api/projection/umap
```

谓词解释：

```text
POST /api/predicate/data-extent
POST /api/predicate/regression
```

完整接口结构可在后端启动后通过 Swagger 查看：

```text
http://127.0.0.1:8000/docs
```

## 常见问题

后端根路径只显示 `DimBridge backend is running.`

这是正常的健康提示。接口说明请打开 `/docs`，前端界面请打开 Vite 地址。

前端页面空白或接口请求失败

确认后端已启动在 `8000` 端口，并确认前端开发服务器的 `/api` 代理没有被改动。浏览器控制台如出现 CORS 错误，需要检查 `backend/app/main.py` 中的 `allow_origins`。

UMAP 首次运行较慢

UMAP 依赖 numba，首次运行可能有编译开销。项目已在后端投影服务中把 `NUMBA_CACHE_DIR` 指向临时目录，便于服务器环境运行。

t-SNE 小样本或单特征数据

后端会自动调整 t-SNE perplexity，并在单特征数据时使用 random 初始化，避免 scikit-learn 的 PCA 初始化限制。

大数据集交互变慢

当前版本将上传数据保存在内存中，并在浏览器端渲染 legacy 交互视图。大数据集建议先抽样或减少 factor columns。

## 迁移检查清单

迁移到新环境时建议依次检查：

1. 安装 Python 3.10+ 和 Node.js 18+。
2. 安装后端依赖：`python -m pip install -r backend/requirements.txt`。
3. 安装前端依赖：`cd frontend && npm install`。
4. 启动后端并打开 `/api/health`。
5. 启动前端并上传一个 CSV 或 XLSX 测试文件。
6. 依次测试 PCA、t-SNE、UMAP、Supervised UMAP。
7. 框选投影视图，确认 Predicate View 和 SPLOM View 联动更新。
8. 生产部署时配置 `/api` 反向代理或 CORS。

## 软著整理建议

用于软件著作权材料整理时，可将以下内容作为项目说明基础：

- 软件名称：DimBridge 可交互降维分析与谓词解释系统。
- 软件架构：React 前端 + FastAPI 后端。
- 核心功能：数据上传、降维投影、交互刷选、谓词解释、SPLOM 联动分析。
- 主要算法：PCA、t-SNE、UMAP、Supervised UMAP、Predicate Regression。
- 主要源码目录：`backend/app`、`frontend/src/components`、`frontend/src/legacy`。
