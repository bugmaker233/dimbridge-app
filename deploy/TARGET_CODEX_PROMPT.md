# Target-machine Codex prompt

请在目标服务器上完成 `/opt/dimbridge-app` 的部署，并严格遵守以下要求：

1. 先阅读 `/opt/dimbridge-app/deploy/DEPLOY_GPU.md`，再执行命令。
2. 不修改降维、谓词归纳和因子重要性算法；Predicate View 展示数量允许在 1–100 间调整，默认 20。
3. 使用 Python 3.10.x 创建 `/opt/dimbridge-app/backend/.venv`。
4. 先运行 `nvidia-smi`，根据实际 NVIDIA 驱动选择官方兼容的 PyTorch CUDA wheel；不要猜测 CUDA 版本。
5. PyTorch 安装完成后，再安装 `backend/requirements-server.txt`。
6. 必须运行 `deploy/verify_gpu.py`，并确认 `torch.cuda.is_available()` 为 `True`。
7. 后端只能启动一个 Uvicorn worker，因为当前数据集存储在进程内存中。
8. 不使用 `--reload`，使用提供的 systemd 服务模板。
9. 使用提供的 Nginx 模板，以同源方式提供前端并代理 `/api/`。
10. 完成后验证 `/api/health`、文件上传、UMAP、谓词归纳、全区间因子过滤、Predicate View 的 1–100 展示数量以及 SPLOM Top 6，并报告实际 Python、PyTorch、CUDA、GPU 和依赖版本。

若服务器已有域名、服务用户、Nginx 目录或安全策略，请修改模板中的占位值，但保留单 worker 和 `/api/` 同源代理约束。
