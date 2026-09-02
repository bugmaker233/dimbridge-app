# DimBridge GPU server deployment

本说明用于将 DimBridge 部署到带 NVIDIA GPU 的目标服务器。默认安装目录为
`/opt/dimbridge-app`，默认服务用户为 `dimbridge`。如果目标环境不同，请同步修改
systemd 和 Nginx 模板。

## 1. Important constraints

- Backend runtime: Python 3.10.x.
- Frontend runtime: static files in `frontend/dist`; Node.js is not required at runtime.
- GPU acceleration applies to PyTorch predicate regression only. PCA, t-SNE and UMAP use CPU in the current implementation.
- Uploaded datasets are currently stored in backend process memory.
- Run exactly one Uvicorn worker. Multiple workers can return `Dataset not found` because they do not share memory.
- Backend restart clears uploaded datasets.
- Do not use `--reload` in production.
- The bundled frontend expects same-origin API requests under `/api`.

## 2. Package contents

The deployment archive contains:

```text
dimbridge-app/
  backend/app/                    FastAPI source
  backend/requirements.txt       Development dependency ranges
  backend/requirements-server.txt Tested server dependencies, excluding PyTorch
  frontend/dist/                 Pre-built static frontend
  frontend/src/                  Frontend source
  frontend/package.json
  frontend/package-lock.json
  deploy/                         This guide and service templates
```

Mac-specific `node_modules`, Python environments, caches, local sample data and unrelated scripts are excluded.

## 3. Extract and prepare directories

Run as an administrator and adapt the archive location as needed:

```bash
mkdir -p /opt
tar -xzf dimbridge-app-gpu-deploy.tar.gz -C /opt

useradd --system --user-group --home-dir /opt/dimbridge-app --shell /sbin/nologin dimbridge || true
chown -R dimbridge:dimbridge /opt/dimbridge-app
```

If the NVIDIA devices require membership in `video` or `render`, add the service user to the groups that exist on the target server:

```bash
getent group video >/dev/null && usermod -aG video dimbridge
getent group render >/dev/null && usermod -aG render dimbridge
```

Inspect `ls -l /dev/nvidia*` first; group names can differ between servers.

## 4. Create the Python environment

Python 3.10 must already be available on the target machine:

```bash
cd /opt/dimbridge-app/backend
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Confirm the interpreter before installing packages:

```bash
python --version
which python
```

Expected path:

```text
/opt/dimbridge-app/backend/.venv/bin/python
```

## 5. Install the CUDA build of PyTorch

First inspect the actual GPU and driver:

```bash
nvidia-smi
```

Then use the official PyTorch installation selector to choose a wheel compatible with the target driver. Do not infer the wheel solely from the `CUDA Version` label in `nvidia-smi`, and do not copy the Mac PyTorch package.

Install only PyTorch; torchvision and torchaudio are not required by this project. The command has this general form, but `VERSION` and `cuXXX` must be selected for the target server:

```bash
python -m pip install "torch==VERSION" --index-url https://download.pytorch.org/whl/cuXXX
```

After PyTorch is installed, install the remaining tested dependencies:

```bash
python -m pip install -r requirements-server.txt
```

Run the included GPU smoke test:

```bash
cd /opt/dimbridge-app
backend/.venv/bin/python deploy/verify_gpu.py
```

Deployment must not continue as a GPU deployment unless it reports:

```text
CUDA available: True
```

If it is `False`, inspect the NVIDIA driver, installed PyTorch wheel, service-user device permissions and `CUDA_VISIBLE_DEVICES`.

## 6. Backend smoke test

Run the backend temporarily in the foreground:

```bash
cd /opt/dimbridge-app/backend
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

From another terminal:

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

Confirm that the response contains:

```json
"status": "ok",
"torch_cuda_available": true
```

Stop the foreground process after the check.

## 7. Install the systemd service

Review `deploy/dimbridge-backend.service`. Change `User`, `Group`, `WorkingDirectory` and `ExecStart` if the deployment path differs.

```bash
cp /opt/dimbridge-app/deploy/dimbridge-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now dimbridge-backend
systemctl status dimbridge-backend
journalctl -u dimbridge-backend -n 100 --no-pager
```

The service intentionally uses `--workers 1`.

## 8. Install the Nginx site

The archive already contains a production frontend build. No Node.js installation is required unless rebuilding the frontend on the target machine.

Review `deploy/nginx-dimbridge.conf` and replace `dimbridge.example.com` with the real domain. Install it in the Nginx configuration directory used by the target server, then test and reload Nginx:

```bash
nginx -t
systemctl reload nginx
```

The provided configuration:

- serves `frontend/dist`;
- routes `/api/` to `127.0.0.1:8000`;
- allows uploads up to 100 MB;
- allows up to 30 minutes for projection or predicate requests.

Add HTTPS using the target environment's certificate process before external exposure.

## 9. Optional frontend rebuild

The bundled `frontend/dist` is portable static output. If source changes are made on the target server, install a supported Node.js release and rebuild from the lock file:

```bash
cd /opt/dimbridge-app/frontend
npm ci
npm run build
```

Never copy `frontend/node_modules` from macOS to Linux.

## 10. Acceptance checks

Complete all checks through the public Nginx address:

1. `GET /api/health` returns `torch_cuda_available: true`.
2. Upload a CSV or XLSX file.
3. Generate PCA and UMAP projections.
4. Brush a proper subset of samples in Predicate Regression mode.
5. Confirm predicate results include factor rank, importance (F1 drop) and range reduction.
6. Confirm the Predicate View factor limit accepts 1–100 (default 20), excludes factors whose learned interval covers the full global range, and the SPLOM displays no more than the first 6 ranked factors.
7. Restart the backend and confirm that the UI reports missing uploaded data rather than silently using stale state.
8. Check browser memory while using the 6-factor SPLOM with the expected maximum dataset size.

Useful diagnostics:

```bash
systemctl status dimbridge-backend
journalctl -u dimbridge-backend -f
nvidia-smi
curl -fsS http://127.0.0.1:8000/api/health
```

## 11. Current production limitations

This package is suitable for a single-instance internal deployment. Before multi-user or high-availability deployment, implement:

- persistent/shared dataset storage;
- authentication and user-level data isolation;
- upload-size validation in the backend;
- a queue or concurrency limit for expensive GPU predicate jobs;
- cancellation of stale predicate requests;
- monitoring for backend memory, GPU memory and request duration.
