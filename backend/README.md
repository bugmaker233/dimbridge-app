# Backend

This backend exposes the standalone DimBridge APIs used by the React frontend.

## Endpoints

- `GET /api/health`: service and runtime status
- `POST /api/dataset/upload`: upload a CSV/XLSX dataset
- `GET /api/dataset/{dataset_id}`: fetch uploaded dataset metadata
- `POST /api/projection/run`: compute PCA, t-SNE, UMAP, or supervised UMAP projection
- `POST /api/projection/umap`: compatibility endpoint for UMAP projection
- `POST /api/predicate/data-extent`: compute interval extents from selection masks
- `POST /api/predicate/regression`: preserve the legacy regression implementation
- `POST /api/predicate/paper-regression`: optimize DimBridge equations 5-10
- `POST /api/predicate/rpi`: run multi-solution recursive predicate induction

## Runtime Notes

- Both regression engines use PyTorch. The paper endpoint adds the explicit L1 feature-selection
  term and both squared-L2 sequence smoothness terms described in the paper.
- RPI uses quantile bins for numeric base predicates and F1 for recursive refinement. The PIXAL
  and DimBridge papers do not publish an official RPI implementation or prescribe a numeric
  binning method, so this module is a documented reconstruction rather than a line-for-line port.
- RPI returns its best candidate in `predicates` for compatibility and all retained alternatives
  in `candidate_solutions`. Search limits and truncation status are returned in `diagnostics`.
- Projection uses scikit-learn for PCA/t-SNE and umap-learn for UMAP.
- CSV, XLSX, and XLS uploads are supported. XLSX uses `openpyxl`; XLS uses `xlrd`.
- Dataset uploads are stored in memory for the current backend process.
- Predicate endpoints normalize numeric columns and drop rows with invalid numeric values so
  mask validation stays consistent with the frontend widget rendering path.

## Install

If you use the existing `visulization` conda environment, install the missing backend packages
inside that environment:

```bash
source /home/descfly/anaconda3/bin/activate visulization
cd /data/chenruisi/CodeReading/dimbridge-jupyter/dimbridge-app/backend
python -m pip install -r requirements.txt
```

## Run

```bash
source /home/descfly/anaconda3/bin/activate visulization
cd /data/chenruisi/CodeReading/dimbridge-jupyter/dimbridge-app/backend
uvicorn app.main:app --reload --port 8000
```

## Quick Checks

```bash
curl http://127.0.0.1:8000/api/health
```

Open the API docs after startup:

```text
http://127.0.0.1:8000/docs
```
