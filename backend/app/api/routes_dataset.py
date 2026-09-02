from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import DatasetSummary
from app.services.dataset_service import (
    all_rows,
    dataset_store,
    numeric_columns,
    preview_rows,
    renderable_numeric_frame,
)


router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def _to_summary(record) -> DatasetSummary:
    dataframe = record.dataframe
    return DatasetSummary(
        dataset_id=record.dataset_id,
        filename=record.filename,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=dataframe.columns.tolist(),
        numeric_columns=numeric_columns(dataframe),
        renderable_row_count=len(renderable_numeric_frame(dataframe)),
        preview=preview_rows(dataframe),
        records=all_rows(dataframe),
    )


@router.post("/upload", response_model=DatasetSummary)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetSummary:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        record = dataset_store.create_from_upload(file.filename or "dataset", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _to_summary(record)


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset(dataset_id: str) -> DatasetSummary:
    record = dataset_store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return _to_summary(record)
