from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


@dataclass
class DatasetRecord:
    dataset_id: str
    filename: str
    dataframe: pd.DataFrame


class DatasetStore:
    def __init__(self) -> None:
        self._datasets: dict[str, DatasetRecord] = {}

    def create_from_upload(self, filename: str, content: bytes) -> DatasetRecord:
        dataframe = self._read_dataframe(filename=filename, content=content)
        dataset_id = uuid4().hex
        record = DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            dataframe=dataframe,
        )
        self._datasets[dataset_id] = record
        return record

    def get(self, dataset_id: str) -> DatasetRecord | None:
        return self._datasets.get(dataset_id)

    def _read_dataframe(self, filename: str, content: bytes) -> pd.DataFrame:
        suffix = Path(filename).suffix.lower()
        file_like = BytesIO(content)

        if suffix == ".csv":
            dataframe = pd.read_csv(file_like)
        elif suffix in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(file_like)
        else:
            raise ValueError(f"Unsupported file type: {suffix or '<unknown>'}")

        if dataframe.empty:
            raise ValueError("Uploaded dataset is empty.")
        if len(dataframe.columns) == 0:
            raise ValueError("Uploaded dataset has no columns.")
        return dataframe


dataset_store = DatasetStore()


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def preview_rows(dataframe: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    return dataframe.head(limit).where(pd.notnull(dataframe), None).to_dict(orient="records")


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include="number").columns.tolist()


def all_rows(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    return dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")


def renderable_numeric_frame(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_columns = columns or numeric_columns(dataframe)
    if not selected_columns:
        return pd.DataFrame()

    numeric_frame = dataframe[selected_columns].apply(pd.to_numeric, errors="coerce")
    valid_rows = np.isfinite(numeric_frame.to_numpy(dtype=float)).all(axis=1)
    return numeric_frame.loc[valid_rows].reset_index(drop=True)
