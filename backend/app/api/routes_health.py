import platform

import torch
from fastapi import APIRouter

from app.schemas import HealthResponse


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="dimbridge-backend",
        version="0.1.0",
        python_version=platform.python_version(),
        torch_available=True,
        torch_cuda_available=torch.cuda.is_available(),
        torch_version=torch.__version__,
    )
