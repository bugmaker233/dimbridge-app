from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_dataset import router as dataset_router
from app.api.routes_health import router as health_router
from app.api.routes_predicate import router as predicate_router
from app.api.routes_projection import router as projection_router


app = FastAPI(
    title="DimBridge Backend",
    version="0.1.0",
    description="Standalone backend service for the DimBridge analysis application.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(dataset_router)
app.include_router(predicate_router)
app.include_router(projection_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DimBridge backend is running."}
