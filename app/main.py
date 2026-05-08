"""FastAPI app: serves the frontend and exposes a /predict inference endpoint."""

import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import logging_config
from inference import Predictor

HERE = Path(__file__).parent
_predictor: Predictor | None = None
log = logging.getLogger("app")


class _RequestLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging_config.configure()
    global _predictor
    model_path = Path(os.environ.get("MODEL_PATH", str(HERE / "model" / "best_model.pt")))
    _predictor = Predictor(model_path)
    log.info("model loaded", extra={"model_path": str(model_path)})
    yield


app = FastAPI(title="GI Endoscopy Classifier", lifespan=lifespan)
app.add_middleware(_RequestLogger)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(HERE / "static" / "index.html"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/samples")
def list_samples() -> dict[str, list[str]]:
    """Return {class_name: [url, ...]} for all bundled sample images."""
    samples_dir = HERE / "static" / "samples"
    if not samples_dir.exists():
        return {}
    return {
        cls.name: sorted(
            f"/static/samples/{cls.name}/{f.name}"
            for f in cls.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for cls in sorted(samples_dir.iterdir())
        if cls.is_dir()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    start = time.perf_counter()
    result = _predictor.predict(image)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    top = result["predictions"][0]
    log.info(
        "prediction",
        extra={
            "predicted_class": top["class"],
            "confidence": round(top["probability"], 4),
            "duration_ms": duration_ms,
        },
    )
    return JSONResponse(result)
