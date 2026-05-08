"""FastAPI app: serves the frontend and exposes a /predict inference endpoint."""

import io
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

import logging_config
from inference import Predictor

HERE = Path(__file__).parent
_predictor: Predictor | None = None
log = logging.getLogger("app")

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Sliding-window counter per client IP. Limits the expensive /predict endpoint
# to prevent cost abuse on Cloud Run's per-request billing.
_RATE_LIMIT = 20          # max requests per window
_RATE_WINDOW = 60.0       # window size in seconds
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

_rate_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW
    timestamps = [t for t in _rate_store[ip] if t > cutoff]
    if len(timestamps) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {_RATE_LIMIT} requests per minute.",
            headers={"Retry-After": str(int(_RATE_WINDOW))},
        )
    timestamps.append(now)
    _rate_store[ip] = timestamps


# ── Middleware ────────────────────────────────────────────────────────────────

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


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging_config.configure()
    global _predictor
    model_path = Path(os.environ.get("MODEL_PATH", str(HERE / "model" / "best_model.pt")))
    _predictor = Predictor(model_path)
    log.info("model loaded", extra={"model_path": str(model_path)})
    yield


# ── App ───────────────────────────────────────────────────────────────────────

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
async def predict(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    _check_rate_limit(request.client.host)

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum upload size is 10 MB.")

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
