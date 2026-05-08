"""FastAPI app: serves the frontend and exposes a /predict inference endpoint."""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import Predictor

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    model_path = Path(os.environ.get("MODEL_PATH", "model/best_model.pt"))
    _predictor = Predictor(model_path)
    yield


app = FastAPI(title="GI Endoscopy Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/samples")
def list_samples() -> dict[str, list[str]]:
    """Return {class_name: [url, ...]} for all bundled sample images."""
    samples_dir = Path("static/samples")
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

    predictions = _predictor.predict(image)
    return JSONResponse({"predictions": predictions})
