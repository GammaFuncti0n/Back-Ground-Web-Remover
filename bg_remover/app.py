"""BG remover service."""

import time
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
import yaml
from PIL import Image

from .placeholder import Placeholder
from .deeplab import DeepLab
from .birefnet import BiRefNet
from .isnet import ISNet

app = FastAPI(title="Back Ground Remove")

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

processor_name = config['processor']
if processor_name == 'white_border':
    processor = Placeholder()
elif processor_name in ('deeplabv3_mobilenet', 'deeplabv3plus_resnet101'):
    processor = DeepLab(name=processor_name)
elif processor_name == 'birefnet':
    processor = BiRefNet()
elif processor_name == 'isnet':
    processor = ISNet()
else:
    raise ValueError(f"Unknown processor: {processor_name}")


def _sync_device() -> None:
    """Flush async GPU/MPS work so timing reflects real inference."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@app.on_event("startup")
def preload_model() -> None:
    """Warm up weights/model so the first request is not painfully slow."""
    if hasattr(processor, "_init_model"):
        processor._init_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/remove")
async def remove_background_endpoint(file: UploadFile = File(...)) -> Response:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        with Image.open(BytesIO(raw)) as img:
            _sync_device()
            t0 = time.perf_counter()
            processed = processor.remove(img)
            _sync_device()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Processing failed: {exc}") from exc

    print(f"[inference] {processor_name}: {elapsed_ms:.1f} ms")

    buffer = BytesIO()
    processed.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="result.png"',
            "X-Inference-Time-Ms": f"{elapsed_ms:.1f}",
        },
    )
