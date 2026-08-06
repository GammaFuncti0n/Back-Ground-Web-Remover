"""Web UI: upload photo, call pipeline, show/download result."""

import base64
import os
from io import BytesIO
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="BG Removal Web")

PIPELINE_URL = os.getenv("PIPELINE_URL", "http://127.0.0.1:8001").rstrip("/")
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"result_b64": None, "error": None},
    )


@app.post("/", response_class=HTMLResponse)
async def process(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"result_b64": None, "error": "Загрузите файл изображения"},
            status_code=400,
        )

    raw = await file.read()
    if not raw:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"result_b64": None, "error": "Пустой файл"},
            status_code=400,
        )

    filename = file.filename or "image.png"
    content_type = file.content_type or "application/octet-stream"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{PIPELINE_URL}/process",
                files={"file": (filename, BytesIO(raw), content_type)},
            )
    except httpx.RequestError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"result_b64": None, "error": f"Pipeline недоступен: {exc}"},
            status_code=502,
        )

    if response.status_code != 200:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result_b64": None,
                "error": f"Ошибка pipeline ({response.status_code}): {response.text}",
            },
            status_code=502,
        )

    result_b64 = base64.b64encode(response.content).decode("ascii")
    return templates.TemplateResponse(
        request,
        "index.html",
        {"result_b64": result_b64, "error": None},
    )
