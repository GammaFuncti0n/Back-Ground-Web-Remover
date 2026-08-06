"""Pipeline: accept image, call bg_remover, return result."""

import os
from io import BytesIO

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

app = FastAPI(title="BG Removal Pipeline")

BG_REMOVER_URL = os.getenv("BG_REMOVER_URL", "http://127.0.0.1:8002").rstrip("/")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process(file: UploadFile = File(...)) -> Response:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "image.png"
    content_type = file.content_type or "application/octet-stream"

    # Placeholder slot for future pipeline steps (resize, normalize, etc.)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BG_REMOVER_URL}/remove",
                files={"file": (filename, BytesIO(raw), content_type)},
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"bg_remover unreachable: {exc}",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"bg_remover error ({response.status_code}): {response.text}",
        )

    return Response(
        content=response.content,
        media_type=response.headers.get("content-type", "image/png"),
        headers={"Content-Disposition": 'inline; filename="result.png"'},
    )
