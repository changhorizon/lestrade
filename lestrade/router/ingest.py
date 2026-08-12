from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..rag.engine import engine

router = APIRouter()


@router.post("/api/ingest")
async def ingest_text(text: str = Form(...), source: str = Form(None)):
    try:
        chunk_count = await engine.add_text(text, source=source)
        return {"status": "ok", "chunks": chunk_count, "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@router.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        chunk_count = await engine.add_text(text, source=file.filename)
        return {"status": "ok", "chunks": chunk_count, "source": file.filename}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Only UTF-8 text files are supported")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@router.get("/api/ingest/stats")
async def ingest_stats():
    return {
        "total_vectors": engine.index.ntotal if engine.index else 0,
        "index_path": str(engine.index),
    }
