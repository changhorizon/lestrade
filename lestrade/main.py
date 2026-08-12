import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ingestion.watcher import watch_loop
from .rag.engine import engine
from .ratelimit import RateLimitMiddleware
from .router import chat, ingest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FAISS index loaded: %d vectors", engine.index.ntotal if engine.index else 0)
    task = asyncio.create_task(watch_loop())
    yield
    task.cancel()


app = FastAPI(title="Lestrade", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)

app.include_router(chat.router)
app.include_router(ingest.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
