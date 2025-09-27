from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel
from api.v1.routes import router as api_v1_router
from core.config import settings
from core.db import async_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    if settings.ENV == "development":
        # async with async_engine.begin() as conn:
        #     await conn.run_sync(SQLModel.metadata.create_all)
        pass
    yield

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

@app.get("/health")
async def health():
    return {"status": "ok"}

