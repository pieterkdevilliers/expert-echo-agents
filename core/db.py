import os
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from core.config import settings

# Decide DB URL based on environment
if settings.ENV == "development":
    # Local SQLite async
    DB_FILE = "dev.db"
    DATABASE_URL = f"sqlite+aiosqlite:///{DB_FILE}"
else:
    # Production PostgreSQL async
    DATABASE_URL = settings.DATABASE_URL.replace(
        "postgres://", "postgresql+asyncpg://"
    )

# Create async engine
async_engine = create_async_engine(DATABASE_URL, echo=True)

# Async session factory
async_session = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Dependency for FastAPI routes
async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
