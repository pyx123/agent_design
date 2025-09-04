"""Database connection and session management."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from sqlalchemy import create_engine, pool, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import settings

# Base class for SQLAlchemy models
Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database_url
        self._engine = None
        self._async_engine = None
        self._sessionmaker = None
        self._async_sessionmaker = None
        self._initialized = False
    
    @property
    def engine(self):
        """Get or create sync engine."""
        if self._engine is None:
            # Convert async SQLite URL to sync
            url = self.database_url.replace("sqlite+aiosqlite://", "sqlite://")
            self._engine = create_engine(
                url,
                poolclass=pool.StaticPool if "sqlite" in url else None,
                connect_args={"check_same_thread": False} if "sqlite" in url else {},
                echo=settings.database.echo,
            )
        return self._engine
    
    @property
    def async_engine(self):
        """Get or create async engine."""
        if self._async_engine is None:
            # Ensure URL is async for SQLite
            url = self.database_url
            if "sqlite://" in url and "aiosqlite" not in url:
                url = url.replace("sqlite://", "sqlite+aiosqlite://")
            
            self._async_engine = create_async_engine(
                url,
                echo=settings.database.echo,
                pool_pre_ping=settings.database.pool_pre_ping,
            )
        return self._async_engine
    
    @property
    def session_factory(self):
        """Get sync session factory."""
        if self._sessionmaker is None:
            self._sessionmaker = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
            )
        return self._sessionmaker
    
    @property
    def async_session_factory(self):
        """Get async session factory."""
        if self._async_sessionmaker is None:
            self._async_sessionmaker = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._async_sessionmaker
    
    async def initialize(self):
        """Initialize database (create tables from DDL)."""
        if self._initialized:
            return
        
        # Read and execute DDL
        ddl_path = Path(__file__).parent.parent.parent / "config" / "sql" / "ddl.sql"
        if not ddl_path.exists():
            raise FileNotFoundError(f"DDL file not found: {ddl_path}")
        
        with open(ddl_path, "r") as f:
            ddl_script = f.read()
        
        # Execute DDL using aiosqlite directly for SQLite
        if "sqlite" in self.database_url:
            db_path = self.database_url.split("///")[-1]
            async with aiosqlite.connect(db_path) as db:
                await db.executescript(ddl_script)
                await db.commit()
        else:
            # For other databases, use async engine
            async with self.async_engine.begin() as conn:
                await conn.execute(ddl_script)
        
        self._initialized = True
    
    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Get an async database session."""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_sync_session(self):
        """Get a sync database session (for LangGraph checkpoint)."""
        return self.session_factory()
    
    async def close(self):
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._engine:
            self._engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


# Helper functions
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Dependency for FastAPI to get database session."""
    async with db_manager.get_session() as session:
        yield session


async def init_database():
    """Initialize the database."""
    await db_manager.initialize()


async def close_database():
    """Close database connections."""
    await db_manager.close()