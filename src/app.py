"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.api.routes import router as api_router
from src.api.websocket import websocket_endpoint
from src.services import init_database, close_database
from src.tools import init_mcp_clients, close_mcp_clients, init_tool_registry

# Configure logging
logging.basicConfig(
    level=settings.observability.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if settings.observability.log_format == "text" else None
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting DevOps Agent API...")
    
    # Initialize database
    await init_database()
    logger.info("Database initialized")
    
    # Initialize MCP clients
    try:
        await init_mcp_clients()
        await init_tool_registry()
        logger.info("MCP clients initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize MCP clients: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down DevOps Agent API...")
    await close_database()
    await close_mcp_clients()


# Create FastAPI app
app = FastAPI(
    title="DevOps Agent API",
    description="AI-powered troubleshooting system using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
)

# Add CORS middleware
if settings.security.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.auth_enabled,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API routes
app.include_router(api_router)

# Add WebSocket endpoint
app.add_websocket_route("/ws", websocket_endpoint)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DevOps Agent API",
        "version": "0.1.0",
        "status": "running",
        "environment": settings.environment,
        "docs": "/docs" if settings.is_development else None,
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.is_development else None,
        }
    )


# CLI entry point
def main():
    """Run the application."""
    import uvicorn
    
    uvicorn.run(
        "src.app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload and settings.is_development,
        workers=settings.server.workers if not settings.server.reload else 1,
        log_level=settings.observability.log_level.lower(),
    )


if __name__ == "__main__":
    main()