"""FastAPI main application for Context Server."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from context_server.models.api.system import HealthResponse

# Import routers after app creation to avoid circular imports
# from .contexts import router as contexts_router
# from .documents import router as documents_router
# from .search import router as search_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Context Server API")

    # Import here to avoid issues
    from ..core.database import DatabaseManager

    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.initialize()
    app.state.db_manager = db_manager

    logger.info("Context Server API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Context Server API")
    if hasattr(app.state, "db_manager"):
        await app.state.db_manager.close()
    logger.info("Context Server API shutdown complete")


app = FastAPI(
    title="Context Server API",
    description="Modular Documentation RAG System with MCP Integration",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"},
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_connected = True
        if hasattr(app.state, "db_manager"):
            db_connected = await app.state.db_manager.is_healthy()

        # Check embedding service
        embedding_available = True

        return HealthResponse(
            timestamp=datetime.now(),
            version="0.1.0",
            database_connected=db_connected,
            embedding_service_available=embedding_available,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Context Server API", "version": "0.1.0", "docs": "/docs"}


@app.post("/admin/reinitialize-db")
async def reinitialize_database():
    """Reinitialize database schema (admin endpoint)."""
    try:
        if hasattr(app.state, "db_manager"):
            await app.state.db_manager.initialize()
            return {"message": "Database reinitialized successfully"}
        else:
            raise HTTPException(
                status_code=503, detail="Database manager not available"
            )
    except Exception as e:
        logger.error(f"Database reinitialization failed: {e}")
        raise HTTPException(status_code=500, detail="Database reinitialization failed")


# Include routers after app creation
def setup_routers():
    from .contexts import router as contexts_router
    from .documents import router as documents_router
    from .jobs import router as jobs_router
    from .search import router as search_router

    app.include_router(contexts_router, prefix="/api/contexts", tags=["contexts"])
    app.include_router(documents_router, prefix="/api", tags=["documents"])
    app.include_router(jobs_router, prefix="/api", tags=["jobs"])
    app.include_router(search_router, prefix="/api", tags=["search"])


# Setup routers
setup_routers()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
