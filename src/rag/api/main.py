"""
Main FastAPI application.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .routes.rag_routes import router as rag_router
from .dependencies import check_services_health
from .schemas.requests import HealthResponse
from ..config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    settings.ensure_directories()
    print(f"🚀 RAG API starting up...")
    print(f"📊 Data directory: {settings.data_dir}")
    print(f"🔍 ChromaDB directory: {settings.chromadb_dir}")
    
    yield
    
    # Shutdown
    print("🛑 RAG API shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Document RAG API",
        description="Modern Document RAG System with Mistral OCR Pipeline",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(rag_router)
    
    return app


# Create app instance
app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        services = await check_services_health()
        
        # Determine overall status
        all_healthy = all(services.values())
        status = "healthy" if all_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            version="2.0.0",
            services=services
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Document RAG API",
        "version": "2.0.0",
        "description": "Modern Document RAG System with Mistral OCR Pipeline",
        "endpoints": {
            "health": "/health",
            "search": "/api/v1/search",
            "ask": "/api/v1/ask",
            "stats": "/api/v1/stats"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.rag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
