"""
FastAPI main application
GraphRAG API for financial data querying
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import config

from app.api.routes import query, markets, database

# Create FastAPI app
app = FastAPI(
    title="GraphRAG API",
    description="AI-powered graph query API for financial data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(markets.router, prefix="/api/markets", tags=["Markets"])
app.include_router(database.router, prefix="/api/database", tags=["Database"])


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "GraphRAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.FASTAPI_HOST,
        port=config.FASTAPI_PORT,
        reload=True
    )
