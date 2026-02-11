"""
FastAPI main application
KARGA Query API for financial data
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import config

from app.api.routes import query, markets, database, signals  # report module disabled (requires anthropic package)

# Initialize Sentry for error tracking (production only)
if config.SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        integrations=[
            FastApiIntegration(),
            StarletteIntegration(),
        ],
        # Set traces_sample_rate to 1.0 to capture 100% of transactions
        # Lower in production (0.1 = 10%)
        traces_sample_rate=0.1,
        # Capture 100% of errors
        profiles_sample_rate=1.0,
        environment=config.ENVIRONMENT,
        release=f"karga-backend@2.0.0",
    )
    print(f"[SENTRY] Initialized for environment: {config.ENVIRONMENT}")
else:
    print("[SENTRY] Disabled (no DSN configured)")

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="KARGA API",
    description="Natural language to graph query API for financial data",
    version="2.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Limit to needed methods
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests
)

# Include routers
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(markets.router, prefix="/api/markets", tags=["Markets"])
app.include_router(database.router, prefix="/api/database", tags=["Database"])
app.include_router(signals.router, prefix="/api")
# app.include_router(report.router, prefix="/api/report", tags=["Reports"])  # Disabled - requires anthropic package


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "KARGA API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request):
    """Health check endpoint with database connectivity test"""
    from app.database.connection import get_db

    health_status = {
        "status": "healthy",
        "version": "2.0.0",
        "services": {}
    }

    # Check database connection
    try:
        db = get_db()
        db.collection("Company").count()
        health_status["services"]["database"] = "connected"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["services"]["database"] = f"error: {str(e)[:100]}"

    return health_status


@app.get("/debug/version")
def version_check():
    """Debug endpoint to check which version of code is running"""
    return {
        "version": "2.0-OPTIMIZED",
        "markets_query": "simplified-no-collect",
        "timestamp": "2026-01-15",
        "message": "If you see this, the NEW optimized code is running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.FASTAPI_HOST,
        port=config.FASTAPI_PORT,
        reload=True
    )
