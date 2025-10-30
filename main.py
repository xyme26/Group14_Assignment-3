"""
FastAPI Backend Application for Flight Delay Prediction System
Assignment 3 - Full-Stack Web Development for AI Application

This is the main entry point for the FastAPI backend that integrates
the machine learning models from Assignment 2 to provide real-time
flight delay predictions and analytics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from routers import predictions, analytics, airlines
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.analytics_service import AnalyticsService
from utils.constants import API_VERSION, APP_NAME, APP_DESCRIPTION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
data_service = None
prediction_service = None
analytics_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    Loads ML models and data on startup, cleans up on shutdown.
    """
    global data_service, prediction_service, analytics_service
    
    # Startup
    logger.info("Starting up Flight Delay Prediction API...")
    try:
        # Initialize data service
        data_service = DataService()
        await data_service.initialize()
        logger.info("Data service initialized successfully")
        
        # Initialize prediction service
        prediction_service = PredictionService(data_service)
        logger.info("Prediction service initialized successfully")
        
        # Initialize analytics service
        analytics_service = AnalyticsService(data_service)
        logger.info("Analytics service initialized successfully")
        
        # Set services in routers
        predictions.set_services(prediction_service, data_service)
        analytics.set_services(analytics_service, data_service)
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Flight Delay Prediction API...")
    if data_service:
        await data_service.cleanup()

# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    predictions.router,
    prefix="/api/v1/predictions",
    tags=["Predictions"]
)

app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["Analytics"]
)

app.include_router(
    airlines.router,
    prefix="/api/v1/airlines",
    tags=["Airlines & Reference Data"]
)

@app.get("/")
async def root():
    """
    Root endpoint providing API information and health status.
    """
    return {
        "message": "Flight Delay Prediction API",
        "version": API_VERSION,
        "status": "healthy",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    """
    global data_service
    
    if data_service and data_service.is_ready():
        return {
            "status": "healthy",
            "models_loaded": True,
            "data_ready": True
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "models_loaded": False,
                "data_ready": False
            }
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Global HTTP exception handler for consistent error responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors.
    """
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    """
    Run the FastAPI application directly for development.
    """
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
