"""
Prediction API routes for flight delay predictions.
Handles prediction requests and responses for ML model integration.
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import time

from schemas.prediction import (
    PredictionRequest, PredictionResponse, ClassificationPrediction,
    RegressionPrediction, BatchPredictionRequest, BatchPredictionResponse,
    PredictionError
)
from services.prediction_service import PredictionService
from services.data_service import DataService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global services (will be injected)
prediction_service: PredictionService = None
data_service: DataService = None


def get_prediction_service() -> PredictionService:
    """Get prediction service instance."""
    global prediction_service, data_service
    if prediction_service is None:
        if data_service is None:
            raise HTTPException(status_code=503, detail="Data service not available")
        prediction_service = PredictionService(data_service)
    return prediction_service


def set_services(pred_service: PredictionService, data_serv: DataService):
    """Set service instances (called from main.py)."""
    global prediction_service, data_service
    prediction_service = pred_service
    data_service = data_serv


@router.post("/delay-risk", response_model=PredictionResponse)
async def predict_delay_risk(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """
    Predict delay risk category for a flight.
    
    This endpoint uses the trained classification model to predict whether
    a flight will have Low, Medium, or High delay risk based on the input parameters.
    
    Args:
        request (PredictionRequest): Flight details for prediction
        background_tasks (BackgroundTasks): FastAPI background tasks
        
    Returns:
        PredictionResponse: Prediction results with confidence scores
        
    Raises:
        HTTPException: If prediction fails or service is unavailable
    """
    start_time = time.time()
    
    try:
        # Get prediction service
        pred_service = get_prediction_service()
        
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        prediction_result = pred_service.predict_delay_risk(input_data)
        
        # Create classification prediction
        classification_pred = ClassificationPrediction(
            predicted_category=prediction_result["predicted_category"],
            confidence_scores=prediction_result["confidence_scores"],
            probability=prediction_result["probability"]
        )
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            success=True,
            predictions={
                "classification": classification_pred.dict(),
                "model_info": prediction_result["model_info"]
            },
            input_data=input_data,
            model_info=prediction_result["model_info"],
            processing_time_ms=total_processing_time
        )
        
        # Log prediction
        logger.info(f"Delay risk prediction completed: {prediction_result['predicted_category']}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in delay risk prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in delay risk prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@router.post("/delay-duration", response_model=PredictionResponse)
async def predict_delay_duration(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """
    Predict delay duration in minutes for a flight.
    
    This endpoint uses the trained regression model to predict the expected
    delay duration in minutes for a flight based on the input parameters.
    
    Args:
        request (PredictionRequest): Flight details for prediction
        background_tasks (BackgroundTasks): FastAPI background tasks
        
    Returns:
        PredictionResponse: Prediction results with delay duration
        
    Raises:
        HTTPException: If prediction fails or service is unavailable
    """
    start_time = time.time()
    
    try:
        # Get prediction service
        pred_service = get_prediction_service()
        
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        prediction_result = pred_service.predict_delay_duration(input_data)
        
        # Create regression prediction
        regression_pred = RegressionPrediction(
            predicted_delay_minutes=prediction_result["predicted_delay_minutes"],
            confidence_interval=prediction_result.get("confidence_interval"),
            prediction_accuracy=prediction_result.get("prediction_accuracy")
        )
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            success=True,
            predictions={
                "regression": regression_pred.dict(),
                "model_info": prediction_result["model_info"]
            },
            input_data=input_data,
            model_info=prediction_result["model_info"],
            processing_time_ms=total_processing_time
        )
        
        # Log prediction
        logger.info(f"Delay duration prediction completed: {prediction_result['predicted_delay_minutes']} minutes")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in delay duration prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in delay duration prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@router.post("/both", response_model=PredictionResponse)
async def predict_both(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """
    Make both delay risk and delay duration predictions.
    
    This endpoint combines both classification and regression predictions
    to provide comprehensive delay analysis for a flight.
    
    Args:
        request (PredictionRequest): Flight details for prediction
        background_tasks (BackgroundTasks): FastAPI background tasks
        
    Returns:
        PredictionResponse: Combined prediction results
        
    Raises:
        HTTPException: If prediction fails or service is unavailable
    """
    start_time = time.time()
    
    try:
        # Get prediction service
        pred_service = get_prediction_service()
        
        # Convert request to dict
        input_data = request.dict()
        
        # Make combined prediction
        prediction_result = pred_service.predict_both(input_data)
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            success=True,
            predictions=prediction_result,
            input_data=input_data,
            model_info={
                "classification_model": prediction_result["classification"]["model_info"],
                "regression_model": prediction_result["regression"]["model_info"]
            },
            processing_time_ms=total_processing_time
        )
        
        # Log prediction
        logger.info(f"Combined prediction completed: {prediction_result['classification']['predicted_category']}, {prediction_result['regression']['predicted_delay_minutes']} min")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in combined prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in combined prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
) -> BatchPredictionResponse:
    """
    Make predictions for multiple flights in batch.
    
    This endpoint allows processing multiple prediction requests in a single call,
    which is more efficient for bulk operations.
    
    Args:
        request (BatchPredictionRequest): List of flight details for prediction
        background_tasks (BackgroundTasks): FastAPI background tasks
        
    Returns:
        BatchPredictionResponse: Batch prediction results
        
    Raises:
        HTTPException: If prediction fails or service is unavailable
    """
    start_time = time.time()
    
    try:
        # Get prediction service
        pred_service = get_prediction_service()
        
        # Convert requests to list of dicts
        input_data_list = [req.dict() for req in request.predictions]
        
        # Make batch prediction
        batch_result = pred_service.predict_batch(input_data_list)
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000
        
        # Create individual prediction responses
        results = []
        for i, result in enumerate(batch_result["results"]):
            if result.get("success", False):
                pred_response = PredictionResponse(
                    success=True,
                    predictions=result,
                    input_data=input_data_list[i],
                    model_info=result.get("model_info", {}),
                    processing_time_ms=result.get("processing_time_ms", 0)
                )
                results.append(pred_response)
            else:
                error_response = PredictionResponse(
                    success=False,
                    predictions={"error": result.get("error", "Unknown error")},
                    input_data=input_data_list[i],
                    model_info={},
                    processing_time_ms=0
                )
                results.append(error_response)
        
        # Create batch response
        response = BatchPredictionResponse(
            success=batch_result["success"],
            results=results,
            total_predictions=batch_result["total_predictions"],
            successful_predictions=batch_result["successful_predictions"],
            failed_predictions=batch_result["failed_predictions"],
            processing_time_ms=total_processing_time
        )
        
        # Log batch prediction
        logger.info(f"Batch prediction completed: {batch_result['successful_predictions']}/{batch_result['total_predictions']} successful")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")


@router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded ML models.
    
    Returns:
        Dict[str, Any]: Model information including available models, features, and metadata
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        # Get prediction service
        pred_service = get_prediction_service()
        
        # Get model information
        model_info = pred_service.get_model_info()
        
        return {
            "success": True,
            "model_info": model_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting model info")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for prediction service.
    
    Returns:
        Dict[str, Any]: Health status of the prediction service
    """
    try:
        # Check if services are available
        pred_service = get_prediction_service()
        
        return {
            "status": "healthy",
            "service": "predictions",
            "models_loaded": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Prediction service health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "predictions",
            "models_loaded": False,
            "error": str(e),
            "timestamp": time.time()
        }
