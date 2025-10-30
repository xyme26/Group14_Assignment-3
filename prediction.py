"""
Pydantic schemas for prediction-related API requests and responses.
Defines data validation models for flight delay predictions.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from utils.constants import AUSTRALIAN_AIRLINES, AUSTRALIAN_AIRPORTS, DELAY_RISK_CATEGORIES


class DelayRiskCategory(str, Enum):
    """Enumeration for delay risk categories."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class PredictionRequest(BaseModel):
    """
    Schema for flight delay prediction requests.
    Validates input data for making predictions.
    """
    route: str = Field(..., description="Flight route in format 'Origin-Destination'")
    departing_port: str = Field(..., description="Departure airport code")
    arriving_port: str = Field(..., description="Arrival airport code")
    airline: str = Field(..., description="Airline name")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    year: Optional[int] = Field(None, ge=2020, le=2030, description="Year (2020-2030)")
    sectors_scheduled: int = Field(..., ge=1, le=1000, description="Number of scheduled sectors")
    sectors_flown: Optional[int] = Field(None, ge=0, description="Number of sectors actually flown")
    cancellations: Optional[int] = Field(0, ge=0, description="Number of cancellations")
    departures_on_time: Optional[int] = Field(0, ge=0, description="Number of on-time departures")
    arrivals_on_time: Optional[int] = Field(0, ge=0, description="Number of on-time arrivals")
    departures_delayed: Optional[int] = Field(0, ge=0, description="Number of delayed departures")
    arrivals_delayed: Optional[int] = Field(0, ge=0, description="Number of delayed arrivals")
    
    @validator('route')
    def validate_route(cls, v):
        """Validate route format."""
        if not v or '-' not in v:
            raise ValueError('Route must be in format "Origin-Destination"')
        parts = v.split('-')
        if len(parts) != 2:
            raise ValueError('Route must contain exactly one dash separator')
        return v.strip()
    
    @validator('departing_port')
    def validate_departing_port(cls, v):
        """Validate departing port is a valid Australian airport."""
        if v not in AUSTRALIAN_AIRPORTS:
            raise ValueError(f'Invalid departing port: {v}')
        return v
    
    @validator('arriving_port')
    def validate_arriving_port(cls, v):
        """Validate arriving port is a valid Australian airport."""
        if v not in AUSTRALIAN_AIRPORTS:
            raise ValueError(f'Invalid arriving port: {v}')
        return v
    
    @validator('airline')
    def validate_airline(cls, v):
        """Validate airline is a valid Australian airline."""
        if v not in AUSTRALIAN_AIRLINES:
            raise ValueError(f'Invalid airline: {v}')
        return v
    
    @validator('year', pre=True, always=True)
    def set_default_year(cls, v):
        """Set default year to current year if not provided."""
        if v is None:
            return datetime.now().year
        return v
    
    @validator('sectors_flown', pre=True, always=True)
    def set_default_sectors_flown(cls, v, values):
        """Set default sectors_flown to sectors_scheduled if not provided."""
        if v is None and 'sectors_scheduled' in values:
            return values['sectors_scheduled']
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"


class PredictionResponse(BaseModel):
    """
    Schema for flight delay prediction responses.
    Contains prediction results and metadata.
    """
    success: bool = Field(..., description="Whether the prediction was successful")
    predictions: Dict[str, Any] = Field(..., description="Prediction results")
    input_data: Dict[str, Any] = Field(..., description="Processed input data used for prediction")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ClassificationPrediction(BaseModel):
    """
    Schema for classification prediction results.
    Contains delay risk category prediction.
    """
    predicted_category: DelayRiskCategory = Field(..., description="Predicted delay risk category")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each category")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class RegressionPrediction(BaseModel):
    """
    Schema for regression prediction results.
    Contains delay duration prediction.
    """
    predicted_delay_minutes: float = Field(..., ge=0.0, description="Predicted delay in minutes")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval for prediction")
    prediction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction accuracy score")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "predicted_delay_minutes": 15.5,
                "confidence_interval": {
                    "lower": 10.2,
                    "upper": 20.8
                },
                "prediction_accuracy": 0.85
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Schema for batch prediction requests.
    Allows multiple predictions in a single request.
    """
    predictions: List[PredictionRequest] = Field(..., min_items=1, max_items=100, 
                                               description="List of prediction requests (1-100)")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "route": "Sydney-Melbourne",
                        "departing_port": "Sydney",
                        "arriving_port": "Melbourne",
                        "airline": "Qantas",
                        "month": 6,
                        "sectors_scheduled": 10
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Schema for batch prediction responses.
    Contains results for multiple predictions.
    """
    success: bool = Field(..., description="Whether all predictions were successful")
    results: List[PredictionResponse] = Field(..., description="Individual prediction results")
    total_predictions: int = Field(..., description="Total number of predictions processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    processing_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch prediction timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictionError(BaseModel):
    """
    Schema for prediction error responses.
    Provides detailed error information.
    """
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "error": "Invalid input data",
                "error_code": "VALIDATION_ERROR",
                "details": {
                    "field": "route",
                    "message": "Route must be in format 'Origin-Destination'"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
