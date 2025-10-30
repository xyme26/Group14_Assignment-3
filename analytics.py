"""
Pydantic schemas for analytics-related API requests and responses.
Defines data validation models for dashboard analytics and data visualization.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from enum import Enum

from utils.constants import AUSTRALIAN_AIRLINES, AUSTRALIAN_AIRPORTS, DELAY_RISK_CATEGORIES


class ChartType(str, Enum):
    """Enumeration for supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"


class AnalyticsFilter(BaseModel):
    """
    Schema for analytics filter parameters.
    Allows filtering of data for analytics queries.
    """
    start_date: Optional[date] = Field(None, description="Start date for data filtering (YYYY-MM-DD)")
    end_date: Optional[date] = Field(None, description="End date for data filtering (YYYY-MM-DD)")
    airline: Optional[str] = Field(None, description="Filter by specific airline")
    airport: Optional[str] = Field(None, description="Filter by specific airport")
    delay_risk_category: Optional[str] = Field(None, description="Filter by delay risk category")
    route: Optional[str] = Field(None, description="Filter by specific route")
    limit: Optional[int] = Field(1000, ge=1, le=10000, description="Maximum number of records to return")
    
    @validator('airline')
    def validate_airline(cls, v):
        """Validate airline filter."""
        if v is not None and v not in AUSTRALIAN_AIRLINES:
            raise ValueError(f'Invalid airline filter: {v}')
        return v
    
    @validator('airport')
    def validate_airport(cls, v):
        """Validate airport filter."""
        if v is not None and v not in AUSTRALIAN_AIRPORTS:
            raise ValueError(f'Invalid airport filter: {v}')
        return v
    
    @validator('delay_risk_category')
    def validate_delay_risk_category(cls, v):
        """Validate delay risk category filter."""
        if v is not None and v not in DELAY_RISK_CATEGORIES:
            raise ValueError(f'Invalid delay risk category: {v}')
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validate that end_date is after start_date."""
        if v and 'start_date' in values and values['start_date']:
            if v < values['start_date']:
                raise ValueError('End date must be after start date')
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ChartData(BaseModel):
    """
    Schema for chart data responses.
    Contains data points for visualization.
    """
    chart_type: ChartType = Field(..., description="Type of chart")
    title: str = Field(..., description="Chart title")
    x_axis_label: str = Field(..., description="X-axis label")
    y_axis_label: str = Field(..., description="Y-axis label")
    data: List[Dict[str, Any]] = Field(..., description="Chart data points")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional chart metadata")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class AnalyticsResponse(BaseModel):
    """
    Schema for analytics API responses.
    Contains analytics data and chart information.
    """
    success: bool = Field(..., description="Whether the analytics query was successful")
    data: Dict[str, Any] = Field(..., description="Analytics data")
    charts: List[ChartData] = Field(..., description="Chart data for visualization")
    filters_applied: AnalyticsFilter = Field(..., description="Filters applied to the query")
    total_records: int = Field(..., description="Total number of records in the dataset")
    filtered_records: int = Field(..., description="Number of records after filtering")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DelayAnalytics(BaseModel):
    """
    Schema for delay analytics data.
    Contains aggregated delay statistics.
    """
    total_flights: int = Field(..., description="Total number of flights")
    delayed_flights: int = Field(..., description="Number of delayed flights")
    on_time_flights: int = Field(..., description="Number of on-time flights")
    cancellation_rate: float = Field(..., ge=0.0, le=1.0, description="Cancellation rate")
    average_delay_minutes: float = Field(..., ge=0.0, description="Average delay in minutes")
    median_delay_minutes: float = Field(..., ge=0.0, description="Median delay in minutes")
    delay_risk_distribution: Dict[str, int] = Field(..., description="Distribution of delay risk categories")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "total_flights": 1000,
                "delayed_flights": 250,
                "on_time_flights": 750,
                "cancellation_rate": 0.05,
                "average_delay_minutes": 15.5,
                "median_delay_minutes": 8.0,
                "delay_risk_distribution": {
                    "Low": 600,
                    "Medium": 300,
                    "High": 100
                }
            }
        }


class AirlinePerformance(BaseModel):
    """
    Schema for airline performance analytics.
    Contains performance metrics for airlines.
    """
    airline: str = Field(..., description="Airline name")
    total_flights: int = Field(..., description="Total flights for this airline")
    on_time_percentage: float = Field(..., ge=0.0, le=100.0, description="On-time percentage")
    average_delay_minutes: float = Field(..., ge=0.0, description="Average delay in minutes")
    delay_rate: float = Field(..., ge=0.0, le=1.0, description="Delay rate")
    cancellation_rate: float = Field(..., ge=0.0, le=1.0, description="Cancellation rate")
    ranking: Optional[int] = Field(None, description="Performance ranking among airlines")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "airline": "Qantas",
                "total_flights": 5000,
                "on_time_percentage": 85.5,
                "average_delay_minutes": 12.3,
                "delay_rate": 0.145,
                "cancellation_rate": 0.02,
                "ranking": 1
            }
        }


class RouteAnalytics(BaseModel):
    """
    Schema for route analytics data.
    Contains performance metrics for specific routes.
    """
    route: str = Field(..., description="Route name")
    total_flights: int = Field(..., description="Total flights on this route")
    average_delay_minutes: float = Field(..., ge=0.0, description="Average delay in minutes")
    delay_rate: float = Field(..., ge=0.0, le=1.0, description="Delay rate")
    on_time_percentage: float = Field(..., ge=0.0, le=100.0, description="On-time percentage")
    busiest_month: Optional[str] = Field(None, description="Month with highest traffic")
    least_busy_month: Optional[str] = Field(None, description="Month with lowest traffic")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "route": "Sydney-Melbourne",
                "total_flights": 10000,
                "average_delay_minutes": 8.5,
                "delay_rate": 0.12,
                "on_time_percentage": 88.0,
                "busiest_month": "December",
                "least_busy_month": "February"
            }
        }


class TimeSeriesData(BaseModel):
    """
    Schema for time series analytics data.
    Contains data points over time for trend analysis.
    """
    data_date: date = Field(..., description="Date of the data point")
    value: float = Field(..., description="Value for this date")
    category: Optional[str] = Field(None, description="Category of the data point")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat()
        }


class DashboardData(BaseModel):
    """
    Schema for complete dashboard data.
    Contains all analytics data for the dashboard.
    """
    delay_analytics: DelayAnalytics = Field(..., description="Overall delay analytics")
    airline_performance: List[AirlinePerformance] = Field(..., description="Airline performance data")
    route_analytics: List[RouteAnalytics] = Field(..., description="Route analytics data")
    time_series_data: List[TimeSeriesData] = Field(..., description="Time series data for trends")
    charts: List[ChartData] = Field(..., description="Chart data for visualization")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExportRequest(BaseModel):
    """
    Schema for data export requests.
    Allows exporting analytics data in various formats.
    """
    format: str = Field(..., description="Export format (csv, json, excel)")
    filters: Optional[AnalyticsFilter] = Field(None, description="Filters to apply before export")
    include_charts: bool = Field(False, description="Whether to include chart data in export")
    
    @validator('format')
    def validate_format(cls, v):
        """Validate export format."""
        valid_formats = ['csv', 'json', 'excel']
        if v.lower() not in valid_formats:
            raise ValueError(f'Invalid export format: {v}. Must be one of {valid_formats}')
        return v.lower()
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "format": "csv",
                "filters": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "airline": "Qantas"
                },
                "include_charts": False
            }
        }
