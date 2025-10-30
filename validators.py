"""
Input validation utilities for the Flight Delay Prediction API.
Provides validation functions for user inputs and data integrity checks.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ValidationError
import pandas as pd
import numpy as np
from datetime import datetime

from .constants import VALIDATION_RULES, AUSTRALIAN_AIRLINES, AUSTRALIAN_AIRPORTS


class ValidationError(Exception):
    """Custom validation error for input validation failures."""
    pass


def validate_airline(airline: str) -> bool:
    """
    Validate if the provided airline is a valid Australian airline.
    
    Args:
        airline (str): Airline name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return airline in AUSTRALIAN_AIRLINES


def validate_airport(airport: str) -> bool:
    """
    Validate if the provided airport is a valid Australian airport.
    
    Args:
        airport (str): Airport name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return airport in AUSTRALIAN_AIRPORTS


def validate_month(month: int) -> bool:
    """
    Validate if the provided month is within valid range.
    
    Args:
        month (int): Month number to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return VALIDATION_RULES["min_month"] <= month <= VALIDATION_RULES["max_month"]


def validate_year(year: int) -> bool:
    """
    Validate if the provided year is within valid range.
    
    Args:
        year (int): Year to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return VALIDATION_RULES["min_year"] <= year <= VALIDATION_RULES["max_year"]


def validate_sectors_scheduled(sectors: int) -> bool:
    """
    Validate if the number of scheduled sectors is within valid range.
    
    Args:
        sectors (int): Number of scheduled sectors to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return (VALIDATION_RULES["min_sectors_scheduled"] <= 
            sectors <= VALIDATION_RULES["max_sectors_scheduled"])


def validate_delay_minutes(delay_minutes: float) -> bool:
    """
    Validate if delay minutes are within valid range.
    
    Args:
        delay_minutes (float): Delay minutes to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return (VALIDATION_RULES["min_delay_minutes"] <= 
            delay_minutes <= VALIDATION_RULES["max_delay_minutes"])


def validate_route(route: str) -> bool:
    """
    Validate if the route format is correct (Origin-Destination).
    
    Args:
        route (str): Route string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not route or not isinstance(route, str):
        return False
    
    # Check if route contains a dash separator
    if "-" not in route:
        return False
    
    # Split route into origin and destination
    parts = route.split("-")
    if len(parts) != 2:
        return False
    
    origin, destination = parts[0].strip(), parts[1].strip()
    
    # Validate both origin and destination are valid airports
    return validate_airport(origin) and validate_airport(destination)


def validate_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate prediction input data and return cleaned data.
    
    Args:
        data (Dict[str, Any]): Input data to validate
        
    Returns:
        Dict[str, Any]: Cleaned and validated data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    cleaned_data = {}
    
    # Required fields for prediction
    required_fields = [
        "route", "departing_port", "arriving_port", "airline", 
        "month", "sectors_scheduled"
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        else:
            cleaned_data[field] = data[field]
    
    # Validate route
    if "route" in data and not validate_route(data["route"]):
        errors.append("Invalid route format. Expected format: 'Origin-Destination'")
    
    # Validate airports
    if "departing_port" in data and not validate_airport(data["departing_port"]):
        errors.append(f"Invalid departing port: {data['departing_port']}")
    
    if "arriving_port" in data and not validate_airport(data["arriving_port"]):
        errors.append(f"Invalid arriving port: {data['arriving_port']}")
    
    # Validate airline
    if "airline" in data and not validate_airline(data["airline"]):
        errors.append(f"Invalid airline: {data['airline']}")
    
    # Validate month
    if "month" in data:
        try:
            month = int(data["month"])
            if not validate_month(month):
                errors.append(f"Invalid month: {month}. Must be between 1-12")
            else:
                cleaned_data["month"] = month
        except (ValueError, TypeError):
            errors.append("Month must be a valid integer")
    
    # Validate year (optional, defaults to current year)
    if "year" in data:
        try:
            year = int(data["year"])
            if not validate_year(year):
                errors.append(f"Invalid year: {year}. Must be between 2020-2030")
            else:
                cleaned_data["year"] = year
        except (ValueError, TypeError):
            errors.append("Year must be a valid integer")
    else:
        cleaned_data["year"] = datetime.now().year
    
    # Validate sectors scheduled
    if "sectors_scheduled" in data:
        try:
            sectors = int(data["sectors_scheduled"])
            if not validate_sectors_scheduled(sectors):
                errors.append(f"Invalid sectors scheduled: {sectors}. Must be between 1-1000")
            else:
                cleaned_data["sectors_scheduled"] = sectors
        except (ValueError, TypeError):
            errors.append("Sectors scheduled must be a valid integer")
    
    # Optional fields with defaults
    cleaned_data.setdefault("sectors_flown", cleaned_data.get("sectors_scheduled", 0))
    cleaned_data.setdefault("cancellations", 0)
    cleaned_data.setdefault("departures_on_time", 0)
    cleaned_data.setdefault("arrivals_on_time", 0)
    cleaned_data.setdefault("departures_delayed", 0)
    cleaned_data.setdefault("arrivals_delayed", 0)
    
    if errors:
        raise ValidationError(f"Validation failed: {'; '.join(errors)}")
    
    return cleaned_data


def validate_analytics_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate analytics filter parameters.
    
    Args:
        filters (Dict[str, Any]): Filter parameters to validate
        
    Returns:
        Dict[str, Any]: Cleaned and validated filters
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    cleaned_filters = {}
    
    # Validate date range
    if "start_date" in filters:
        try:
            start_date = pd.to_datetime(filters["start_date"])
            cleaned_filters["start_date"] = start_date
        except (ValueError, TypeError):
            errors.append("Invalid start_date format. Use YYYY-MM-DD")
    
    if "end_date" in filters:
        try:
            end_date = pd.to_datetime(filters["end_date"])
            cleaned_filters["end_date"] = end_date
        except (ValueError, TypeError):
            errors.append("Invalid end_date format. Use YYYY-MM-DD")
    
    # Validate airline filter
    if "airline" in filters:
        if not validate_airline(filters["airline"]):
            errors.append(f"Invalid airline filter: {filters['airline']}")
        else:
            cleaned_filters["airline"] = filters["airline"]
    
    # Validate airport filter
    if "airport" in filters:
        if not validate_airport(filters["airport"]):
            errors.append(f"Invalid airport filter: {filters['airport']}")
        else:
            cleaned_filters["airport"] = filters["airport"]
    
    # Validate delay risk category filter
    if "delay_risk_category" in filters:
        valid_categories = ["Low", "Medium", "High"]
        if filters["delay_risk_category"] not in valid_categories:
            errors.append(f"Invalid delay risk category: {filters['delay_risk_category']}")
        else:
            cleaned_filters["delay_risk_category"] = filters["delay_risk_category"]
    
    # Validate limit parameter
    if "limit" in filters:
        try:
            limit = int(filters["limit"])
            if limit <= 0 or limit > 10000:
                errors.append("Limit must be between 1 and 10000")
            else:
                cleaned_filters["limit"] = limit
        except (ValueError, TypeError):
            errors.append("Limit must be a valid integer")
    else:
        cleaned_filters["limit"] = 1000  # Default limit
    
    if errors:
        raise ValidationError(f"Filter validation failed: {'; '.join(errors)}")
    
    return cleaned_filters


def sanitize_string_input(value: Any) -> str:
    """
    Sanitize string input to prevent injection attacks.
    
    Args:
        value (Any): Input value to sanitize
        
    Returns:
        str: Sanitized string
    """
    if not isinstance(value, str):
        return str(value)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`', '$']
    sanitized = value
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    return sanitized.strip()


def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame has the required structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if valid, False otherwise
    """
    if df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    return len(missing_columns) == 0
