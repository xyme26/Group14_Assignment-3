"""
Airlines and reference data API routes.
Provides reference data for airlines, airports, and other lookup information.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
import time

from utils.constants import AUSTRALIAN_AIRLINES, AUSTRALIAN_AIRPORTS, DELAY_RISK_CATEGORIES, SEASONS

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/airlines")
async def get_airlines() -> Dict[str, Any]:
    """
    Get list of Australian airlines with their codes.
    
    This endpoint provides a comprehensive list of Australian airlines
    that are supported by the prediction system.
    
    Returns:
        Dict[str, Any]: List of airlines with their IATA codes
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        airlines_list = [
            {
                "name": name,
                "code": code,
                "country": "Australia"
            }
            for name, code in AUSTRALIAN_AIRLINES.items()
        ]
        
        return {
            "success": True,
            "airlines": airlines_list,
            "total_count": len(airlines_list),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting airlines list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting airlines list")


@router.get("/airports")
async def get_airports() -> Dict[str, Any]:
    """
    Get list of Australian airports with their codes.
    
    This endpoint provides a comprehensive list of Australian airports
    that are supported by the prediction system.
    
    Returns:
        Dict[str, Any]: List of airports with their IATA codes
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        airports_list = [
            {
                "name": name,
                "code": code,
                "country": "Australia"
            }
            for name, code in AUSTRALIAN_AIRPORTS.items()
        ]
        
        return {
            "success": True,
            "airports": airports_list,
            "total_count": len(airports_list),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting airports list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting airports list")


@router.get("/delay-risk-categories")
async def get_delay_risk_categories() -> Dict[str, Any]:
    """
    Get list of delay risk categories.
    
    This endpoint provides the available delay risk categories
    used in the classification model.
    
    Returns:
        Dict[str, Any]: List of delay risk categories
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        categories_list = [
            {
                "category": category,
                "value": value,
                "description": f"Delay risk level: {category}"
            }
            for category, value in DELAY_RISK_CATEGORIES.items()
        ]
        
        return {
            "success": True,
            "categories": categories_list,
            "total_count": len(categories_list),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting delay risk categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting delay risk categories")


@router.get("/seasons")
async def get_seasons() -> Dict[str, Any]:
    """
    Get list of seasons.
    
    This endpoint provides the available seasons used in the system
    for temporal analysis and feature engineering.
    
    Returns:
        Dict[str, Any]: List of seasons
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        seasons_list = [
            {
                "name": season,
                "value": value,
                "description": f"Season: {season}"
            }
            for season, value in SEASONS.items()
        ]
        
        return {
            "success": True,
            "seasons": seasons_list,
            "total_count": len(seasons_list),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting seasons list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting seasons list")


@router.get("/months")
async def get_months() -> Dict[str, Any]:
    """
    Get list of months with their numbers and names.
    
    This endpoint provides month information for date selection
    and temporal analysis.
    
    Returns:
        Dict[str, Any]: List of months with numbers and names
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        months_list = [
            {"number": 1, "name": "January", "abbreviation": "Jan"},
            {"number": 2, "name": "February", "abbreviation": "Feb"},
            {"number": 3, "name": "March", "abbreviation": "Mar"},
            {"number": 4, "name": "April", "abbreviation": "Apr"},
            {"number": 5, "name": "May", "abbreviation": "May"},
            {"number": 6, "name": "June", "abbreviation": "Jun"},
            {"number": 7, "name": "July", "abbreviation": "Jul"},
            {"number": 8, "name": "August", "abbreviation": "Aug"},
            {"number": 9, "name": "September", "abbreviation": "Sep"},
            {"number": 10, "name": "October", "abbreviation": "Oct"},
            {"number": 11, "name": "November", "abbreviation": "Nov"},
            {"number": 12, "name": "December", "abbreviation": "Dec"}
        ]
        
        return {
            "success": True,
            "months": months_list,
            "total_count": len(months_list),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting months list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting months list")


@router.get("/years")
async def get_years() -> Dict[str, Any]:
    """
    Get list of supported years.
    
    This endpoint provides the range of years supported by the system
    for temporal analysis and predictions.
    
    Returns:
        Dict[str, Any]: List of supported years
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        # Generate years from 2020 to 2030
        years_list = [
            {
                "year": year,
                "description": f"Year {year}"
            }
            for year in range(2020, 2031)
        ]
        
        return {
            "success": True,
            "years": years_list,
            "total_count": len(years_list),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting years list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting years list")


@router.get("/reference-data")
async def get_all_reference_data() -> Dict[str, Any]:
    """
    Get all reference data in a single response.
    
    This endpoint provides all reference data including airlines,
    airports, categories, seasons, months, and years in one call.
    
    Returns:
        Dict[str, Any]: Complete reference data
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        # Get all reference data
        airlines_response = await get_airlines()
        airports_response = await get_airports()
        categories_response = await get_delay_risk_categories()
        seasons_response = await get_seasons()
        months_response = await get_months()
        years_response = await get_years()
        
        return {
            "success": True,
            "reference_data": {
                "airlines": airlines_response["airlines"],
                "airports": airports_response["airports"],
                "delay_risk_categories": categories_response["categories"],
                "seasons": seasons_response["seasons"],
                "months": months_response["months"],
                "years": years_response["years"]
            },
            "counts": {
                "airlines": airlines_response["total_count"],
                "airports": airports_response["total_count"],
                "delay_risk_categories": categories_response["total_count"],
                "seasons": seasons_response["total_count"],
                "months": months_response["total_count"],
                "years": years_response["total_count"]
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting all reference data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting reference data")


@router.get("/airline/{airline_name}")
async def get_airline_details(airline_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific airline.
    
    This endpoint provides detailed information about a specific airline
    including its code and additional metadata.
    
    Args:
        airline_name (str): Name of the airline to get details for
        
    Returns:
        Dict[str, Any]: Detailed airline information
        
    Raises:
        HTTPException: If airline not found or service is unavailable
    """
    try:
        if airline_name not in AUSTRALIAN_AIRLINES:
            raise HTTPException(status_code=404, detail=f"Airline '{airline_name}' not found")
        
        airline_code = AUSTRALIAN_AIRLINES[airline_name]
        
        return {
            "success": True,
            "airline": {
                "name": airline_name,
                "code": airline_code,
                "country": "Australia",
                "description": f"Australian airline: {airline_name} ({airline_code})"
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting airline details for {airline_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting airline details")


@router.get("/airport/{airport_name}")
async def get_airport_details(airport_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific airport.
    
    This endpoint provides detailed information about a specific airport
    including its code and additional metadata.
    
    Args:
        airport_name (str): Name of the airport to get details for
        
    Returns:
        Dict[str, Any]: Detailed airport information
        
    Raises:
        HTTPException: If airport not found or service is unavailable
    """
    try:
        if airport_name not in AUSTRALIAN_AIRPORTS:
            raise HTTPException(status_code=404, detail=f"Airport '{airport_name}' not found")
        
        airport_code = AUSTRALIAN_AIRPORTS[airport_name]
        
        return {
            "success": True,
            "airport": {
                "name": airport_name,
                "code": airport_code,
                "country": "Australia",
                "description": f"Australian airport: {airport_name} ({airport_code})"
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting airport details for {airport_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting airport details")


@router.get("/search")
async def search_reference_data(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Category to search in (airlines, airports, all)")
) -> Dict[str, Any]:
    """
    Search reference data by query.
    
    This endpoint allows searching through reference data using a text query.
    It can search across airlines, airports, or all categories.
    
    Args:
        query (str): Search query
        category (Optional[str]): Category to search in
        
    Returns:
        Dict[str, Any]: Search results
        
    Raises:
        HTTPException: If search fails or service is unavailable
    """
    try:
        query_lower = query.lower()
        results = {
            "airlines": [],
            "airports": [],
            "total_matches": 0
        }
        
        # Search airlines
        if category is None or category == "airlines" or category == "all":
            for name, code in AUSTRALIAN_AIRLINES.items():
                if (query_lower in name.lower() or 
                    query_lower in code.lower()):
                    results["airlines"].append({
                        "name": name,
                        "code": code,
                        "match_type": "airline"
                    })
        
        # Search airports
        if category is None or category == "airports" or category == "all":
            for name, code in AUSTRALIAN_AIRPORTS.items():
                if (query_lower in name.lower() or 
                    query_lower in code.lower()):
                    results["airports"].append({
                        "name": name,
                        "code": code,
                        "match_type": "airport"
                    })
        
        # Calculate total matches
        results["total_matches"] = len(results["airlines"]) + len(results["airports"])
        
        return {
            "success": True,
            "query": query,
            "category": category or "all",
            "results": results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error searching reference data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during search")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for airlines service.
    
    Returns:
        Dict[str, Any]: Health status of the airlines service
    """
    try:
        return {
            "status": "healthy",
            "service": "airlines",
            "reference_data_available": True,
            "airlines_count": len(AUSTRALIAN_AIRLINES),
            "airports_count": len(AUSTRALIAN_AIRPORTS),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Airlines service health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "airlines",
            "reference_data_available": False,
            "error": str(e),
            "timestamp": time.time()
        }
