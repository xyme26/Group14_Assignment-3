"""
Constants and configuration values for the Flight Delay Prediction API.
Contains airline codes, airport codes, and other reference data.
"""

# API Configuration
API_VERSION = "1.0.0"
APP_NAME = "Flight Delay Prediction API"
APP_DESCRIPTION = """
A comprehensive API for predicting flight delays and analyzing aviation data.
Integrates machine learning models to provide real-time predictions and analytics
for flight delay risk assessment and delay duration estimation.
"""

# Australian Airlines
AUSTRALIAN_AIRLINES = {
    "Qantas": "QF",
    "Jetstar": "JQ", 
    "Virgin Australia": "VA",
    "Rex": "ZL",
    "Alliance Airlines": "QQ",
    "Skytrans": "QN",
    "Cobham Aviation": "NC"
}

# Major Australian Airports
AUSTRALIAN_AIRPORTS = {
    "Sydney": "SYD",
    "Melbourne": "MEL", 
    "Brisbane": "BNE",
    "Perth": "PER",
    "Adelaide": "ADL",
    "Darwin": "DRW",
    "Hobart": "HBA",
    "Canberra": "CBR",
    "Cairns": "CNS",
    "Gold Coast": "OOL",
    "Newcastle": "NTL",
    "Townsville": "TSV",
    "Alice Springs": "ASP",
    "Ayers Rock": "AYQ",
    "Kalgoorlie": "KGI",
    "Newman": "ZNE"
}

# Delay Risk Categories
DELAY_RISK_CATEGORIES = {
    "Low": 0,
    "Medium": 1, 
    "High": 2
}

# Seasons
SEASONS = {
    "Summer": 0,
    "Autumn": 1,
    "Winter": 2,
    "Spring": 3
}

# Peak Season Months (Australian summer and winter holidays)
PEAK_SEASON_MONTHS = [1, 2, 6, 7, 12]

# Model Configuration
MODEL_CONFIG = {
    "trained_models": "trained_models.joblib",
    "classification_model": "trained_models.joblib",
    "regression_model": "trained_models.joblib", 
    "feature_scaler": "feature_scaler.joblib",
    "feature_encoders": "feature_encoders.joblib",
    "training_metadata": "training_metadata.joblib"
}

# Data Configuration
DATA_CONFIG = {
    "processed_data": "processed_flight_data.csv",
    "feature_matrix": "feature_matrix.csv", 
    "classification_target": "classification_target.csv",
    "regression_target": "regression_target.csv"
}

# API Response Messages
MESSAGES = {
    "success": "Operation completed successfully",
    "prediction_success": "Prediction generated successfully",
    "data_loaded": "Data loaded successfully",
    "model_loaded": "Model loaded successfully",
    "validation_error": "Input validation failed",
    "model_error": "Model prediction failed",
    "data_error": "Data processing failed",
    "not_found": "Resource not found",
    "server_error": "Internal server error"
}

# Validation Rules
VALIDATION_RULES = {
    "min_sectors_scheduled": 1,
    "max_sectors_scheduled": 1000,
    "min_year": 2020,
    "max_year": 2030,
    "min_month": 1,
    "max_month": 12,
    "min_delay_minutes": 0,
    "max_delay_minutes": 1440  # 24 hours in minutes
}

# Chart Configuration for Analytics
CHART_CONFIG = {
    "delay_by_airline": {
        "title": "Average Delay Rate by Airline",
        "x_axis": "Airline",
        "y_axis": "Average Delay Rate (%)"
    },
    "delay_by_route": {
        "title": "Delay Distribution by Route",
        "x_axis": "Route",
        "y_axis": "Number of Flights"
    },
    "delay_by_month": {
        "title": "Delay Trends by Month",
        "x_axis": "Month",
        "y_axis": "Average Delay Rate (%)"
    },
    "delay_by_season": {
        "title": "Delay Patterns by Season",
        "x_axis": "Season", 
        "y_axis": "Average Delay Rate (%)"
    }
}
