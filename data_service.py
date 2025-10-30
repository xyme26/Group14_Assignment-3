"""
Data service for loading and managing flight data and ML models.
Handles data loading, preprocessing, and model initialization.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import asyncio
from datetime import datetime

from utils.constants import MODEL_CONFIG, DATA_CONFIG
from utils.validators import validate_dataframe_structure

logger = logging.getLogger(__name__)


class DataService:
    """
    Service for managing flight data and ML models.
    Handles loading, preprocessing, and providing access to data and models.
    """
    
    def __init__(self):
        """Initialize the data service."""
        self.base_path = Path(__file__).parent.parent
        self.models_path = self.base_path / "models"
        self.data_path = self.base_path / "data"
        
        # Data storage
        self.flight_data: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.classification_target: Optional[pd.Series] = None
        self.regression_target: Optional[pd.Series] = None
        
        # Model storage
        self.trained_models: Optional[Dict[str, Any]] = None
        self.feature_scaler: Optional[Any] = None
        self.feature_encoders: Optional[Dict[str, Any]] = None
        self.training_metadata: Optional[Dict[str, Any]] = None
        
        # Status flags
        self._is_initialized = False
        self._models_loaded = False
        self._data_loaded = False
    
    async def initialize(self) -> None:
        """
        Initialize the data service by loading data and models.
        This method should be called during application startup.
        """
        try:
            logger.info("Initializing data service...")
            
            # Load data and models concurrently
            await asyncio.gather(
                self._load_data(),
                self._load_models()
            )
            
            self._is_initialized = True
            logger.info("Data service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize data service: {str(e)}")
            raise
    
    async def _load_data(self) -> None:
        """Load flight data from CSV files."""
        try:
            logger.info("Loading flight data...")
            
            # Load processed flight data
            flight_data_path = self.data_path / DATA_CONFIG["processed_data"]
            if not flight_data_path.exists():
                raise FileNotFoundError(f"Flight data file not found: {flight_data_path}")
            
            self.flight_data = pd.read_csv(flight_data_path)
            logger.info(f"Loaded flight data: {len(self.flight_data)} records")
            
            # Load feature matrix
            feature_matrix_path = self.data_path / DATA_CONFIG["feature_matrix"]
            if feature_matrix_path.exists():
                self.feature_matrix = pd.read_csv(feature_matrix_path, index_col=0)
                logger.info(f"Loaded feature matrix: {self.feature_matrix.shape}")
            
            # Load classification target
            classification_target_path = self.data_path / DATA_CONFIG["classification_target"]
            if classification_target_path.exists():
                classification_df = pd.read_csv(classification_target_path, index_col=0)
                self.classification_target = classification_df.iloc[:, 0]
                logger.info(f"Loaded classification target: {len(self.classification_target)} records")
            
            # Load regression target
            regression_target_path = self.data_path / DATA_CONFIG["regression_target"]
            if regression_target_path.exists():
                regression_df = pd.read_csv(regression_target_path, index_col=0)
                self.regression_target = regression_df.iloc[:, 0]
                logger.info(f"Loaded regression target: {len(self.regression_target)} records")
            
            self._data_loaded = True
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    async def _load_models(self) -> None:
        """Load trained ML models and preprocessing objects."""
        try:
            logger.info("Loading ML models...")
            
            # Load trained models
            models_path = self.models_path / MODEL_CONFIG["trained_models"]
            if not models_path.exists():
                raise FileNotFoundError(f"Trained models file not found: {models_path}")
            
            self.trained_models = joblib.load(models_path)
            logger.info(f"Loaded trained models: {list(self.trained_models.keys())}")
            
            # Load feature scaler
            scaler_path = self.models_path / MODEL_CONFIG["feature_scaler"]
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            # Load feature encoders
            encoders_path = self.models_path / MODEL_CONFIG["feature_encoders"]
            if encoders_path.exists():
                self.feature_encoders = joblib.load(encoders_path)
                logger.info(f"Loaded feature encoders: {list(self.feature_encoders.keys())}")
            
            # Load training metadata
            metadata_path = self.models_path / MODEL_CONFIG["training_metadata"]
            if metadata_path.exists():
                self.training_metadata = joblib.load(metadata_path)
                logger.info("Loaded training metadata")
            
            self._models_loaded = True
            logger.info("Model loading completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def get_flight_data(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get flight data with optional filtering.
        
        Args:
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            
        Returns:
            pd.DataFrame: Filtered flight data
        """
        if not self._data_loaded:
            raise RuntimeError("Data service not initialized")
        
        data = self.flight_data.copy()
        
        if filters:
            data = self._apply_filters(data, filters)
        
        return data
    
    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Get the feature matrix for ML predictions.
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        if not self._data_loaded:
            raise RuntimeError("Data service not initialized")
        
        return self.feature_matrix.copy()
    
    def get_classification_target(self) -> pd.Series:
        """
        Get the classification target variable.
        
        Returns:
            pd.Series: Classification target
        """
        if not self._data_loaded:
            raise RuntimeError("Data service not initialized")
        
        return self.classification_target.copy()
    
    def get_regression_target(self) -> pd.Series:
        """
        Get the regression target variable.
        
        Returns:
            pd.Series: Regression target
        """
        if not self._data_loaded:
            raise RuntimeError("Data service not initialized")
        
        return self.regression_target.copy()
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific trained model.
        
        Args:
            model_name (str): Name of the model to retrieve
            
        Returns:
            Any: Trained model object
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.trained_models[model_name]
    
    def get_feature_scaler(self) -> Any:
        """
        Get the feature scaler.
        
        Returns:
            Any: Feature scaler object
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")
        
        return self.feature_scaler
    
    def get_feature_encoders(self) -> Dict[str, Any]:
        """
        Get the feature encoders.
        
        Returns:
            Dict[str, Any]: Dictionary of feature encoders
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")
        
        return self.feature_encoders
    
    def get_training_metadata(self) -> Dict[str, Any]:
        """
        Get the training metadata.
        
        Returns:
            Dict[str, Any]: Training metadata
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")
        
        return self.training_metadata
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the flight data.
        
        Args:
            data (pd.DataFrame): Data to filter
            filters (Dict[str, Any]): Filters to apply
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = data.copy()
        
        # Apply date filters
        if 'start_date' in filters and filters['start_date']:
            start_date = pd.to_datetime(filters['start_date'])
            if 'Year' in filtered_data.columns and 'Month_Num' in filtered_data.columns:
                filtered_data['Date'] = pd.to_datetime(
                    filtered_data[['Year', 'Month_Num']].assign(day=1)
                )
                filtered_data = filtered_data[filtered_data['Date'] >= start_date]
        
        if 'end_date' in filters and filters['end_date']:
            end_date = pd.to_datetime(filters['end_date'])
            if 'Date' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Date'] <= end_date]
        
        # Apply airline filter
        if 'airline' in filters and filters['airline']:
            filtered_data = filtered_data[filtered_data['Airline'] == filters['airline']]
        
        # Apply airport filter
        if 'airport' in filters and filters['airport']:
            airport = filters['airport']
            filtered_data = filtered_data[
                (filtered_data['Departing_Port'] == airport) |
                (filtered_data['Arriving_Port'] == airport)
            ]
        
        # Apply delay risk category filter
        if 'delay_risk_category' in filters and filters['delay_risk_category']:
            filtered_data = filtered_data[
                filtered_data['Delay_Risk_Category'] == filters['delay_risk_category']
            ]
        
        # Apply route filter
        if 'route' in filters and filters['route']:
            filtered_data = filtered_data[filtered_data['Route'] == filters['route']]
        
        # Apply limit
        if 'limit' in filters and filters['limit']:
            limit = min(filters['limit'], len(filtered_data))
            filtered_data = filtered_data.head(limit)
        
        return filtered_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dict[str, Any]: Data summary information
        """
        if not self._data_loaded:
            return {"error": "Data not loaded"}
        
        summary = {
            "flight_data_records": len(self.flight_data) if self.flight_data is not None else 0,
            "feature_matrix_shape": self.feature_matrix.shape if self.feature_matrix is not None else (0, 0),
            "classification_target_records": len(self.classification_target) if self.classification_target is not None else 0,
            "regression_target_records": len(self.regression_target) if self.regression_target is not None else 0,
            "models_loaded": list(self.trained_models.keys()) if self.trained_models else [],
            "data_loaded": self._data_loaded,
            "models_loaded_flag": self._models_loaded,
            "last_updated": datetime.now().isoformat()
        }
        
        return summary
    
    def is_ready(self) -> bool:
        """
        Check if the data service is ready for use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self._is_initialized and self._data_loaded and self._models_loaded
    
    async def cleanup(self) -> None:
        """
        Clean up resources when shutting down.
        """
        logger.info("Cleaning up data service...")
        
        # Clear data from memory
        self.flight_data = None
        self.feature_matrix = None
        self.classification_target = None
        self.regression_target = None
        
        # Clear models from memory
        self.trained_models = None
        self.feature_scaler = None
        self.feature_encoders = None
        self.training_metadata = None
        
        self._is_initialized = False
        self._data_loaded = False
        self._models_loaded = False
        
        logger.info("Data service cleanup completed")
