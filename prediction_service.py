"""
Prediction service for flight delay predictions.
Handles ML model predictions and data preprocessing for real-time predictions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from .data_service import DataService
from utils.validators import validate_prediction_input
from utils.constants import DELAY_RISK_CATEGORIES, PEAK_SEASON_MONTHS

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for making flight delay predictions using trained ML models.
    Handles data preprocessing, feature engineering, and model predictions.
    """
    
    def __init__(self, data_service: DataService):
        """
        Initialize the prediction service.
        
        Args:
            data_service (DataService): Data service instance for accessing models and data
        """
        self.data_service = data_service
        self._feature_columns = None
        self._categorical_columns = None
        self._numerical_columns = None
    
    def _initialize_feature_info(self) -> None:
        """Initialize feature column information from training metadata."""
        if self._feature_columns is not None:
            return
        
        try:
            metadata = self.data_service.get_training_metadata()
            if metadata and 'feature_columns' in metadata:
                self._feature_columns = metadata['feature_columns']
                self._categorical_columns = metadata.get('categorical_columns', [])
                self._numerical_columns = metadata.get('numerical_columns', [])
            else:
                # Fallback: infer from feature matrix
                feature_matrix = self.data_service.get_feature_matrix()
                if feature_matrix is not None:
                    self._feature_columns = feature_matrix.columns.tolist()
                    self._categorical_columns = feature_matrix.select_dtypes(include=['object', 'category']).columns.tolist()
                    self._numerical_columns = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
            
            logger.info(f"Initialized feature info: {len(self._feature_columns)} total features")
            
        except Exception as e:
            logger.error(f"Failed to initialize feature info: {str(e)}")
            raise
    
    def _preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for ML model prediction.
        
        Args:
            input_data (Dict[str, Any]): Raw input data
            
        Returns:
            np.ndarray: Preprocessed feature array
        """
        try:
            self._initialize_feature_info()
            
            # Create a DataFrame from input data
            df = pd.DataFrame([input_data])
            
            # Feature engineering
            df = self._engineer_features(df)
            
            # Encode categorical variables
            df = self._encode_categorical_features(df)
            
            # Select and order features
            if self._feature_columns:
                missing_features = set(self._feature_columns) - set(df.columns)
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
                    # Add missing features with default values
                    for feature in missing_features:
                        df[feature] = 0
                
                df = df[self._feature_columns]
            
            # Scale features
            scaler = self.data_service.get_feature_scaler()
            if scaler:
                features_array = scaler.transform(df.values)
            else:
                features_array = df.values
            
            return features_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess input: {str(e)}")
            raise
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from input data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        try:
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Calculate derived features
            if 'Sectors_Scheduled' in df.columns and 'Sectors_Flown' in df.columns:
                df['Sectors_Flown'] = df['Sectors_Flown'].fillna(df['Sectors_Scheduled'])
                df['Cancellations'] = df['Cancellations'].fillna(0)
                df['Departures_On_Time'] = df['Departures_On_Time'].fillna(0)
                df['Arrivals_On_Time'] = df['Arrivals_On_Time'].fillna(0)
                df['Departures_Delayed'] = df['Departures_Delayed'].fillna(0)
                df['Arrivals_Delayed'] = df['Arrivals_Delayed'].fillna(0)
            
            # Calculate delay rate
            if 'Departures_Delayed' in df.columns and 'Sectors_Flown' in df.columns:
                df['Overall_Delay_Rate'] = df['Departures_Delayed'] / df['Sectors_Flown'].replace(0, 1)
            else:
                df['Overall_Delay_Rate'] = 0.0
            
            # Calculate estimated delay minutes (placeholder - will be predicted)
            df['Estimated_Delay_Minutes'] = 0.0
            
            # Add year if not present
            if 'Year' not in df.columns:
                df['Year'] = datetime.now().year
            
            # Add month number if not present
            if 'Month_Num' not in df.columns and 'Month' in df.columns:
                df['Month_Num'] = df['Month']
            
            # Add season information
            if 'Month_Num' in df.columns:
                df['Season'] = df['Month_Num'].apply(self._get_season)
                df['Season_encoded'] = df['Season'].map({'Summer': 0, 'Autumn': 1, 'Winter': 2, 'Spring': 3})
            
            # Add peak season flag
            if 'Month_Num' in df.columns:
                df['Is_Peak_Season'] = df['Month_Num'].isin(PEAK_SEASON_MONTHS).astype(int)
            
            # Add route statistics (placeholder values - in real scenario, these would be calculated from historical data)
            df['Route_Total_Flights'] = df['Sectors_Scheduled']
            df['Route_Avg_Delay_Rate'] = df['Overall_Delay_Rate']
            df['Airline_Avg_Delay_Rate'] = 0.2  # Placeholder
            df['Departure_Port_Delay_Rate'] = 0.2  # Placeholder
            df['Arrival_Port_Delay_Rate'] = 0.2  # Placeholder
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to engineer features: {str(e)}")
            raise
    
    def _get_season(self, month: int) -> str:
        """Get season name from month number."""
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using trained encoders.
        
        Args:
            df (pd.DataFrame): DataFrame with categorical features
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        try:
            encoders = self.data_service.get_feature_encoders()
            if not encoders:
                logger.warning("No feature encoders available")
                return df
            
            df_encoded = df.copy()
            
            # Encode each categorical feature
            for feature, encoder in encoders.items():
                if feature in df_encoded.columns:
                    try:
                        # Handle unseen categories
                        df_encoded[feature] = df_encoded[feature].astype(str)
                        encoded_values = encoder.transform(df_encoded[feature].values.reshape(-1, 1))
                        df_encoded[feature + '_encoded'] = encoded_values.flatten()
                    except Exception as e:
                        logger.warning(f"Failed to encode {feature}: {str(e)}")
                        df_encoded[feature + '_encoded'] = 0
            
            return df_encoded
            
        except Exception as e:
            logger.error(f"Failed to encode categorical features: {str(e)}")
            raise
    
    def predict_delay_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict delay risk category for a flight.
        
        Args:
            input_data (Dict[str, Any]): Input data for prediction
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        start_time = time.time()
        
        try:
            # Validate input data
            validated_data = validate_prediction_input(input_data)
            
            # Preprocess input
            features = self._preprocess_input(validated_data)
            
            # Get classification model
            models = self.data_service.get_trained_models()
            if 'classification' not in models:
                raise ValueError("Classification model not found")
            
            classification_model = models['classification']
            
            # Make prediction
            prediction = classification_model.predict(features)
            prediction_proba = classification_model.predict_proba(features)
            
            # Get confidence scores
            classes = classification_model.classes_
            confidence_scores = dict(zip(classes, prediction_proba[0]))
            
            # Map prediction to category name
            predicted_category = classes[prediction[0]]
            probability = float(prediction_proba[0][prediction[0]])
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "predicted_category": predicted_category,
                "confidence_scores": confidence_scores,
                "probability": probability,
                "processing_time_ms": processing_time,
                "model_info": {
                    "model_type": "classification",
                    "model_name": type(classification_model).__name__,
                    "features_used": len(features[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Classification prediction failed: {str(e)}")
            raise
    
    def predict_delay_duration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict delay duration in minutes for a flight.
        
        Args:
            input_data (Dict[str, Any]): Input data for prediction
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        start_time = time.time()
        
        try:
            # Validate input data
            validated_data = validate_prediction_input(input_data)
            
            # Preprocess input
            features = self._preprocess_input(validated_data)
            
            # Get regression model
            models = self.data_service.get_trained_models()
            if 'regression' not in models:
                raise ValueError("Regression model not found")
            
            regression_model = models['regression']
            
            # Make prediction
            prediction = regression_model.predict(features)
            predicted_delay = float(prediction[0])
            
            # Calculate confidence interval (if model supports it)
            confidence_interval = None
            if hasattr(regression_model, 'predict_interval'):
                try:
                    interval = regression_model.predict_interval(features, alpha=0.05)
                    confidence_interval = {
                        "lower": float(interval[0][0]),
                        "upper": float(interval[0][1])
                    }
                except:
                    pass
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "predicted_delay_minutes": max(0, predicted_delay),  # Ensure non-negative
                "confidence_interval": confidence_interval,
                "processing_time_ms": processing_time,
                "model_info": {
                    "model_type": "regression",
                    "model_name": type(regression_model).__name__,
                    "features_used": len(features[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Regression prediction failed: {str(e)}")
            raise
    
    def predict_both(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make both classification and regression predictions.
        
        Args:
            input_data (Dict[str, Any]): Input data for prediction
            
        Returns:
            Dict[str, Any]: Combined prediction results
        """
        try:
            # Make both predictions
            classification_result = self.predict_delay_risk(input_data)
            regression_result = self.predict_delay_duration(input_data)
            
            # Combine results
            combined_result = {
                "classification": classification_result,
                "regression": regression_result,
                "input_data": input_data,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Combined prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make predictions for multiple inputs in batch.
        
        Args:
            input_data_list (List[Dict[str, Any]]): List of input data
            
        Returns:
            Dict[str, Any]: Batch prediction results
        """
        start_time = time.time()
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        try:
            for i, input_data in enumerate(input_data_list):
                try:
                    result = self.predict_both(input_data)
                    result["batch_index"] = i
                    results.append(result)
                    successful_predictions += 1
                except Exception as e:
                    logger.error(f"Batch prediction {i} failed: {str(e)}")
                    results.append({
                        "batch_index": i,
                        "error": str(e),
                        "success": False
                    })
                    failed_predictions += 1
            
            total_processing_time = (time.time() - start_time) * 1000
            
            return {
                "results": results,
                "total_predictions": len(input_data_list),
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "processing_time_ms": total_processing_time,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            models = self.data_service.get_trained_models()
            metadata = self.data_service.get_training_metadata()
            
            model_info = {
                "available_models": list(models.keys()) if models else [],
                "model_types": {},
                "training_metadata": metadata,
                "feature_columns": self._feature_columns,
                "categorical_columns": self._categorical_columns,
                "numerical_columns": self._numerical_columns
            }
            
            if models:
                for name, model in models.items():
                    model_info["model_types"][name] = type(model).__name__
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise
