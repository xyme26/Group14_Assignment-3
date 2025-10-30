"""
Analytics service for flight delay data analysis and visualization.
Handles data aggregation, statistics calculation, and chart data generation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import json

from .data_service import DataService
from utils.validators import validate_analytics_filters
from utils.constants import CHART_CONFIG, DELAY_RISK_CATEGORIES

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service for analyzing flight delay data and generating analytics.
    Handles data aggregation, statistics calculation, and chart data preparation.
    """
    
    def __init__(self, data_service: DataService):
        """
        Initialize the analytics service.
        
        Args:
            data_service (DataService): Data service instance for accessing data
        """
        self.data_service = data_service
    
    def get_delay_analytics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get overall delay analytics for the dataset.
        
        Args:
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            
        Returns:
            Dict[str, Any]: Delay analytics data
        """
        try:
            # Get filtered data
            data = self.data_service.get_flight_data(filters)
            
            if data.empty:
                return {
                    "total_flights": 0,
                    "delayed_flights": 0,
                    "on_time_flights": 0,
                    "cancellation_rate": 0.0,
                    "average_delay_minutes": 0.0,
                    "median_delay_minutes": 0.0,
                    "delay_risk_distribution": {}
                }
            
            # Calculate basic statistics
            total_flights = len(data)
            delayed_flights = len(data[data['Overall_Delay_Rate'] > 0])
            on_time_flights = total_flights - delayed_flights
            
            # Calculate cancellation rate
            if 'Cancellations' in data.columns and 'Sectors_Scheduled' in data.columns:
                total_cancellations = data['Cancellations'].sum()
                total_scheduled = data['Sectors_Scheduled'].sum()
                cancellation_rate = total_cancellations / total_scheduled if total_scheduled > 0 else 0.0
            else:
                cancellation_rate = 0.0
            
            # Calculate delay statistics
            if 'Estimated_Delay_Minutes' in data.columns:
                delay_data = data[data['Estimated_Delay_Minutes'] > 0]['Estimated_Delay_Minutes']
                average_delay = delay_data.mean() if len(delay_data) > 0 else 0.0
                median_delay = delay_data.median() if len(delay_data) > 0 else 0.0
            else:
                average_delay = 0.0
                median_delay = 0.0
            
            # Calculate delay risk distribution
            delay_risk_distribution = {}
            if 'Delay_Risk_Category' in data.columns:
                risk_counts = data['Delay_Risk_Category'].value_counts()
                for category in DELAY_RISK_CATEGORIES.keys():
                    delay_risk_distribution[category] = int(risk_counts.get(category, 0))
            else:
                # Estimate based on delay rate
                low_risk = len(data[data['Overall_Delay_Rate'] <= 0.1])
                medium_risk = len(data[(data['Overall_Delay_Rate'] > 0.1) & (data['Overall_Delay_Rate'] <= 0.3)])
                high_risk = len(data[data['Overall_Delay_Rate'] > 0.3])
                
                delay_risk_distribution = {
                    "Low": low_risk,
                    "Medium": medium_risk,
                    "High": high_risk
                }
            
            return {
                "total_flights": int(total_flights),
                "delayed_flights": int(delayed_flights),
                "on_time_flights": int(on_time_flights),
                "cancellation_rate": float(cancellation_rate),
                "average_delay_minutes": float(average_delay),
                "median_delay_minutes": float(median_delay),
                "delay_risk_distribution": delay_risk_distribution
            }
            
        except Exception as e:
            logger.error(f"Failed to get delay analytics: {str(e)}")
            raise
    
    def get_airline_performance(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get airline performance analytics.
        
        Args:
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            
        Returns:
            List[Dict[str, Any]]: List of airline performance data
        """
        try:
            # Get filtered data
            data = self.data_service.get_flight_data(filters)
            
            if data.empty or 'Airline' not in data.columns:
                return []
            
            # Group by airline and calculate metrics
            airline_stats = data.groupby('Airline').agg({
                'Sectors_Scheduled': 'sum',
                'Sectors_Flown': 'sum',
                'Cancellations': 'sum',
                'Overall_Delay_Rate': 'mean',
                'Estimated_Delay_Minutes': 'mean'
            }).reset_index()
            
            # Calculate performance metrics
            performance_data = []
            for _, row in airline_stats.iterrows():
                total_flights = int(row['Sectors_Scheduled'])
                on_time_flights = int(row['Sectors_Flown'] - row['Cancellations'])
                delay_rate = float(row['Overall_Delay_Rate'])
                avg_delay = float(row['Estimated_Delay_Minutes'])
                cancellation_rate = float(row['Cancellations'] / total_flights) if total_flights > 0 else 0.0
                on_time_percentage = float((on_time_flights / total_flights) * 100) if total_flights > 0 else 0.0
                
                performance_data.append({
                    "airline": row['Airline'],
                    "total_flights": total_flights,
                    "on_time_percentage": on_time_percentage,
                    "average_delay_minutes": avg_delay,
                    "delay_rate": delay_rate,
                    "cancellation_rate": cancellation_rate
                })
            
            # Sort by on-time percentage (descending)
            performance_data.sort(key=lambda x: x['on_time_percentage'], reverse=True)
            
            # Add ranking
            for i, airline_data in enumerate(performance_data):
                airline_data['ranking'] = i + 1
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get airline performance: {str(e)}")
            raise
    
    def get_route_analytics(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get route analytics data.
        
        Args:
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            
        Returns:
            List[Dict[str, Any]]: List of route analytics data
        """
        try:
            # Get filtered data
            data = self.data_service.get_flight_data(filters)
            
            if data.empty or 'Route' not in data.columns:
                return []
            
            # Group by route and calculate metrics
            route_stats = data.groupby('Route').agg({
                'Sectors_Scheduled': 'sum',
                'Overall_Delay_Rate': 'mean',
                'Estimated_Delay_Minutes': 'mean',
                'Month_Num': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
            }).reset_index()
            
            # Calculate route metrics
            route_data = []
            for _, row in route_stats.iterrows():
                total_flights = int(row['Sectors_Scheduled'])
                avg_delay = float(row['Estimated_Delay_Minutes'])
                delay_rate = float(row['Overall_Delay_Rate'])
                on_time_percentage = float((1 - delay_rate) * 100)
                
                # Find busiest and least busy months
                route_data_filtered = data[data['Route'] == row['Route']]
                if not route_data_filtered.empty and 'Month_Num' in route_data_filtered.columns:
                    month_counts = route_data_filtered['Month_Num'].value_counts()
                    busiest_month = month_counts.idxmax() if len(month_counts) > 0 else None
                    least_busy_month = month_counts.idxmin() if len(month_counts) > 0 else None
                    
                    # Convert month numbers to names
                    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                                 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                    
                    busiest_month_name = month_names.get(busiest_month) if busiest_month else None
                    least_busy_month_name = month_names.get(least_busy_month) if least_busy_month else None
                else:
                    busiest_month_name = None
                    least_busy_month_name = None
                
                route_data.append({
                    "route": row['Route'],
                    "total_flights": total_flights,
                    "average_delay_minutes": avg_delay,
                    "delay_rate": delay_rate,
                    "on_time_percentage": on_time_percentage,
                    "busiest_month": busiest_month_name,
                    "least_busy_month": least_busy_month_name
                })
            
            # Sort by total flights (descending)
            route_data.sort(key=lambda x: x['total_flights'], reverse=True)
            
            return route_data
            
        except Exception as e:
            logger.error(f"Failed to get route analytics: {str(e)}")
            raise
    
    def get_time_series_data(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get time series data for trend analysis.
        
        Args:
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            
        Returns:
            List[Dict[str, Any]]: List of time series data points
        """
        try:
            # Get filtered data
            data = self.data_service.get_flight_data(filters)
            
            if data.empty:
                return []
            
            # Create date column if not exists
            if 'Year' in data.columns and 'Month_Num' in data.columns:
                data['Date'] = pd.to_datetime(data[['Year', 'Month_Num']].assign(day=1))
            else:
                return []
            
            # Group by date and calculate metrics
            time_series = data.groupby('Date').agg({
                'Sectors_Scheduled': 'sum',
                'Overall_Delay_Rate': 'mean',
                'Estimated_Delay_Minutes': 'mean'
            }).reset_index()
            
            # Convert to list of dictionaries
            time_series_data = []
            for _, row in time_series.iterrows():
                time_series_data.append({
                    "date": row['Date'].strftime('%Y-%m-%d'),
                    "total_flights": int(row['Sectors_Scheduled']),
                    "average_delay_rate": float(row['Overall_Delay_Rate']),
                    "average_delay_minutes": float(row['Estimated_Delay_Minutes'])
                })
            
            # Sort by date
            time_series_data.sort(key=lambda x: x['date'])
            
            return time_series_data
            
        except Exception as e:
            logger.error(f"Failed to get time series data: {str(e)}")
            raise
    
    def get_chart_data(self, chart_type: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get chart data for specific chart types.
        
        Args:
            chart_type (str): Type of chart to generate data for
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            
        Returns:
            Dict[str, Any]: Chart data
        """
        try:
            if chart_type == "delay_by_airline":
                return self._get_delay_by_airline_chart(filters)
            elif chart_type == "delay_by_route":
                return self._get_delay_by_route_chart(filters)
            elif chart_type == "delay_by_month":
                return self._get_delay_by_month_chart(filters)
            elif chart_type == "delay_by_season":
                return self._get_delay_by_season_chart(filters)
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")
                
        except Exception as e:
            logger.error(f"Failed to get chart data for {chart_type}: {str(e)}")
            raise
    
    def _get_delay_by_airline_chart(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get delay by airline chart data."""
        airline_performance = self.get_airline_performance(filters)
        
        chart_data = {
            "chart_type": "bar",
            "title": "Average Delay Rate by Airline",
            "x_axis_label": "Airline",
            "y_axis_label": "Average Delay Rate (%)",
            "data": [
                {
                    "x": item["airline"],
                    "y": item["delay_rate"] * 100,
                    "total_flights": item["total_flights"],
                    "on_time_percentage": item["on_time_percentage"]
                }
                for item in airline_performance
            ]
        }
        
        return chart_data
    
    def _get_delay_by_route_chart(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get delay by route chart data."""
        route_analytics = self.get_route_analytics(filters)
        
        # Limit to top 20 routes for better visualization
        top_routes = route_analytics[:20]
        
        chart_data = {
            "chart_type": "bar",
            "title": "Delay Distribution by Route (Top 20)",
            "x_axis_label": "Route",
            "y_axis_label": "Number of Flights",
            "data": [
                {
                    "x": item["route"],
                    "y": item["total_flights"],
                    "delay_rate": item["delay_rate"],
                    "average_delay_minutes": item["average_delay_minutes"]
                }
                for item in top_routes
            ]
        }
        
        return chart_data
    
    def _get_delay_by_month_chart(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get delay by month chart data."""
        data = self.data_service.get_flight_data(filters)
        
        if data.empty or 'Month_Num' not in data.columns:
            return {
                "chart_type": "line",
                "title": "Delay Trends by Month",
                "x_axis_label": "Month",
                "y_axis_label": "Average Delay Rate (%)",
                "data": []
            }
        
        # Group by month
        monthly_stats = data.groupby('Month_Num').agg({
            'Overall_Delay_Rate': 'mean',
            'Sectors_Scheduled': 'sum'
        }).reset_index()
        
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                      5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                      9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        chart_data = {
            "chart_type": "line",
            "title": "Delay Trends by Month",
            "x_axis_label": "Month",
            "y_axis_label": "Average Delay Rate (%)",
            "data": [
                {
                    "x": month_names.get(row['Month_Num'], f"Month {row['Month_Num']}"),
                    "y": row['Overall_Delay_Rate'] * 100,
                    "total_flights": int(row['Sectors_Scheduled'])
                }
                for _, row in monthly_stats.iterrows()
            ]
        }
        
        return chart_data
    
    def _get_delay_by_season_chart(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get delay by season chart data."""
        data = self.data_service.get_flight_data(filters)
        
        if data.empty or 'Season' not in data.columns:
            return {
                "chart_type": "pie",
                "title": "Delay Patterns by Season",
                "x_axis_label": "Season",
                "y_axis_label": "Average Delay Rate (%)",
                "data": []
            }
        
        # Group by season
        seasonal_stats = data.groupby('Season').agg({
            'Overall_Delay_Rate': 'mean',
            'Sectors_Scheduled': 'sum'
        }).reset_index()
        
        chart_data = {
            "chart_type": "pie",
            "title": "Delay Patterns by Season",
            "x_axis_label": "Season",
            "y_axis_label": "Average Delay Rate (%)",
            "data": [
                {
                    "label": row['Season'],
                    "value": row['Overall_Delay_Rate'] * 100,
                    "total_flights": int(row['Sectors_Scheduled'])
                }
                for _, row in seasonal_stats.iterrows()
            ]
        }
        
        return chart_data
    
    def get_dashboard_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get complete dashboard data including all analytics and charts.
        
        Args:
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            
        Returns:
            Dict[str, Any]: Complete dashboard data
        """
        try:
            # Validate filters
            if filters:
                validated_filters = validate_analytics_filters(filters)
            else:
                validated_filters = {}
            
            # Get all analytics data
            delay_analytics = self.get_delay_analytics(validated_filters)
            airline_performance = self.get_airline_performance(validated_filters)
            route_analytics = self.get_route_analytics(validated_filters)
            time_series_data = self.get_time_series_data(validated_filters)
            
            # Get chart data
            charts = []
            chart_types = ["delay_by_airline", "delay_by_route", "delay_by_month", "delay_by_season"]
            
            for chart_type in chart_types:
                try:
                    chart_data = self.get_chart_data(chart_type, validated_filters)
                    charts.append(chart_data)
                except Exception as e:
                    logger.warning(f"Failed to generate {chart_type} chart: {str(e)}")
            
            return {
                "delay_analytics": delay_analytics,
                "airline_performance": airline_performance,
                "route_analytics": route_analytics,
                "time_series_data": time_series_data,
                "charts": charts,
                "filters_applied": validated_filters,
                "total_records": len(self.data_service.get_flight_data()),
                "filtered_records": len(self.data_service.get_flight_data(validated_filters)),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {str(e)}")
            raise
