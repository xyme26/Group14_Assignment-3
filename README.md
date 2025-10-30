# Flight Delay Prediction API

A comprehensive FastAPI backend for predicting flight delays and analyzing aviation data. This system integrates machine learning models from Assignment 2 to provide real-time predictions and analytics for flight delay risk assessment and delay duration estimation.

## 🚀 Features

- **Real-time Predictions**: Classification and regression models for delay risk and duration
- **Comprehensive Analytics**: Dashboard data with multiple chart types and visualizations
- **Reference Data**: Complete airline and airport information for Australian aviation
- **Data Export**: Export analytics data in CSV, JSON, and Excel formats
- **RESTful API**: Well-documented endpoints with proper error handling
- **Input Validation**: Robust validation for all input parameters
- **Health Monitoring**: Health check endpoints for service monitoring

## 📋 Requirements

- Python 3.8+
- FastAPI 0.104.1+
- scikit-learn 1.3.2+
- pandas 2.1.3+
- numpy 1.24.3+

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**:
   Make sure the following files exist in the `models/` directory:
   - `trained_models.joblib`
   - `feature_scaler.joblib`
   - `feature_encoders.joblib`
   - `training_metadata.joblib`

5. **Ensure data files are present**:
   Make sure the following files exist in the `data/` directory:
   - `processed_flight_data.csv`
   - `feature_matrix.csv`
   - `classification_target.csv`
   - `regression_target.csv`

## 🚀 Running the Application

### Development Mode

```bash
python main.py
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📚 API Documentation

### Base Endpoints

- `GET /` - API information and health status
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Prediction Endpoints

#### `POST /api/v1/predictions/delay-risk`
Predict delay risk category (Low, Medium, High) for a flight.

**Request Body:**
```json
{
  "route": "Sydney-Melbourne",
  "departing_port": "Sydney",
  "arriving_port": "Melbourne",
  "airline": "Qantas",
  "month": 6,
  "year": 2024,
  "sectors_scheduled": 10
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "classification": {
      "predicted_category": "Low",
      "confidence_scores": {
        "Low": 0.8,
        "Medium": 0.15,
        "High": 0.05
      },
      "probability": 0.8
    }
  },
  "processing_time_ms": 45.2
}
```

#### `POST /api/v1/predictions/delay-duration`
Predict delay duration in minutes for a flight.

**Request Body:** Same as delay-risk endpoint

**Response:**
```json
{
  "success": true,
  "predictions": {
    "regression": {
      "predicted_delay_minutes": 15.5,
      "confidence_interval": {
        "lower": 10.2,
        "upper": 20.8
      }
    }
  },
  "processing_time_ms": 52.1
}
```

#### `POST /api/v1/predictions/both`
Make both classification and regression predictions in a single call.

#### `POST /api/v1/predictions/batch`
Make predictions for multiple flights in batch (1-100 flights).

### Analytics Endpoints

#### `GET /api/v1/analytics/dashboard`
Get complete dashboard data including all analytics and charts.

**Query Parameters:**
- `start_date` (optional): Start date filter (YYYY-MM-DD)
- `end_date` (optional): End date filter (YYYY-MM-DD)
- `airline` (optional): Filter by specific airline
- `airport` (optional): Filter by specific airport
- `delay_risk_category` (optional): Filter by delay risk category
- `route` (optional): Filter by specific route
- `limit` (optional): Maximum records to return (default: 1000)

#### `GET /api/v1/analytics/delay-analytics`
Get delay analytics data including statistics and distributions.

#### `GET /api/v1/analytics/airline-performance`
Get airline performance metrics and rankings.

#### `GET /api/v1/analytics/route-analytics`
Get route performance metrics and traffic patterns.

#### `GET /api/v1/analytics/time-series`
Get time series data for trend analysis.

#### `GET /api/v1/analytics/charts/{chart_type}`
Get chart data for specific chart types:
- `delay_by_airline` - Bar chart of delay rates by airline
- `delay_by_route` - Bar chart of delay distribution by route
- `delay_by_month` - Line chart of delay trends by month
- `delay_by_season` - Pie chart of delay patterns by season

#### `POST /api/v1/analytics/export`
Export analytics data in various formats (CSV, JSON, Excel).

### Reference Data Endpoints

#### `GET /api/v1/airlines/airlines`
Get list of Australian airlines with their IATA codes.

#### `GET /api/v1/airlines/airports`
Get list of Australian airports with their IATA codes.

#### `GET /api/v1/airlines/delay-risk-categories`
Get list of delay risk categories.

#### `GET /api/v1/airlines/seasons`
Get list of seasons.

#### `GET /api/v1/airlines/months`
Get list of months with numbers and names.

#### `GET /api/v1/airlines/years`
Get list of supported years (2020-2030).

#### `GET /api/v1/airlines/reference-data`
Get all reference data in a single response.

#### `GET /api/v1/airlines/search`
Search reference data by query string.

## 🔧 Configuration

### Environment Variables

The API can be configured using environment variables:

- `API_HOST`: Host address (default: 0.0.0.0)
- `API_PORT`: Port number (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)

### Model Configuration

The API automatically loads models from the `models/` directory. Ensure the following files are present:

- `trained_models.joblib`: Contains both classification and regression models
- `feature_scaler.joblib`: Feature scaling object
- `feature_encoders.joblib`: Categorical feature encoders
- `training_metadata.joblib`: Training metadata and feature information

### Data Configuration

The API loads data from the `data/` directory. Ensure the following files are present:

- `processed_flight_data.csv`: Processed flight data
- `feature_matrix.csv`: Feature matrix for ML models
- `classification_target.csv`: Classification target variable
- `regression_target.csv`: Regression target variable

## 🧪 Testing

### Health Check

Test if the API is running:

```bash
curl http://localhost:8000/health
```

### Test Prediction

Test a prediction endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/predictions/delay-risk" \
     -H "Content-Type: application/json" \
     -d '{
       "route": "Sydney-Melbourne",
       "departing_port": "Sydney",
       "arriving_port": "Melbourne",
       "airline": "Qantas",
       "month": 6,
       "sectors_scheduled": 10
     }'
```

### Test Analytics

Test analytics endpoint:

```bash
curl "http://localhost:8000/api/v1/analytics/dashboard?limit=100"
```

## 📊 Data Flow

1. **Input Validation**: All inputs are validated using Pydantic schemas
2. **Feature Engineering**: Input data is processed and features are engineered
3. **Model Prediction**: Preprocessed data is fed to trained ML models
4. **Response Formatting**: Results are formatted and returned as JSON

## 🔍 Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid input data or validation errors
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation errors in request body
- **500 Internal Server Error**: Server-side errors
- **503 Service Unavailable**: Service not ready or models not loaded

## 📈 Performance

- **Prediction Latency**: Typically < 100ms per prediction
- **Batch Processing**: Supports up to 100 predictions per batch
- **Concurrent Requests**: Handles multiple concurrent requests
- **Memory Usage**: Optimized for efficient memory usage

## 🔒 Security

- **Input Validation**: All inputs are validated and sanitized
- **CORS Configuration**: Configurable CORS settings
- **Error Messages**: Sanitized error messages to prevent information leakage

## 🚀 Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```

### Production Considerations

- Use a production ASGI server like Gunicorn with Uvicorn workers
- Implement proper logging and monitoring
- Use environment variables for configuration
- Set up health checks and load balancing
- Consider using a reverse proxy like Nginx

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of COS30049 Assignment 3 - Full-Stack Web Development for AI Application.

## 🆘 Support

For issues and questions:

1. Check the API documentation at `/docs`
2. Review the logs for error details
3. Ensure all required model and data files are present
4. Verify the Python environment and dependencies

## 📝 Changelog

### Version 1.0.0
- Initial release
- Complete prediction API
- Analytics and visualization endpoints
- Reference data endpoints
- Comprehensive documentation
