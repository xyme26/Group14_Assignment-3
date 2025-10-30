"""
Startup script for the Flight Delay Prediction API.
This script provides an easy way to start the API with proper configuration.
"""

import uvicorn
import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files are present."""
    base_path = Path(__file__).parent
    
    # Check model files
    models_path = base_path / "models"
    required_models = [
        "trained_models.joblib",
        "feature_scaler.joblib", 
        "feature_encoders.joblib",
        "training_metadata.joblib"
    ]
    
    missing_models = []
    for model_file in required_models:
        if not (models_path / model_file).exists():
            missing_models.append(model_file)
    
    if missing_models:
        print("❌ Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nPlease ensure all model files are present in the models/ directory.")
        return False
    
    # Check data files
    data_path = base_path / "data"
    required_data = [
        "processed_flight_data.csv",
        "feature_matrix.csv",
        "classification_target.csv", 
        "regression_target.csv"
    ]
    
    missing_data = []
    for data_file in required_data:
        if not (data_path / data_file).exists():
            missing_data.append(data_file)
    
    if missing_data:
        print("❌ Missing data files:")
        for data in missing_data:
            print(f"   - {data}")
        print("\nPlease ensure all data files are present in the data/ directory.")
        return False
    
    print("✅ All required files are present")
    return True

def main():
    """Main startup function."""
    print("🚀 Starting Flight Delay Prediction API")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Startup failed due to missing files")
        sys.exit(1)
    
    # Get configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"📡 Starting server on {host}:{port}")
    print(f"📊 Log level: {log_level}")
    print(f"🔄 Auto-reload: {reload}")
    print("\n🌐 API will be available at:")
    print(f"   - API Base URL: http://localhost:{port}")
    print(f"   - Interactive Docs: http://localhost:{port}/docs")
    print(f"   - ReDoc: http://localhost:{port}/redoc")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
