"""
Tests for model training module
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.utils import load_config, get_project_root


class TestDataPreparation:
    """Test data preparation functions"""
    
    def test_csv_loading(self):
        """Test CSV loading functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,title,category,description,price,rating,created_at\n")
            f.write("1,Test Product,Electronics,Test Description,100,4.5,2024-01-01\n")
            f.write("2,Another Product,Fashion,Another Description,50,3.8,2024-01-02\n")
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path)
            
            assert len(df) == 2
            assert list(df.columns) == ["id", "title", "category", "description", "price", "rating", "created_at"]
            assert df["id"].tolist() == [1, 2]
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_csv_data_types(self):
        """Test that CSV data has correct types"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,title,category,description,price,rating,created_at\n")
            f.write("1,Product A,Electronics,Desc A,99.99,4.5,2024-01-01\n")
            f.write("2,Product B,Fashion,Desc B,49.99,3.8,2024-01-02\n")
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path)
            
            # Check that numeric columns are numeric
            assert df["price"].dtype in [np.float64, float]
            assert df["rating"].dtype in [np.float64, float]
            assert df["id"].dtype in [np.int64, int]
            
            # Check string columns
            assert df["title"].dtype == object
            assert df["description"].dtype == object
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_combined_text_field(self):
        """Test combining title and description"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,title,category,description,price,rating,created_at\n")
            f.write("1,Premium Headphones,Electronics,High quality audio,99.99,4.5,2024-01-01\n")
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path)
            
            # Simulate data preparation
            combined_text = (df["title"] + " " + df["description"]).values
            
            assert len(combined_text) == 1
            assert "Premium Headphones" in combined_text[0]
            assert "High quality audio" in combined_text[0]
            assert combined_text[0].count(" ") >= 4  # At least 4 spaces
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_label_creation_from_rating(self):
        """Test binary label creation from ratings"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,title,category,description,price,rating,created_at\n")
            f.write("1,Product A,Electronics,Desc A,99.99,4.5,2024-01-01\n")  # > 3.5 -> 1
            f.write("2,Product B,Fashion,Desc B,49.99,3.0,2024-01-02\n")      # <= 3.5 -> 0
            f.write("3,Product C,Home,Desc C,29.99,3.5,2024-01-03\n")         # <= 3.5 -> 0
            f.write("4,Product D,Sports,Desc D,19.99,4.0,2024-01-04\n")       # > 3.5 -> 1
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path)
            
            # Simulate label creation
            y = (df["rating"] > 3.5).astype(int).values
            
            assert y[0] == 1  # 4.5 > 3.5
            assert y[1] == 0  # 3.0 <= 3.5
            assert y[2] == 0  # 3.5 <= 3.5
            assert y[3] == 1  # 4.0 > 3.5
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_data_split_logic(self):
        """Test train/val split logic"""
        # Create sample data
        X = np.array([f"text_{i}" for i in range(100)])
        y = np.array([i % 2 for i in range(100)])
        
        # Simulate 80/20 split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        assert len(X_train) == 80
        assert len(X_val) == 20
        assert len(y_train) == 80
        assert len(y_val) == 20


class TestConfigLoading:
    """Test configuration loading"""
    
    def test_config_loads_successfully(self):
        """Test that config file loads without error"""
        config = load_config()
        
        assert config is not None
        assert isinstance(config, dict)
    
    def test_config_has_required_sections(self):
        """Test that config has required sections"""
        config = load_config()
        
        required_keys = ["project_name", "data", "model", "mlflow"]
        for key in required_keys:
            assert key in config, f"Missing key: {key}"
    
    def test_model_config_parameters(self):
        """Test model config has required parameters"""
        config = load_config()
        model_config = config.get("model", {})
        
        assert "learning_rate" in model_config
        assert "batch_size" in model_config
        assert "epochs" in model_config
        assert isinstance(model_config["learning_rate"], (int, float))
        assert isinstance(model_config["batch_size"], int)
        assert isinstance(model_config["epochs"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
