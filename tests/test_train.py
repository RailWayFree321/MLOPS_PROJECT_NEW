"""
Tests for model training module
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.train import MarketingCreativeModel, train_pipeline
from src.utils import load_config, get_project_root


class TestMarketingCreativeModel:
    """Test model training functionality"""
    
    @pytest.fixture
    def sample_data_csv(self):
        """Create sample training data CSV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write("id,title,category,description,price,rating,created_at\n")
            
            # Write sample data
            for i in range(20):
                f.write(f'{i},Product {i},Electronics,Description {i},{10+i},{2+i%4},2024-01-01T00:00:00\n')
            
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_model_initialization(self):
        """Test model initialization"""
        config = load_config()
        model = MarketingCreativeModel(config)
        
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device is not None
    
    def test_prepare_data(self, sample_data_csv):
        """Test data preparation"""
        config = load_config()
        model = MarketingCreativeModel(config)
        
        X, y = model.prepare_data(sample_data_csv)
        
        # Check data shape
        assert len(X) == 20
        assert len(y) == 20
        
        # Check data types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # Check labels are binary
        assert set(np.unique(y)) == {0, 1} or set(np.unique(y)) == {0} or set(np.unique(y)) == {1}
    
    def test_prepare_data_combines_fields(self, sample_data_csv):
        """Test that prepare_data combines title and description"""
        config = load_config()
        model = MarketingCreativeModel(config)
        
        X, y = model.prepare_data(sample_data_csv)
        
        # Check that combined text contains both title and description
        for text in X:
            assert isinstance(text, str)
            assert len(text) > 0
            # Should have space between title and description
            assert "Product" in text and "Description" in text
    
    def test_metrics_initialization(self):
        """Test that metrics are initialized"""
        config = load_config()
        model = MarketingCreativeModel(config)
        
        metrics = model.get_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # Should be empty before training
    
    def test_model_save(self, sample_data_csv):
        """Test model saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config()
            model = MarketingCreativeModel(config)
            
            # Prepare and train (minimal)
            X, y = model.prepare_data(sample_data_csv)
            
            # Set dummy metrics for testing save
            model.metrics = {
                "accuracy": 0.85,
                "train_loss": 0.15,
                "val_loss": 0.20,
                "dataset_size": len(X),
                "train_size": int(0.8 * len(X)),
                "val_size": int(0.2 * len(X))
            }
            
            save_path = Path(tmpdir) / "model.pkl"
            model.save(str(save_path))
            
            # Check that model files were created
            model_dir = Path(tmpdir) / "model"
            assert model_dir.exists()
            
            # Check for metrics file
            metrics_file = model_dir / "metrics.json"
            assert metrics_file.exists()
            
            # Verify metrics content
            with open(metrics_file) as f:
                saved_metrics = json.load(f)
            assert saved_metrics["accuracy"] == 0.85


class TestTrainingPipeline:
    """Test the complete training pipeline"""
    
    @pytest.fixture
    def sample_data_csv(self):
        """Create sample training data CSV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write("id,title,category,description,price,rating,created_at\n")
            
            # Write sample data with variety of ratings
            for i in range(30):
                rating = 2 + (i % 4)
                f.write(f'{i},Product {i},Electronics,Description {i},{10+i},{rating},2024-01-01T00:00:00\n')
            
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_training_pipeline_structure(self, sample_data_csv):
        """Test that training pipeline returns expected structure"""
        success, results = train_pipeline(data_path=sample_data_csv, log_run=False)
        
        # Check return values
        assert isinstance(success, bool)
        assert isinstance(results, dict)
        
        # Check results structure
        assert "status" in results
        assert "model_accuracy" in results
    
    def test_training_pipeline_success(self, sample_data_csv):
        """Test successful training pipeline execution"""
        success, results = train_pipeline(data_path=sample_data_csv, log_run=False)
        
        assert success is True
        assert results["status"] == "success"
        
        # Check metrics
        metrics = results.get("metrics", {})
        assert "accuracy" in metrics
        assert "dataset_size" in metrics
        assert metrics["dataset_size"] == 30
    
    def test_training_pipeline_accuracy_in_range(self, sample_data_csv):
        """Test that accuracy is in valid range"""
        success, results = train_pipeline(data_path=sample_data_csv, log_run=False)
        
        assert success is True
        accuracy = results.get("model_accuracy", 0)
        
        # Accuracy should be between 0 and 1
        assert 0 <= accuracy <= 1


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
