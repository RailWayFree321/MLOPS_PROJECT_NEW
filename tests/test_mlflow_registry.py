"""
Tests for MLflow registry integration
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.mlflow_registry import MLflowRegistry, check_and_register_model
from src.utils import load_config


class TestMLflowRegistry:
    """Test MLflow registry operations"""
    
    def test_registry_initialization(self):
        """Test MLflow registry initialization"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient"):
                registry = MLflowRegistry(config)
                
                assert registry.config is not None
                assert registry.model_name is not None
    
    def test_get_best_model_accuracy_no_models(self):
        """Test getting best model accuracy when no models exist"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                mock_instance.get_latest_model_version.side_effect = Exception("No models")
                
                registry = MLflowRegistry(config)
                accuracy = registry.get_best_model_accuracy()
                
                assert accuracy is None
    
    def test_register_model_no_previous(self):
        """Test registering model when no previous models exist"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                # Mock no previous model
                with patch.object(MLflowRegistry, "get_best_model_accuracy", return_value=None):
                    registry = MLflowRegistry(config)
                    should_register, message = registry.register_model(
                        "runs://test/model",
                        0.85,
                        1
                    )
                    
                    assert should_register is True
                    assert "No previous model" in message
    
    def test_register_model_better(self):
        """Test registering model that is better than previous"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                # Mock better model
                with patch.object(MLflowRegistry, "get_best_model_accuracy", return_value=0.80):
                    registry = MLflowRegistry(config)
                    should_register, message = registry.register_model(
                        "runs://test/model",
                        0.85,  # Current accuracy is higher
                        1
                    )
                    
                    assert should_register is True
                    assert "improved" in message.lower()
    
    def test_register_model_worse(self):
        """Test registering model that is worse than previous"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                # Mock worse model
                with patch.object(MLflowRegistry, "get_best_model_accuracy", return_value=0.90):
                    registry = MLflowRegistry(config)
                    should_register, message = registry.register_model(
                        "runs://test/model",
                        0.80,  # Current accuracy is lower
                        1
                    )
                    
                    assert should_register is False
                    assert "did not improve" in message.lower()
    
    def test_compare_with_previous(self):
        """Test comparing current model with previous versions"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                with patch.object(MLflowRegistry, "get_all_model_accuracies", return_value={"1": 0.75, "2": 0.80}):
                    with patch.object(MLflowRegistry, "get_best_model_accuracy", return_value=0.80):
                        registry = MLflowRegistry(config)
                        comparison = registry.compare_with_previous(0.85)
                        
                        assert comparison["current_accuracy"] == 0.85
                        assert comparison["best_accuracy"] == 0.80
                        assert comparison["is_improved"] is True
                        assert comparison["num_previous_versions"] == 2


class TestCheckAndRegisterModel:
    """Test the check and register model function"""
    
    def test_check_and_register_model(self):
        """Test checking and registering model"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                with patch("src.mlflow_registry.mlflow.active_run", return_value=None):
                    with patch.object(MLflowRegistry, "get_all_model_accuracies", return_value={}):
                        with patch.object(MLflowRegistry, "get_best_model_accuracy", return_value=None):
                            with patch.object(MLflowRegistry, "register_model", return_value=(True, "Test message")):
                                results = check_and_register_model(
                                    "models/best_model",
                                    0.85,
                                    1,
                                    config
                                )
                                
                                assert results["current_accuracy"] == 0.85
                                assert results["dataset_version"] == 1
                                assert "model_path" in results
                                assert "comparison" in results


class TestModelComparison:
    """Test model comparison logic"""
    
    def test_improved_model_detection(self):
        """Test detection of improved model"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                with patch.object(MLflowRegistry, "get_best_model_accuracy", return_value=0.80):
                    with patch.object(MLflowRegistry, "get_all_model_accuracies", return_value={"1": 0.75, "2": 0.80}):
                        registry = MLflowRegistry(config)
                        comparison = registry.compare_with_previous(0.90)
                        
                        assert comparison["is_improved"] is True
    
    def test_non_improved_model_detection(self):
        """Test detection of non-improved model"""
        config = load_config()
        
        with patch("src.mlflow_registry.mlflow.set_tracking_uri"):
            with patch("src.mlflow_registry.MlflowClient") as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                with patch.object(MLflowRegistry, "get_best_model_accuracy", return_value=0.90):
                    with patch.object(MLflowRegistry, "get_all_model_accuracies", return_value={"1": 0.85, "2": 0.90}):
                        registry = MLflowRegistry(config)
                        comparison = registry.compare_with_previous(0.85)
                        
                        assert comparison["is_improved"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
