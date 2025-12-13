"""
MLflow Model Registry Integration
Handles model registration and comparison with previous versions
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from src.utils import load_config, get_project_root

logger = logging.getLogger(__name__)

PROJECT_ROOT = get_project_root()


class MLflowRegistry:
    """Manages MLflow model registry operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MLflow registry
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.mlflow_config = self.config.get("mlflow", {})
        self.registry_config = self.config.get("model_registry", {})
        
        tracking_uri = self.mlflow_config.get("tracking_uri", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient(tracking_uri)
        self.model_name = self.registry_config.get("model_name", "marketing_creative_model")
        
        logger.info(f"MLflow Registry initialized with tracking URI: {tracking_uri}")
    
    def get_best_model_accuracy(self) -> Optional[float]:
        """
        Get the accuracy of the best registered model
        
        Returns:
            Best model accuracy or None if no model is registered
        """
        try:
            latest_version = self.client.get_latest_model_version(self.model_name)
            
            # Get run information
            run = mlflow.get_run(latest_version.run_id)
            accuracy = run.data.metrics.get("accuracy")
            
            logger.info(f"Best model accuracy: {accuracy}")
            return accuracy
        
        except Exception as e:
            logger.info(f"No registered model found: {str(e)}")
            return None
    
    def get_all_model_accuracies(self) -> Dict[str, float]:
        """
        Get accuracies of all registered models
        
        Returns:
            Dictionary mapping model version to accuracy
        """
        accuracies = {}
        
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            for version in versions:
                try:
                    run = mlflow.get_run(version.run_id)
                    accuracy = run.data.metrics.get("accuracy", 0)
                    accuracies[version.version] = accuracy
                except Exception as e:
                    logger.warning(f"Could not get accuracy for version {version.version}: {e}")
            
            logger.info(f"Retrieved accuracies for {len(accuracies)} models")
        
        except Exception as e:
            logger.warning(f"Could not retrieve model versions: {e}")
        
        return accuracies
    
    def register_model(
        self,
        model_uri: str,
        current_accuracy: float,
        dataset_version: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Register a new model if it's better than the current best
        
        Args:
            model_uri: MLflow model URI
            current_accuracy: Current model accuracy
            dataset_version: Dataset version used for training
            
        Returns:
            Tuple of (should_register, message)
        """
        best_accuracy = self.get_best_model_accuracy()
        
        if best_accuracy is None:
            logger.info("No previous model found. Registering this model.")
            return True, "No previous model to compare"
        
        if current_accuracy > best_accuracy:
            logger.info(f"Model improved: {best_accuracy:.4f} -> {current_accuracy:.4f}")
            return True, f"Model improved from {best_accuracy:.4f} to {current_accuracy:.4f}"
        else:
            logger.info(f"Model did not improve: current={current_accuracy:.4f}, best={best_accuracy:.4f}")
            return False, f"Model did not improve (current={current_accuracy:.4f}, best={best_accuracy:.4f})"
    
    def create_new_version(
        self,
        model_uri: str,
        version_description: str = None
    ) -> Optional[str]:
        """
        Create a new model version in the registry
        
        Args:
            model_uri: MLflow model URI
            version_description: Description for the model version
            
        Returns:
            Registered model version or None if failed
        """
        try:
            result = mlflow.register_model(model_uri, self.model_name)
            logger.info(f"Registered model version: {result.version}")
            return result.version
        
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            return None
    
    def get_model_path(self, version: str = None) -> Optional[str]:
        """
        Get the path to a specific model version
        
        Args:
            version: Model version (latest if None)
            
        Returns:
            Model path or None if not found
        """
        try:
            if version is None:
                model_version = self.client.get_latest_model_version(self.model_name)
            else:
                model_version = self.client.get_model_version(self.model_name, version)
            
            return model_version.source
        
        except Exception as e:
            logger.error(f"Could not get model path: {str(e)}")
            return None
    
    def compare_with_previous(self, current_accuracy: float) -> Dict[str, Any]:
        """
        Compare current model with all previous versions
        
        Args:
            current_accuracy: Current model accuracy
            
        Returns:
            Comparison results dictionary
        """
        all_accuracies = self.get_all_model_accuracies()
        best_accuracy = self.get_best_model_accuracy()
        
        comparison = {
            "current_accuracy": current_accuracy,
            "best_accuracy": best_accuracy,
            "is_improved": current_accuracy > best_accuracy if best_accuracy else True,
            "all_versions": all_accuracies,
            "num_previous_versions": len(all_accuracies)
        }
        
        logger.info(f"Comparison results: {json.dumps(comparison, indent=2)}")
        
        return comparison


def check_and_register_model(
    model_path: str,
    current_accuracy: float,
    dataset_version: int,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Check if model is better and register if needed
    
    Args:
        model_path: Path to the trained model
        current_accuracy: Current model accuracy
        dataset_version: Dataset version used
        config: Configuration dictionary
        
    Returns:
        Dictionary with registration results
    """
    registry = MLflowRegistry(config)
    
    # Compare with previous models
    comparison = registry.compare_with_previous(current_accuracy)
    
    results = {
        "model_path": model_path,
        "current_accuracy": current_accuracy,
        "dataset_version": dataset_version,
        "comparison": comparison,
        "registered": False,
        "message": ""
    }
    
    # Check if model should be registered
    should_register, message = registry.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model" if mlflow.active_run() else "",
        current_accuracy,
        dataset_version
    )
    
    results["message"] = message
    results["registered"] = should_register
    
    logger.info(f"Registration decision: {should_register} - {message}")
    
    return results
