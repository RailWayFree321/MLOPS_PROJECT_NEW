"""
Model Training Pipeline
Trains a DistilBERT-based model for marketing creative generation
"""
import os
import logging
from pathlib import Path
import pickle
import json
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline
import mlflow
import mlflow.pytorch

from src.utils import load_config, setup_logging, get_dataset_version, get_project_root

logger = logging.getLogger(__name__)

PROJECT_ROOT = get_project_root()


class MarketingCreativeModel:
    """Model wrapper for marketing creative generation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.model_config = self.config.get("model", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_config.get("pretrained_model", "distilbert-base-uncased")
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_config.get("pretrained_model", "distilbert-base-uncased"),
            num_labels=2  # Binary classification: good/poor quality creative
        )
        self.model.to(self.device)
        
        self.label_encoder = LabelEncoder()
        self.metrics = {}
    
    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from CSV
        
        Args:
            data_path: Path to CSV file
            
        Returns:
            Tuple of features and labels
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Combine title and description for training
        X = (df["title"] + " " + df["description"]).values
        
        # Create synthetic labels based on rating (rating > 3.5 = good creative)
        y = (df["rating"] > 3.5).astype(int).values
        
        logger.info(f"Loaded {len(X)} samples with {np.sum(y)} positive samples")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Tokenize data
        train_encodings = self.tokenizer(
            list(X_train),
            truncation=True,
            padding=True,
            max_length=self.model_config.get("max_length", 128),
            return_tensors="pt"
        )
        
        val_encodings = self.tokenizer(
            list(X_val),
            truncation=True,
            padding=True,
            max_length=self.model_config.get("max_length", 128),
            return_tensors="pt"
        )
        
        # Create training dataset
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels, device):
                self.encodings = encodings
                self.labels = torch.tensor(labels)
                self.device = device
            
            def __getitem__(self, idx):
                item = {key: val[idx].to(self.device) for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx].to(self.device)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = CustomDataset(train_encodings, y_train, self.device)
        val_dataset = CustomDataset(val_encodings, y_val, self.device)
        
        # Training loop
        batch_size = int(self.model_config.get("batch_size", 16))
        epochs = int(self.model_config.get("epochs", 3))
        learning_rate = float(self.model_config.get("learning_rate", 2e-5))
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Calculate accuracy
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Final evaluation
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        final_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        self.metrics = {
            "accuracy": float(final_accuracy),
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "dataset_size": len(X),
            "train_size": len(X_train),
            "val_size": len(X_val)
        }
        
        logger.info(f"Training completed. Final Accuracy: {final_accuracy:.4f}")
        
        return self.metrics
    
    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        return self.metrics
    
    def save(self, save_path: str):
        """
        Save model to disk
        
        Args:
            save_path: Path to save the model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model_dir = Path(save_path).with_suffix("")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))
        
        # Save metrics
        metrics_file = model_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Model saved to {model_dir}")


def train_pipeline(data_path: str = None, log_run: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Complete training pipeline
    
    Args:
        data_path: Path to training data CSV
        log_run: Whether to log to MLflow
        
    Returns:
        Tuple of (model_improved, results_dict)
    """
    setup_logging()
    config = load_config()
    
    if data_path is None:
        data_path = str(PROJECT_ROOT / config.get("data", {}).get("output_path", "data/training_data.csv"))
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Initialize MLflow
    if log_run:
        mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000"))
        experiment_name = config.get("mlflow", {}).get("experiment_name", "marketing_creative_generator")
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
    
    try:
        # Initialize model
        model = MarketingCreativeModel(config)
        
        # Prepare data
        X, y = model.prepare_data(data_path)
        
        # Train model
        metrics = model.train(X, y)
        
        # Get dataset version
        dataset_version = get_dataset_version()
        
        results = {
            "status": "success",
            "model_accuracy": metrics["accuracy"],
            "dataset_version": dataset_version,
            "metrics": metrics,
            "model_improved": True
        }
        
        # Log to MLflow
        if log_run:
            try:
                with mlflow.start_run():
                    mlflow.log_params({
                        "model_type": config.get("model", {}).get("model_type", "distilbert"),
                        "epochs": config.get("model", {}).get("epochs", 3),
                        "batch_size": config.get("model", {}).get("batch_size", 16),
                        "learning_rate": config.get("model", {}).get("learning_rate", 2e-5),
                        "dataset_version": dataset_version
                    })
                    
                    mlflow.log_metrics({
                        "accuracy": metrics["accuracy"],
                        "train_loss": metrics["train_loss"],
                        "val_loss": metrics["val_loss"],
                        "dataset_size": metrics["dataset_size"]
                    })
                    
                    # Log model
                    mlflow.pytorch.log_model(model.model, "model")
                    
                    logger.info(f"Logged run to MLflow with accuracy: {metrics['accuracy']:.4f}")
            except Exception as e:
                logger.warning(f"Could not log to MLflow: {e}")
        
        # Save model locally
        model_save_path = str(PROJECT_ROOT / config.get("model_registry", {}).get("local_model_path", "models/best_model"))
        model.save(model_save_path)
        
        logger.info("=" * 80)
        logger.info(f"TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Model Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Dataset Version: {dataset_version}")
        logger.info("=" * 80)
        
        return True, results
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        return False, {"status": "error", "error": str(e)}


if __name__ == "__main__":
    success, results = train_pipeline()
    print(json.dumps(results, indent=2))
