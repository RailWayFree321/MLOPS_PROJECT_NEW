"""
Utility functions for the MLOps project
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file (default: config/config.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def setup_logging(log_file: str = None, level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    config = load_config()
    
    if log_file is None:
        log_file = config.get("logging", {}).get("log_file", "logs/pipeline.log")
    
    log_format = config.get("logging", {}).get(
        "format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured with level {level}")


def get_dataset_version(version_file: str = None) -> int:
    """
    Get current dataset version
    
    Args:
        version_file: Path to version file
        
    Returns:
        Current dataset version
    """
    if version_file is None:
        config = load_config()
        version_file = config.get("data", {}).get("version_file", "data/dataset_version.txt")
    
    version_file_path = PROJECT_ROOT / version_file
    
    if version_file_path.exists():
        try:
            with open(version_file_path, "r") as f:
                version = int(f.read().strip())
            return version
        except (ValueError, IOError):
            logger.warning("Could not read version file, initializing to 1")
            return 1
    else:
        # Initialize version file
        version_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(version_file_path, "w") as f:
            f.write("1")
        return 1


def increment_dataset_version(version_file: str = None) -> int:
    """
    Increment and save dataset version
    
    Args:
        version_file: Path to version file
        
    Returns:
        New dataset version
    """
    if version_file is None:
        config = load_config()
        version_file = config.get("data", {}).get("version_file", "data/dataset_version.txt")
    
    current_version = get_dataset_version(version_file)
    new_version = current_version + 1
    
    version_file_path = PROJECT_ROOT / version_file
    version_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(version_file_path, "w") as f:
        f.write(str(new_version))
    
    logger.info(f"Dataset version incremented: {current_version} -> {new_version}")
    return new_version


def get_project_root() -> Path:
    """Get project root directory"""
    return PROJECT_ROOT
