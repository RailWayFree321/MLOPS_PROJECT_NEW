"""
Tests for utility functions
"""
import pytest
import yaml
from pathlib import Path
import tempfile
import logging

from src.utils import (
    load_config,
    setup_logging,
    get_dataset_version,
    increment_dataset_version,
    get_project_root
)


class TestConfigLoading:
    """Test configuration loading"""
    
    def test_load_config_default(self):
        """Test loading default configuration"""
        config = load_config()
        
        assert isinstance(config, dict)
        assert "model" in config
        assert "mlflow" in config
        assert "data" in config
    
    def test_load_config_structure(self):
        """Test configuration has expected structure"""
        config = load_config()
        
        # Check main sections
        assert "model" in config
        assert "mlflow" in config
        assert "data" in config
        assert "model_registry" in config
        
        # Check model config
        model_config = config["model"]
        assert "model_type" in model_config
        assert "batch_size" in model_config
        assert "epochs" in model_config
        
        # Check mlflow config
        mlflow_config = config["mlflow"]
        assert "tracking_uri" in mlflow_config
        assert "experiment_name" in mlflow_config
    
    def test_load_config_custom_file(self):
        """Test loading custom configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"test_key": "test_value"}, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config["test_key"] == "test_value"
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_config_file_not_found(self):
        """Test error handling for missing config file"""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestVersionManagement:
    """Test dataset version management"""
    
    def test_get_dataset_version_new(self):
        """Test getting version for new project"""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = Path(tmpdir) / "version.txt"
            
            # First call should initialize to 1
            version = get_dataset_version(str(version_file))
            assert version == 1
            
            # File should be created
            assert version_file.exists()
            assert version_file.read_text().strip() == "1"
    
    def test_get_dataset_version_existing(self):
        """Test getting existing version"""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = Path(tmpdir) / "version.txt"
            version_file.write_text("5")
            
            version = get_dataset_version(str(version_file))
            assert version == 5
    
    def test_increment_dataset_version(self):
        """Test incrementing dataset version"""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = Path(tmpdir) / "version.txt"
            version_file.write_text("3")
            
            new_version = increment_dataset_version(str(version_file))
            
            assert new_version == 4
            assert version_file.read_text().strip() == "4"
    
    def test_increment_dataset_version_sequential(self):
        """Test sequential increments"""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = Path(tmpdir) / "version.txt"
            version_file.write_text("1")
            
            v1 = increment_dataset_version(str(version_file))
            v2 = increment_dataset_version(str(version_file))
            v3 = increment_dataset_version(str(version_file))
            
            assert v1 == 2
            assert v2 == 3
            assert v3 == 4
            assert version_file.read_text().strip() == "4"


class TestLogging:
    """Test logging setup"""
    
    def test_setup_logging_creates_directory(self):
        """Test that logging setup creates log directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs" / "test.log"
            
            setup_logging(str(log_file), level="INFO")
            
            assert log_file.parent.exists()
    
    def test_setup_logging_level(self):
        """Test logging level configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            # Should not raise exception
            setup_logging(str(log_file), level="DEBUG")
            setup_logging(str(log_file), level="INFO")
            setup_logging(str(log_file), level="WARNING")


class TestProjectRoot:
    """Test project root detection"""
    
    def test_get_project_root(self):
        """Test getting project root"""
        root = get_project_root()
        
        assert isinstance(root, Path)
        assert root.exists()
        # Project root should contain src directory
        assert (root / "src").exists() or root.name == "Proj"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
