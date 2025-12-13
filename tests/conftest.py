"""
Conftest file for pytest fixtures and configuration
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import pytest


@pytest.fixture(scope="session")
def project_root():
    """Provide project root path"""
    return PROJECT_ROOT
