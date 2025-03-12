import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing"""
    with patch.dict(os.environ, {
        'MODEL': 'test-model',
        'TEMPERATURE': '0.7',
    }):
        yield

def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
