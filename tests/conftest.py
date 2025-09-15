"""
Test configuration and fixtures for BreadthFlow test suite.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fastapi_app.core.database import get_db
from fastapi_app.main import app
from fastapi_app.models import Base
from tests.fixtures.test_data import TestDataFactory

# Test database configuration
SQLALCHEMY_DATABASE_URL = "postgresql://test_user:test_password@localhost:5433/breadthflow_test"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override the database dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def setup_test_db():
    """Set up test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def db_session():
    """Create test database session"""
    session = TestingSessionLocal()
    yield session
    session.close()


@pytest.fixture
def test_data_factory():
    """Create test data factory"""
    return TestDataFactory()


@pytest.fixture
def sample_ohlcv_data(test_data_factory):
    """Create sample OHLCV data for testing"""
    return test_data_factory.create_ohlcv_data()


@pytest.fixture
def sample_signal_data(test_data_factory):
    """Create sample signal data for testing"""
    return test_data_factory.create_signal_data()


@pytest.fixture
def sample_user_data(test_data_factory):
    """Create sample user data for testing"""
    return test_data_factory.create_user_data()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
