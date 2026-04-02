import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.database import get_db, Base
from app.core.config import get_settings

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(scope="module")
def test_client():
    """Fixture for test client."""
    Base.metadata.create_all(bind=engine)
    yield client
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user():
    """Fixture for test user."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }


@pytest.fixture
def test_document():
    """Fixture for test document."""
    return {
        "title": "Test Document",
        "content": "This is a test document for RAG.",
        "source": "test_source",
        "metadata_json": '{"key": "value"}'
    }
