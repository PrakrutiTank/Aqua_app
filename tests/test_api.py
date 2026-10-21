
import pytest

from app.app import app


@pytest.fixture
def client():
    """Create a Flask test client for each test."""
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_home_endpoint(client):
    """The homepage should display the AquaGuard frontend."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"AquaGuard" in response.data
    assert "text/html" in response.content_type


def test_health_endpoint(client):
    """The health endpoint should report that the API is healthy."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_features_endpoint(client):
    """The API should return the required model features."""
    response = client.get("/features")

    assert response.status_code == 200

    data = response.get_json()
    assert "required_features" in data
    assert len(data["required_features"]) > 0


def test_missing_features_returns_400(client):
    """An empty prediction request should be rejected."""
    response = client.post("/predict", json={})

    assert response.status_code == 400


def test_invalid_json_type_returns_400(client):
    """A JSON list should be rejected because an object is required."""
    response = client.post("/predict", json=[1, 2, 3])

    assert response.status_code == 400


def test_prediction_with_all_features(client):
    """The API should return a numeric prediction for valid inputs."""
    response = client.get("/features")
    assert response.status_code == 200

    names = response.get_json()["required_features"]
    assert names

    payload = {name: 1.0 for name in names}

    prediction_response = client.post("/predict", json=payload)

    assert prediction_response.status_code == 200

    result = prediction_response.get_json()
    assert "prediction" in result
    assert isinstance(result["prediction"], (int, float))
