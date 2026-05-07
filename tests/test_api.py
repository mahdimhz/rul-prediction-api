import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.model import feature_cols


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_predict_zeros() -> None:
    payload = {feature_name: 0.0 for feature_name in feature_cols}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/predict", json=payload)

    response_json = response.json()
    assert response.status_code == 200
    assert response_json["predicted_rul_seconds"] >= 0.0
    assert response_json["n_features_received"] == 44


@pytest.mark.anyio
async def test_predict_missing_field() -> None:
    payload = {feature_name: 0.0 for feature_name in feature_cols[:10]}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_wrong_type() -> None:
    payload = {feature_name: 0.0 for feature_name in feature_cols}
    payload[feature_cols[0]] = "not_a_number"

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 422
