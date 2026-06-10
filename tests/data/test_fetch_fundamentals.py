import pandas as pd
import pytest
from unittest.mock import patch
import config

MOCK_FMP_RESPONSE = [
    {
        "date": "2024-09-30",
        "grossProfitMargin": 0.46,
        "priceToBookRatio": 42.5,
        "revenue": 94000000000,
    }
]


def test_update_fundamentals_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_fundamentals

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "fundamentals.parquet")
    monkeypatch.setattr(config, "FMP_API_KEY", "test-key")

    with patch("src.data.fetch_fundamentals.requests.get") as mock_get:
        mock_get.return_value.json.return_value = MOCK_FMP_RESPONSE
        mock_get.return_value.raise_for_status = lambda: None
        fetch_fundamentals.update_fundamentals(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "fundamentals.parquet")
    assert set(df.columns) >= {"ticker", "date", "gross_profitability", "pb_ratio"}
    assert df["ticker"].iloc[0] == "AAPL"
    assert df["gross_profitability"].iloc[0] == pytest.approx(0.46)
    assert df["pb_ratio"].iloc[0] == pytest.approx(42.5)


def test_update_fundamentals_no_api_key_skips(tmp_path, monkeypatch):
    from src.data import fetch_fundamentals

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "fundamentals.parquet")
    monkeypatch.setattr(config, "FMP_API_KEY", "")

    fetch_fundamentals.update_fundamentals(symbols=["AAPL"])
    assert not (tmp_path / "fundamentals.parquet").exists()
