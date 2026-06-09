import pandas as pd
import pytest
from unittest.mock import patch
from src.data.fetch_vix import update_vix

MOCK_VIX = pd.DataFrame(
    {"Close": [18.5, 19.2, 17.8]},
    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
)
MOCK_VIX.index.name = "Date"

def test_update_vix_creates_parquet(tmp_path, monkeypatch):
    vix_path = tmp_path / "vix.parquet"
    monkeypatch.setattr("src.data.fetch_vix.VIX_FILE", vix_path)
    monkeypatch.setattr("src.data.fetch_vix.DATA_DIR", tmp_path)
    with patch("src.data.fetch_vix.yf.download", return_value=MOCK_VIX):
        update_vix()
    assert vix_path.exists()
    df = pd.read_parquet(vix_path)
    assert "vix" in df.columns
    assert len(df) == 3
