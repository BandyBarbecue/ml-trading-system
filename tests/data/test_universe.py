import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.data.sp500_universe import load_sp500_tickers, get_symbols

MOCK_DF = pd.DataFrame({
    "Symbol": ["AAPL", "MSFT", "BRK.B"],
    "Security": ["Apple", "Microsoft", "Berkshire"],
    "GICS Sector": ["Technology", "Technology", "Financials"],
    "GICS Sub-Industry": ["Tech Hardware", "Software", "Insurance"],
})

def test_get_symbols_returns_list(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.sp500_universe.UNIVERSE_FILE", tmp_path / "sp500.csv")
    with patch("src.data.sp500_universe.pd.read_html", return_value=[MOCK_DF]):
        symbols = get_symbols()
    assert isinstance(symbols, list)
    assert "AAPL" in symbols
    assert "MSFT" in symbols

def test_brk_dot_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.sp500_universe.UNIVERSE_FILE", tmp_path / "sp500.csv")
    with patch("src.data.sp500_universe.pd.read_html", return_value=[MOCK_DF]):
        symbols = get_symbols()
    # BRK.B -> BRK-B (Alpaca uses hyphens)
    assert "BRK-B" in symbols
    assert "BRK.B" not in symbols

def test_csv_written(tmp_path, monkeypatch):
    csv_path = tmp_path / "sp500.csv"
    monkeypatch.setattr("src.data.sp500_universe.UNIVERSE_FILE", csv_path)
    with patch("src.data.sp500_universe.pd.read_html", return_value=[MOCK_DF]):
        load_sp500_tickers()
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "ticker" in df.columns
    assert "sector" in df.columns
