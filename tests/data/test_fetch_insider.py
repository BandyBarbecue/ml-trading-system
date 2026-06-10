import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import config

MOCK_EDGAR_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form":            ["4", "10-K"],
            "filingDate":      ["2024-05-01", "2024-03-01"],
            "primaryDocument": ["doc1.htm",   "doc2.htm"],
        }
    }
}

MOCK_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>TIM COOK</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-05-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>185.00</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_update_insider_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_insider

    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "insider.parquet")

    responses = []
    # First call: EDGAR submissions JSON
    resp1 = MagicMock()
    resp1.json.return_value = MOCK_EDGAR_SUBMISSIONS
    resp1.raise_for_status = lambda: None
    responses.append(resp1)
    # Second call: Form 4 XML for the first filing
    resp2 = MagicMock()
    resp2.text = MOCK_FORM4_XML
    resp2.raise_for_status = lambda: None
    responses.append(resp2)

    with patch("src.data.fetch_insider.requests.get", side_effect=responses):
        fetch_insider.update_insider(symbols=["AAPL"], cik_map={"AAPL": "0000320193"})

    df = pd.read_parquet(tmp_path / "insider.parquet")
    assert set(df.columns) >= {"ticker", "date", "insider_type", "transaction_type", "shares", "value_usd"}
    assert df["ticker"].iloc[0] == "AAPL"
    assert df["transaction_type"].iloc[0] == "P"
    # CEO gets 2x weight multiplier: 10000 shares * 2.0
    assert df["shares"].iloc[0] == pytest.approx(20000.0)


def test_parse_form4_ceo_weight():
    """CEO transactions should have 2x weight multiplier."""
    from src.data.fetch_insider import _parse_form4

    rows = _parse_form4(MOCK_FORM4_XML, "AAPL")
    assert len(rows) == 1
    assert rows[0]["transaction_type"] == "P"
    assert rows[0]["shares"] == pytest.approx(20000.0)   # 10000 * 2.0
    assert rows[0]["value_usd"] == pytest.approx(20000.0 * 185.0)


def test_parse_form4_ignores_non_signal_codes():
    """Grant codes (A, F) should be ignored."""
    from src.data.fetch_insider import _parse_form4

    grant_xml = MOCK_FORM4_XML.replace(
        "<transactionCode>P</transactionCode>",
        "<transactionCode>A</transactionCode>"
    )
    rows = _parse_form4(grant_xml, "AAPL")
    assert len(rows) == 0  # A = Award/Grant, not a market buy
