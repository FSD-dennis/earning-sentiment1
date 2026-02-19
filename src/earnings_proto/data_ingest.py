from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .config import ProjectConfig

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_nodash}/{primary_doc}"


@dataclass(slots=True)
class EarningsEvent:
    ticker: str
    event_date: pd.Timestamp


def _http_get_json(url: str, headers: dict[str, str], timeout: int = 30) -> dict[str, Any]:
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", value)


def get_sec_ticker_map(config: ProjectConfig, user_agent: str) -> pd.DataFrame:
    """Download/cache SEC ticker -> CIK mapping."""
    cache_path = config.data_raw_dir / "sec_ticker_map.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
    else:
        data = _http_get_json(SEC_TICKER_MAP_URL, headers={"User-Agent": user_agent})
        cache_path.write_text(json.dumps(data, indent=2))

    rows = []
    for _, item in data.items():
        rows.append(
            {
                "ticker": str(item["ticker"]).upper(),
                "cik": str(item["cik_str"]).zfill(10),
                "title": item.get("title", ""),
            }
        )
    return pd.DataFrame(rows)


def get_earnings_events_yfinance(ticker: str, n_events: int) -> list[EarningsEvent]:
    """Pull latest earnings dates from yfinance."""
    tk = yf.Ticker(ticker)
    df = tk.get_earnings_dates(limit=max(n_events * 2, n_events + 2))
    if df is None or df.empty:
        return []

    df = df.reset_index().rename(columns={"Earnings Date": "event_date"})
    # Keep historical events only.
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["event_date"] < now_utc]
    df = df.sort_values("event_date", ascending=False).head(n_events)

    events: list[EarningsEvent] = []
    for dt in df["event_date"].tolist():
        ts = pd.Timestamp(dt).tz_convert(None).normalize()
        events.append(EarningsEvent(ticker=ticker.upper(), event_date=ts))
    return events


def get_price_history_yfinance(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Get daily OHLCV using yfinance."""
    hist = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        interval="1d",
        threads=False,
    )
    if hist is None or hist.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns like ('Close', 'AAPL').
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [str(c[0]).lower() for c in hist.columns]
    else:
        hist.columns = [str(c).lower() for c in hist.columns]

    hist = hist.reset_index().rename(columns=str.lower)
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    hist["ticker"] = ticker.upper()

    if "adj close" not in hist.columns:
        hist["adj close"] = hist.get("close")

    keep = ["date", "ticker", "open", "high", "low", "close", "adj close", "volume"]
    keep = [c for c in keep if c in hist.columns]
    out = hist[keep].rename(columns={"adj close": "adj_close"})
    return out


def _extract_readable_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _filter_earnings_relevant_snippets(text: str, max_chars: int = 12000) -> str:
    keywords = {
        "earnings",
        "guidance",
        "revenue",
        "eps",
        "operating",
        "margin",
        "quarter",
        "outlook",
        "expects",
        "expect",
        "forecast",
    }

    sentences = re.split(r"(?<=[.!?])\s+", text)
    hits = [s for s in sentences if any(k in s.lower() for k in keywords)]

    candidate = " ".join(hits) if hits else text[:max_chars]
    if len(candidate) > max_chars:
        candidate = candidate[:max_chars]
    return candidate


def get_sec_recent_8k_text_around_date(
    cik: str,
    event_date: pd.Timestamp,
    user_agent: str,
    max_days_diff: int = 10,
) -> str:
    """Fetch nearest 8-K filing text near event date from SEC archives."""
    headers = {"User-Agent": user_agent}
    subm = _http_get_json(SEC_SUBMISSIONS_URL.format(cik=cik), headers=headers)

    recent = subm.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    candidates: list[tuple[int, str, str]] = []
    for form, acc, fdate, pdoc in zip(forms, accession_numbers, filing_dates, primary_docs):
        if str(form).upper() != "8-K":
            continue
        try:
            filed = pd.Timestamp(fdate)
            diff = abs((filed - event_date).days)
            if diff <= max_days_diff:
                candidates.append((diff, acc, pdoc))
        except Exception:
            continue

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0])
    _, accession_no, primary_doc = candidates[0]
    accession_no_nodash = str(accession_no).replace("-", "")
    cik_int = str(int(cik))
    filing_url = SEC_ARCHIVES_URL.format(
        cik_int=cik_int,
        accession_no_nodash=accession_no_nodash,
        primary_doc=primary_doc,
    )

    resp = requests.get(filing_url, headers=headers, timeout=40)
    if resp.status_code >= 400:
        return ""

    filing_text = _extract_readable_text_from_html(resp.text)
    return _filter_earnings_relevant_snippets(filing_text)


def build_event_level_dataset(
    config: ProjectConfig,
    user_agent: str = "earnings-prototype/0.1 dennisguo527@gmail.com",
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """Build minimal event-level table with event date + text + price window refs."""
    config.ensure_dirs()

    ticker_map = get_sec_ticker_map(config, user_agent=user_agent)
    cik_by_ticker = dict(zip(ticker_map["ticker"], ticker_map["cik"]))

    rows: list[dict[str, Any]] = []
    for ticker in config.tickers:
        events = get_earnings_events_yfinance(ticker, n_events=config.events_per_ticker)
        if not events:
            continue

        for ev in events:
            cik = cik_by_ticker.get(ev.ticker)
            text = ""
            if cik:
                try:
                    text = get_sec_recent_8k_text_around_date(
                        cik=cik,
                        event_date=ev.event_date,
                        user_agent=user_agent,
                    )
                    time.sleep(sleep_seconds)
                except Exception:
                    text = ""

            rows.append(
                {
                    "ticker": ev.ticker,
                    "event_date": ev.event_date,
                    "cik": cik,
                    "source_text": text,
                }
            )

    events_df = pd.DataFrame(rows)
    if events_df.empty:
        return events_df

    events_df["event_date"] = pd.to_datetime(events_df["event_date"])
    events_df = events_df.sort_values(["event_date", "ticker"]).reset_index(drop=True)

    out_path = config.data_processed_dir / "events_base.parquet"
    events_df.to_parquet(out_path, index=False)
    return events_df


def build_and_cache_price_panels(config: ProjectConfig, events_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build per-ticker OHLCV panel around observed events."""
    if events_df.empty:
        return {}

    panels: dict[str, pd.DataFrame] = {}
    for ticker, g in events_df.groupby("ticker"):
        min_event = pd.to_datetime(g["event_date"]).min() - pd.Timedelta(days=60)
        max_event = pd.to_datetime(g["event_date"]).max() + pd.Timedelta(days=30)

        px = get_price_history_yfinance(ticker, start_date=min_event, end_date=max_event)
        if px.empty:
            continue

        px = px.sort_values("date").reset_index(drop=True)
        cache_path = config.data_raw_dir / f"prices_{_safe_filename(ticker)}.parquet"
        px.to_parquet(cache_path, index=False)
        panels[ticker] = px

    return panels
