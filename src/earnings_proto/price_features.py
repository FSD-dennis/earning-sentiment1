from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLS = {"date", "open", "high", "low", "close", "volume"}


def _validate_price_df(px: pd.DataFrame) -> bool:
    return REQUIRED_PRICE_COLS.issubset(set(px.columns)) and not px.empty


def _nearest_idx_for_event(px: pd.DataFrame, event_date: pd.Timestamp) -> int | None:
    # Use first trading day on/after event_date. Conservative assumption: event is after close.
    cond = px["date"] >= pd.Timestamp(event_date)
    if not cond.any():
        return None
    return int(cond.idxmax())


def _safe_ret(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a / b) - 1.0


def compute_price_features_for_event(
    px: pd.DataFrame,
    event_date: pd.Timestamp,
    horizons: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if not _validate_price_df(px):
        return out

    px = px.sort_values("date").reset_index(drop=True).copy()
    px["ret_1d"] = px["close"].pct_change()

    idx = _nearest_idx_for_event(px, event_date)
    if idx is None or idx < 21:
        return out

    # Pre-event windows reference day before event trading day.
    pre_idx = idx - 1
    pre_1 = max(0, pre_idx - 1)
    pre_5 = max(0, pre_idx - 5)
    pre_20 = max(0, pre_idx - 20)

    close_t1 = float(px.loc[pre_idx, "close"])
    open_t1 = float(px.loc[pre_idx, "open"])

    out["pre_mom_1d"] = _safe_ret(close_t1, float(px.loc[pre_1, "close"]))
    out["pre_mom_5d"] = _safe_ret(close_t1, float(px.loc[pre_5, "close"]))
    out["pre_mom_20d"] = _safe_ret(close_t1, float(px.loc[pre_20, "close"]))

    ret_hist_20 = px.loc[pre_20:pre_idx, "ret_1d"].dropna()
    ret_hist_5 = px.loc[pre_5:pre_idx, "ret_1d"].dropna()
    out["pre_rv_20"] = float(ret_hist_20.std(ddof=0) * np.sqrt(252)) if len(ret_hist_20) > 2 else np.nan
    std_5 = float(ret_hist_5.std(ddof=0)) if len(ret_hist_5) > 2 else np.nan
    std_20 = float(ret_hist_20.std(ddof=0)) if len(ret_hist_20) > 2 else np.nan
    out["pre_vol_compression_5v20"] = (std_5 / std_20) if std_20 and not np.isnan(std_20) else np.nan

    vol_20 = px.loc[pre_20:pre_idx, "volume"].replace(0, np.nan)
    out["pre_volume_spike"] = float(px.loc[pre_idx, "volume"] / vol_20.mean()) if vol_20.notna().any() else np.nan

    # Liquidity gap proxy from daily candles (absolute open-close spread normalized).
    spread = (px.loc[pre_5:pre_idx, "close"] - px.loc[pre_5:pre_idx, "open"]).abs()
    denom = px.loc[pre_5:pre_idx, "open"].replace(0, np.nan)
    liq_proxy = (spread / denom).replace([np.inf, -np.inf], np.nan).dropna()
    out["pre_liquidity_gap_proxy"] = float(liq_proxy.median()) if not liq_proxy.empty else np.nan

    # Event-day overnight gap proxy.
    if idx > 0:
        out["event_gap"] = _safe_ret(float(px.loc[idx, "open"]), float(px.loc[idx - 1, "close"]))
    else:
        out["event_gap"] = np.nan

    # Targets: post-earnings returns from event-day close to horizon close.
    event_close = float(px.loc[idx, "close"])
    for h in horizons:
        tgt_idx = idx + h
        key = f"target_ret_{h}d"
        dir_key = f"target_dir_{h}d"
        if tgt_idx < len(px):
            ret = _safe_ret(float(px.loc[tgt_idx, "close"]), event_close)
            out[key] = ret
            out[dir_key] = float(ret > 0) if not np.isnan(ret) else np.nan
        else:
            out[key] = np.nan
            out[dir_key] = np.nan

    return out


def build_price_feature_table(
    events_df: pd.DataFrame,
    price_panels: dict[str, pd.DataFrame],
    horizons: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, ev in events_df.iterrows():
        ticker = ev["ticker"]
        event_date = pd.Timestamp(ev["event_date"])
        px = price_panels.get(ticker)
        if px is None or px.empty:
            continue

        feat = compute_price_features_for_event(px=px, event_date=event_date, horizons=horizons)
        if not feat:
            continue
        feat["ticker"] = ticker
        feat["event_date"] = event_date
        rows.append(feat)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "event_date"])

    df = df.replace([np.inf, -np.inf], np.nan)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df
