# Earnings Multimodal Fusion Prototype

Local CPU-friendly prototype for predicting short-horizon post-earnings returns using:
- Text modality: SEC earnings-related filing text (8-K around event) with lexicon features.
- Market modality: pre-earnings volatility, momentum, volume/liquidity proxies.
- Fusion: TF-IDF + engineered numeric features into linear baseline models.

## Quick start

1. Install dependencies
2. Run the pipeline entrypoint:

```bash
conda create -n earnings python=3.14.2 -y
conda activate earnings
pip install -r requirements.txt -y
python -m src.earnings_proto.evaluate
```

Outputs are written to:
- `data/raw/` (cached source pulls)
- `data/processed/` (event-level feature table, training dataset)
- `reports/` (metrics and artifacts)

## Notes

- This v1 uses public/free sources and is designed to run on a laptop.
- Earnings call transcripts are often paywalled; this prototype uses SEC earnings-related filing text as a practical baseline and keeps a pluggable LLM interface for future upgrades.


## Feature Description

One row = one earnings event with fused text + price features, plus prediction targets.

### Base Columns
- **ticker**: Stock ticker symbol (uppercase), e.g., AAPL.
- **event_date**: Event date aligned to the trading calendar (first trading day on/after the raw earnings date).
- **source_text**: Cleaned event-related text (e.g., nearby SEC 8-K content). Used as input to TF-IDF in the fusion model.

### Pre-event Momentum (trend into the event)
> In the price feature code, `pre` refers to the trading day **before** the event day.
- **pre_mom_1d**: 1-day momentum into the event.  
  `pre_mom_1d = (Close_pre / Close_pre-1) - 1`
- **pre_mom_5d**: 5-day momentum into the event.  
  `pre_mom_5d = (Close_pre / Close_pre-5) - 1`
- **pre_mom_20d**: 20-day momentum into the event.  
  `pre_mom_20d = (Close_pre / Close_pre-20) - 1`

### Pre-event Volatility (risk/regime)
- **pre_rv_20**: 20-day realized volatility (annualized) based on daily close-to-close returns.  
  `pre_rv_20 = std(ret_1d over last 20 trading days) * sqrt(252)`
- **pre_vol_compression_5v20**: Short-term vs medium-term volatility ratio.  
  `pre_vol_compression_5v20 = std(ret_1d over last 5 days) / std(ret_1d over last 20 days)`  
  Interpretation: `< 1` = volatility compression, `> 1` = volatility expansion.

### Pre-event Volume / Liquidity Proxies
- **pre_volume_spike**: Volume spike on the pre-event day relative to the last 20-day average.  
  `pre_volume_spike = Volume_pre / mean(Volume over last 20 days)`
- **pre_liquidity_gap_proxy**: Coarse “jumpiness/liquidity” proxy from daily candles (median over last 5 days).  
  `pre_liquidity_gap_proxy = median( abs(Close - Open) / Open over last 5 days )`  
  Note: This is a proxy from daily bars (not true bid-ask liquidity).

### Event-day Gap
- **event_gap**: Overnight gap proxy on the event trading day.  
  `event_gap = (Open_event / Close_event-1) - 1`

### Targets (Prediction Labels)
> Targets are labels computed from the event-day close to future closes (not input features).
- **target_ret_1d / target_ret_3d / target_ret_7d**: Post-event return over `h` trading days.  
  `target_ret_hd = (Close_event+h / Close_event) - 1`  
  Used for **regression**.
- **target_dir_1d / target_dir_3d / target_dir_7d**: Direction label derived from `target_ret_hd`.  
  `target_dir_hd = 1 if target_ret_hd > 0 else 0`  
  Used for **classification**.
