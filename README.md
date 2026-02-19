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

This project builds an event-level dataset (one row per earnings event) with fused **text** and **price-based** features, plus **targets** for prediction.

### Columns

- **ticker**  
  Stock ticker symbol (uppercase), e.g., `AAPL`.

- **event_date**  
  The event date aligned to the trading calendar (the first trading day on/after the raw earnings date).

- **source_text**  
  Cleaned text associated with the event (e.g., nearby SEC 8-K content). Used as the main text input for TF-IDF features in the fusion model.

---

### Pre-event Momentum (Price Trend)

> In the price feature code, `pre_idx = idx - 1` (the trading day **before** the event day).

- **pre_mom_1d**  
  1-day pre-event momentum:  
  \[
  \frac{Close_{pre}}{Close_{pre-1}} - 1
  \]
  Measures very short-term trend into the event.

- **pre_mom_5d**  
  5-day pre-event momentum:  
  \[
  \frac{Close_{pre}}{Close_{pre-5}} - 1
  \]
  Measures weekly-scale trend into the event.

- **pre_mom_20d**  
  20-day pre-event momentum:  
  \[
  \frac{Close_{pre}}{Close_{pre-20}} - 1
  \]
  Measures ~1-month trend into the event.

---

### Pre-event Volatility (Risk / Regime)

- **pre_rv_20**  
  20-day realized volatility (annualized) using daily close-to-close returns:  
  \[
  \sigma(\text{ret}_{1d,20}) \times \sqrt{252}
  \]
  Higher values indicate more unstable price behavior before the event.

- **pre_vol_compression_5v20**  
  Short-term vs medium-term volatility ratio:  
  \[
  \frac{\sigma(\text{ret}_{1d,5})}{\sigma(\text{ret}_{1d,20})}
  \]
  Values < 1 suggest volatility compression; > 1 suggests recent volatility expansion.

---

### Pre-event Volume / Liquidity Proxies

- **pre_volume_spike**  
  Pre-event volume spike ratio:  
  \[
  \frac{Volume_{pre}}{\text{mean}(Volume_{pre-20:pre})}
  \]
  Values > 1 indicate unusually high volume right before the event.

- **pre_liquidity_gap_proxy**  
  Coarse liquidity / “jumpiness” proxy from daily candles (median over the last 5 days):  
  \[
  \text{median}\left(\frac{|Close - Open|}{Open}\right)
  \]
  Larger values indicate larger typical open-to-close moves (proxy only; not true bid-ask liquidity).

---

### Event-day Gap

- **event_gap**  
  Overnight gap proxy on the event trading day:  
  \[
  \frac{Open_{event}}{Close_{event-1}} - 1
  \]
  Captures jump at the open relative to the previous close.

---

### Targets (Prediction Labels)

> These are **labels**, not input features. They are computed from the event-day close to future closes.

- **target_ret_1d / target_ret_3d / target_ret_7d**  
  Post-event return over `h` trading days:  
  \[
  \frac{Close_{event+h}}{Close_{event}} - 1
  \]
  Used for **regression**.

- **target_dir_1d / target_dir_3d / target_dir_7d**  
  Direction label derived from `target_ret_hd`:  
  - `1` if `target_ret_hd > 0`  
  - `0` if `target_ret_hd <= 0`  
  Used for **classification**.
