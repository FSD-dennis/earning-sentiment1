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

One row = one earnings event with fused **text** + **price** features, plus prediction targets.

### Base Columns
- **ticker**: Stock ticker symbol (uppercase), e.g., `AAPL`.
- **event_date**: Event date aligned to the trading calendar (first trading day on/after the raw earnings date).
- **source_text**: Cleaned event-related text (e.g., nearby SEC 8-K content). Used as input to TF-IDF in the fusion model.

---

### Pre-event Momentum (trend into the event)

> In the price feature code, `pre` refers to the trading day **before** the event day.

- **pre_mom_1d**: 1-day momentum into the event:

$$
pre\_mom\_{1d}=\frac{Close_{pre}}{Close_{pre-1}}-1
$$

- **pre_mom_5d**: 5-day momentum into the event:

$$
pre\_mom\_{5d}=\frac{Close_{pre}}{Close_{pre-5}}-1
$$

- **pre_mom_20d**: 20-day momentum into the event:

$$
pre\_mom\_{20d}=\frac{Close_{pre}}{Close_{pre-20}}-1
$$

---

### Pre-event Volatility (risk/regime)

- **pre_rv_20**: 20-day realized volatility (annualized) using daily close-to-close returns.

Let $ret_{1d,t}=\frac{Close_t}{Close_{t-1}}-1$. Then:

$$
pre\_rv\_{20}=\sigma\left(ret_{1d,\,pre-20:pre}\right)\cdot\sqrt{252}
$$

- **pre_vol_compression_5v20**: Short-term vs medium-term volatility ratio:

$$
pre\_vol\_compression\_{5v20}=\frac{\sigma\left(ret_{1d,\,pre-5:pre}\right)}{\sigma\left(ret_{1d,\,pre-20:pre}\right)}
$$

Interpretation: $<1$ indicates volatility compression; $>1$ indicates volatility expansion.

---

### Pre-event Volume / Liquidity Proxies

- **pre_volume_spike**: Volume spike on the pre-event day relative to the last 20-day average:

$$
pre\_volume\_spike=\frac{Volume_{pre}}{\mathrm{mean}\left(Volume_{pre-20:pre}\right)}
$$

- **pre_liquidity_gap_proxy**: Coarse “jumpiness/liquidity” proxy from daily candles (median over last 5 days):

$$
pre\_liquidity\_gap\_proxy=\mathrm{median}\left(\frac{|Close-Open|}{Open}\right)_{pre-5:pre}
$$

Note: this is a proxy from daily bars (not a true bid-ask spread).

---

### Event-day Gap

- **event_gap**: Overnight gap proxy on the event trading day:

$$
event\_gap=\frac{Open_{event}}{Close_{event-1}}-1
$$

---

### Targets (Prediction Labels)

> Targets are labels computed from the event-day close to future closes (not input features).

- **target_ret_1d / target_ret_3d / target_ret_7d**: Post-event return over $h$ trading days:

$$
target\_ret_{hd}=\frac{Close_{event+h}}{Close_{event}}-1
$$

Used for **regression**.

- **target_dir_1d / target_dir_3d / target_dir_7d**: Direction label derived from $target\_ret_{hd}$:

$$
target\_dir_{hd}=
\begin{cases}
1, & target\_ret_{hd} > 0 \\
0, & target\_ret_{hd} \le 0
\end{cases}
$$

Used for **classification**.
