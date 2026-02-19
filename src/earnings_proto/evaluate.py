from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .data_ingest import build_and_cache_price_panels, build_event_level_dataset
from .fusion_model import run_time_split_fusion
from .price_features import build_price_feature_table
from .text_features import extract_text_features


def _write_markdown_summary(
    out_path: Path,
    events_df: pd.DataFrame,
    final_df: pd.DataFrame,
    cls_metrics: pd.DataFrame,
    reg_metrics: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Earnings Multimodal Prototype Summary")
    lines.append("")
    lines.append("## Dataset snapshot")
    lines.append(f"- Raw earnings events collected: {len(events_df)}")
    lines.append(f"- Events with complete fused features: {len(final_df)}")
    lines.append(f"- Unique tickers: {final_df['ticker'].nunique() if not final_df.empty else 0}")
    lines.append("")

    lines.append("## Classification metrics (direction)")
    if cls_metrics.empty:
        lines.append("No classification metrics available (insufficient data after filtering).")
    else:
        avg_cls = (
            cls_metrics.groupby("horizon_days")[["accuracy", "f1", "auc"]]
            .mean()
            .round(4)
            .reset_index()
        )
        lines.append(avg_cls.to_markdown(index=False))
    lines.append("")

    lines.append("## Regression metrics (returns)")
    if reg_metrics.empty:
        lines.append("No regression metrics available (insufficient data after filtering).")
    else:
        avg_reg = (
            reg_metrics.groupby("horizon_days")[["mae", "r2"]]
            .mean()
            .round(6)
            .reset_index()
        )
        lines.append(avg_reg.to_markdown(index=False))
    lines.append("")

    lines.append("## Behavior and failure cases")
    lines.append("- Text coverage can be sparse when nearby 8-K filings are unavailable.")
    lines.append("- Daily-bar liquidity proxies are coarse and should be replaced with intraday metrics in v2.")
    lines.append("- Small sample size can make AUC and $R^2$ unstable across folds.")
    lines.append("")

    lines.append("## Improvement plan")
    lines.append("1. Add optional transcript adapter (free-tier API or open transcript archive).")
    lines.append("2. Add market-adjusted targets against SPY and sector ETF baselines.")
    lines.append("3. Replace linear fusion with gradient boosting and calibrated probabilities.")
    lines.append("4. Add SHAP/permutation importance for feature attribution.")

    out_path.write_text("\n".join(lines))


def run_pipeline() -> None:
    config = ProjectConfig()
    config.ensure_dirs()

    events_df = build_event_level_dataset(config=config)
    if events_df.empty:
        raise RuntimeError(
            "No earnings events were collected. Verify internet connectivity and data source availability."
        )

    price_panels = build_and_cache_price_panels(config=config, events_df=events_df)
    price_feat_df = build_price_feature_table(
        events_df=events_df,
        price_panels=price_panels,
        horizons=config.target_horizons,
    )
    if price_feat_df.empty:
        raise RuntimeError(
            "No price features were generated. Check ticker coverage and downloaded OHLCV windows."
        )

    text_feat = extract_text_features(events_df)

    merged = (
        events_df.merge(text_feat.features, on=["ticker", "event_date"], how="left", suffixes=("", "_txt"))
        .merge(price_feat_df, on=["ticker", "event_date"], how="inner", suffixes=("", "_px"))
        .sort_values(["event_date", "ticker"])  # strict temporal order
        .reset_index(drop=True)
    )

    fused_path = config.data_processed_dir / "event_features_fused.parquet"
    merged.to_parquet(fused_path, index=False)

    # Export shareable CSV versions for downstream analysis.
    fused_csv_path = config.data_processed_dir / "event_features_fused.csv"
    merged.to_csv(fused_csv_path, index=False)

    # Compact model-ready export (keeps prediction targets + numeric features + ids).
    target_cols = [f"target_ret_{h}d" for h in config.target_horizons] + [
        f"target_dir_{h}d" for h in config.target_horizons
    ]
    base_cols = ["ticker", "event_date", "source_text"]
    keep_cols = [c for c in base_cols + [c for c in merged.columns if c.startswith("pre_") or c == "event_gap"] + target_cols if c in merged.columns]
    model_export = merged[keep_cols].copy()
    model_csv_path = config.data_processed_dir / "price_prediction_dataset.csv"
    model_export.to_csv(model_csv_path, index=False)

    result = run_time_split_fusion(
        df=merged,
        target_horizons=[h for h in config.target_horizons if h in (1, 3, 7)],
        text_col="source_text",
        min_splits=3,
    )

    cls_out = config.reports_dir / "classification_metrics.csv"
    reg_out = config.reports_dir / "regression_metrics.csv"
    result.classification_metrics.to_csv(cls_out, index=False)
    result.regression_metrics.to_csv(reg_out, index=False)

    # Save quick metadata.
    meta = {
        "n_events_raw": int(len(events_df)),
        "n_events_fused": int(len(merged)),
        "tickers": sorted(merged["ticker"].unique().tolist()) if not merged.empty else [],
    }
    (config.reports_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    summary_path = config.reports_dir / "prototype_summary.md"
    _write_markdown_summary(
        out_path=summary_path,
        events_df=events_df,
        final_df=merged,
        cls_metrics=result.classification_metrics,
        reg_metrics=result.regression_metrics,
    )

    print("Pipeline complete.")
    print(f"Fused data: {fused_path}")
    print(f"Fused CSV: {fused_csv_path}")
    print(f"Model CSV: {model_csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run_pipeline()
