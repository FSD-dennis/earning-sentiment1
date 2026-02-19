from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(slots=True)
class FusionRunResult:
    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame


def _build_feature_pipeline(text_col: str, num_cols: list[str]) -> ColumnTransformer:
    text_pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=1)),
        ]
    )

    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    return ColumnTransformer(
        transformers=[
            ("text", text_pipe, text_col),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def run_time_split_fusion(
    df: pd.DataFrame,
    target_horizons: list[int],
    text_col: str = "source_text",
    min_splits: int = 3,
) -> FusionRunResult:
    if df.empty:
        return FusionRunResult(pd.DataFrame(), pd.DataFrame())

    model_df = df.sort_values("event_date").reset_index(drop=True).copy()
    model_df[text_col] = model_df[text_col].fillna("")

    non_feature_cols = {
        "ticker",
        "event_date",
        text_col,
    }
    non_feature_cols.update({f"target_ret_{h}d" for h in target_horizons})
    non_feature_cols.update({f"target_dir_{h}d" for h in target_horizons})

    num_cols = [c for c in model_df.columns if c not in non_feature_cols]

    tscv = TimeSeriesSplit(n_splits=min_splits)

    cls_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []

    for h in target_horizons:
        y_cls_col = f"target_dir_{h}d"
        y_reg_col = f"target_ret_{h}d"

        work = model_df.dropna(subset=[y_cls_col, y_reg_col]).copy()
        if len(work) < (min_splits + 2):
            continue

        X = work[[text_col] + num_cols]
        y_cls = work[y_cls_col].astype(int).values
        y_reg = work[y_reg_col].astype(float).values

        split_i = 0
        for train_idx, test_idx in tscv.split(X):
            split_i += 1
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_cls_train, y_cls_test = y_cls[train_idx], y_cls[test_idx]
            y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

            feat_pipe_cls = _build_feature_pipeline(text_col=text_col, num_cols=num_cols)
            clf = LogisticRegression(max_iter=400, class_weight="balanced")
            cls_model = Pipeline(steps=[("features", feat_pipe_cls), ("model", clf)])
            cls_model.fit(X_train, y_cls_train)

            y_prob = cls_model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y_cls_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)

            cls_rows.append(
                {
                    "horizon_days": h,
                    "split": split_i,
                    "n_test": len(test_idx),
                    "accuracy": float(accuracy_score(y_cls_test, y_pred)),
                    "f1": float(f1_score(y_cls_test, y_pred, zero_division=0)),
                    "auc": _safe_auc(y_cls_test, y_prob),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )

            feat_pipe_reg = _build_feature_pipeline(text_col=text_col, num_cols=num_cols)
            reg = Ridge(alpha=1.0)
            reg_model = Pipeline(steps=[("features", feat_pipe_reg), ("model", reg)])
            reg_model.fit(X_train, y_reg_train)

            y_hat = reg_model.predict(X_test)
            reg_rows.append(
                {
                    "horizon_days": h,
                    "split": split_i,
                    "n_test": len(test_idx),
                    "mae": float(mean_absolute_error(y_reg_test, y_hat)),
                    "r2": float(r2_score(y_reg_test, y_hat)),
                }
            )

    return FusionRunResult(
        classification_metrics=pd.DataFrame(cls_rows),
        regression_metrics=pd.DataFrame(reg_rows),
    )
