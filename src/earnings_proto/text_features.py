from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


UNCERTAINTY_WORDS = {
    "may",
    "might",
    "could",
    "uncertain",
    "uncertainty",
    "volatile",
    "risk",
    "risks",
    "challenging",
    "headwind",
    "pressure",
}

CONFIDENCE_WORDS = {
    "confident",
    "strong",
    "momentum",
    "resilient",
    "robust",
    "disciplined",
    "execute",
    "leadership",
    "improving",
}

FORWARD_GUIDANCE_WORDS = {
    "expect",
    "expects",
    "guidance",
    "outlook",
    "forecast",
    "target",
    "plan",
    "plans",
    "will",
    "next quarter",
}

CAUTIOUS_WORDS = {
    "however",
    "but",
    "cautious",
    "carefully",
    "monitor",
    "if",
    "subject to",
    "depending",
}

EMOTIONAL_POSITIVE = {
    "pleased",
    "excited",
    "optimistic",
    "proud",
    "encouraged",
}

EMOTIONAL_NEGATIVE = {
    "disappointed",
    "concerned",
    "challenging",
    "difficult",
    "weak",
}


@dataclass(slots=True)
class TextFeatureResult:
    features: pd.DataFrame
    analyzer_name: str


class LocalToneAnalyzer:
    """CPU-friendly sentiment/tone analyzer.

    This is a baseline replacement for a large LLM call and can run offline.
    """

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())

    def _count_ratio(self, tokens: list[str], lexicon: set[str]) -> float:
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in lexicon)
        return hits / len(tokens)

    def analyze(self, text: str) -> dict[str, float]:
        cleaned = (text or "").strip()
        tokens = self._tokenize(cleaned)

        vader = self._vader.polarity_scores(cleaned)

        return {
            "text_len_chars": float(len(cleaned)),
            "text_len_tokens": float(len(tokens)),
            "sent_compound": float(vader["compound"]),
            "sent_pos": float(vader["pos"]),
            "sent_neg": float(vader["neg"]),
            "sent_neu": float(vader["neu"]),
            "uncertainty_ratio": self._count_ratio(tokens, UNCERTAINTY_WORDS),
            "confidence_ratio": self._count_ratio(tokens, CONFIDENCE_WORDS),
            "forward_guidance_ratio": self._count_ratio(tokens, FORWARD_GUIDANCE_WORDS),
            "cautious_ratio": self._count_ratio(tokens, CAUTIOUS_WORDS),
            "emotion_positive_ratio": self._count_ratio(tokens, EMOTIONAL_POSITIVE),
            "emotion_negative_ratio": self._count_ratio(tokens, EMOTIONAL_NEGATIVE),
        }


def extract_text_features(events_df: pd.DataFrame, text_col: str = "source_text") -> TextFeatureResult:
    analyzer = LocalToneAnalyzer()

    rows: list[dict[str, Any]] = []
    for _, row in events_df.iterrows():
        txt = row.get(text_col, "")
        feats = analyzer.analyze(str(txt) if txt is not None else "")
        feats["ticker"] = row["ticker"]
        feats["event_date"] = row["event_date"]
        feats["has_text"] = float(bool(str(txt).strip()))
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return TextFeatureResult(features=feat_df, analyzer_name="local_lexicon_vader")

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat_df["event_date"] = pd.to_datetime(feat_df["event_date"])
    return TextFeatureResult(features=feat_df, analyzer_name="local_lexicon_vader")


class LLMToneAnalyzerInterface:
    """Swappable interface for an external LLM-based tone extractor.

    Keep this abstraction so you can later drop in a free-tier API call or
    lightweight local model without changing feature pipeline code.
    """

    def analyze(self, text: str) -> dict[str, float]:
        raise NotImplementedError("Implement with your preferred LLM provider/model.")
