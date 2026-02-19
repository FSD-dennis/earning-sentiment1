# Earnings Multimodal Prototype Summary

## Dataset snapshot
- Raw earnings events collected: 24
- Events with complete fused features: 24
- Unique tickers: 6

## Classification metrics (direction)
|   horizon_days |   accuracy |     f1 |   auc |
|---------------:|-----------:|-------:|------:|
|              1 |     0.4444 | 0.4889 |  0.75 |
|              3 |     0.4444 | 0.4889 |  0.75 |
|              7 |     0.4444 | 0.4889 |  0.3  |

## Regression metrics (returns)
|   horizon_days |      mae |       r2 |
|---------------:|---------:|---------:|
|              1 | 0.083437 | -1.98014 |
|              3 | 0.101858 | -2.75602 |
|              7 | 0.10539  | -3.12413 |

## Behavior and failure cases
- Text coverage can be sparse when nearby 8-K filings are unavailable.
- Daily-bar liquidity proxies are coarse and should be replaced with intraday metrics in v2.
- Small sample size can make AUC and $R^2$ unstable across folds.

## Improvement plan
1. Add optional transcript adapter (free-tier API or open transcript archive).
2. Add market-adjusted targets against SPY and sector ETF baselines.
3. Replace linear fusion with gradient boosting and calibrated probabilities.
4. Add SHAP/permutation importance for feature attribution.