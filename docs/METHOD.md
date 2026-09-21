# HoloMine Property Price Prediction From Sales Description
# - NLP regression: predict listPrice from listing text only.
# - Target right-skewed (skew ~15.3) -> modeled as log1p(listPrice),
#   predictions inverse-transformed with expm1 + clipped to train [min,max].
# - Metric: MAE on the original price scale (lower is better).
# - Submission: id,listPrice, same row order as sample_submission.csv.

## 1. Validation (frozen)
# - StratifiedKFold(5, seed=42) over 10 qcut bins of log1p(price).
# - folds.csv written once in CELL 2 and frozen; all later experiments reuse
#   the same folds. Baseline median CV-MAE (~550k) is the CV zero point.
# - Zero train/test id overlap asserted; duplicate-text overlap checked.

## 2. Text handling
# - Missing text -> "". MAX_LEN=512 chosen after checking char/4 estimates
#   and exact token lengths on a 1000-row sample (median/p90/%>512).
# - No fitting on test: TF-IDF/Ridge and all scalers fit on train-folds only.

## 3. Models
# - CELL 3 DeBERTa-v3-base + regression head: mean pooling (NOT CLS),
#   per-fold target centering, L1 loss on log-target (aligned with MAE),
#   effective batch 16, lr 2e-5, 3 epochs, cosine + 10% warmup, fp16 AMP,
#   dropout 0.1, dynamic padding. v1 (CLS, no centering) scored 659k CV-MAE,
#   worse than the 550k median baseline; v2 (mean-pool + centering) ~355k.
# - CELL 3B TF-IDF(100k, 1-2gram, sublinear, min_df=2) + Ridge(alpha=0.5) on
#   log1p: CPU ~2 min, CV-MAE ~347k. Cheap safety net, best single model.
# - CELL 3D regex domain features + LightGBM(objective=l1, early stopping
#   100): beds/baths/sqft/acre/lot/HOA/year/garage + flags (pool,
#   waterfront, renovated, fixer, garage, luxury, view) + length features,
#   fillna(-1). CV-MAE ~460k standalone; blend gain vs Ridge only ~800
#   (<< 30k std = tie) -> kept for diversity, not strength.
# - CELL 7 ModernBERT-base: identical setup to CELL 3 for architecture
#   diversity.

## 4. Error analysis (CELL 4)
# - MAE per price decile + MAE% to find where the model errs most; L1 vs MSE
#   loss verdict (L1 kept; MSE re-test skipped on time).

## 5. Blend & final fit (CELL 5/8)
# - Simplex grid search (step 20, weights >= 0, sum 1) optimized on leak-free
#   OOF. Best blend OOF-MAE 336,815.
# - Final test prediction = fold-mean across the 5 fold models, NOT
#   refit-full (14.6k rows is small; fold-mean is more robust and gives free
#   fold ensembling).
# - Post-processing: expm1 -> clip to train [min,max] -> order by
#   sample_submission -> full asserts (columns/rows/order/NaN/inf/range/ids)
#   -> distribution sanity plot (assets/distribution_train_vs_pred.png).
# - The old submission.csv (429k LB) is never overwritten; blends write a
#   separate file so a safe fallback always survives.

## 6. Decision log (recorded numbers)
# - baseline_median ~550,252 | deberta v1 659,867 (LB 732,654) | ridge 346,772
# - deberta-v2 354,855 clean / 409,370 corrupted fold-2 | lgbm 459,963
# - blend v1+ridge w_tf=0.05 -> discarded | CELL 5 blend OOF 336,815.
