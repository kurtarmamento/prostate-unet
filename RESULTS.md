# RESULTS (2.5D Baseline)

This document records the evaluation results for the current baseline.  
**Chosen global threshold (from validation):** `0.30`

## Metrics
| Split | Threshold |  LCC | Mean Dice |
|------:|:---------:|:----:|----------:|
|  Val  |   0.30    |  ✅  |    0.5155 |
|  Test |   0.30    |  ✅  |    0.5155 |

> Dice is computed over full-volume inference on the held-out split using the selected threshold (val-selected) and optional LCC post-processing.

> Sources:  
> - `artifacts/preds_val/summary_val.json` (best_global_threshold, mean)  
> - `artifacts/preds_test/summary_test.json` and/or `scripts/eval_volumes.py` output

## Notes
- Validation was used to select the global threshold (0.30).  
- Test results were computed once using that threshold; LCC enabled.
- For full reproducibility, see `artifacts/run_meta.json`.
