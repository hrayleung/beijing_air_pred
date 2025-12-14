# Preprocessing Pipeline v2.1 - CHANGELOG

## Summary of Fixes

### FIX A: SCALE_TARGETS Properly Implemented

**Problem**: Y was always raw regardless of `Config.SCALE_TARGETS` setting.

**Solution**:
- Added `fit_target_scaler()` function that fits a separate RobustScaler on TRAIN observed values only
- Uses per-target-dimension scaling (D=6 separate center/scale values)
- Only fits on values where `Y_mask==1` (observed)
- Added `apply_target_scaler()` that scales only observed positions
- Saves both `input_scaler` and `target_scaler` in `scaler.pkl`
- Updated `scaler_params.json` to include target scaling parameters

```python
# New scaler.pkl structure:
{
    'input_scaler': RobustScaler,  # For X features
    'target_scaler': RobustScaler or None  # For Y targets (if SCALE_TARGETS=True)
}
```

### FIX B: LightGBM valid_start Respects Max Lag

**Problem**: `valid_start = dt_index[L-1]` but lag168 needs 168 prior values, causing NaN-heavy early rows.

**Solution**:
```python
min_origin_idx = max(LOOKBACK - 1, max(LGBM_LAGS), max(LGBM_ROLLING_WINDOWS) - 1)
# = max(167, 168, 167) = 168

valid_start = dt_index[min_origin_idx]
```

This ensures all lag features have sufficient history.

### FIX C: P2 Outputs Include Y_mask

**Problem**: P2 saved Y with NaN→0 but dropped masks, making it impossible to distinguish missing from zero values.

**Solution**:
- `non_causal_impute()` now returns mask (like `causal_impute()`)
- P2 `.npz` files now include `Y_mask`
- README updated to emphasize Y_mask must be used for loss/metrics

```python
# P2 now saves:
{
    'X': ...,
    'Y': ...,
    'Y_mask': ...,  # NEW - must use in evaluation
    'X_flat': ...,
    'Y_flat': ...,
    'datetime_origins': ...
}
```

### FIX D: Rigorous Validation Test #3

**Problem**: Test 3 only printed counts, didn't actually verify Y values match raw targets.

**Solution**: Implemented rigorous spot-check:
```python
K = 200  # samples per split
for each sampled (i, h, s, d):
    y_stored = windows[split]['Y'][i, h, s, d]
    y_mask = windows[split]['Y_mask'][i, h, s, d]
    y_raw = raw_splits[split][0][raw_idx, s, target_indices[d]]
    
    if np.isnan(y_raw):
        assert y_mask == 0 and y_stored == 0
    else:
        assert y_mask == 1
        if SCALE_TARGETS:
            expected = (y_raw - center) / scale
        else:
            expected = y_raw
        assert np.isclose(y_stored, expected)
```

### FIX E: Lag Naming Clarification

**Clarification in code and README**:
- `lag1` = value at t-1 (shift(1))
- `lag168` = value at t-168 (shift(168))
- **NO lag0** included - current value not used as feature
- All features strictly causal (use values from time < t)

---

## File Changes

| File | Change |
|------|--------|
| `preprocessing_pipeline_v2.1.py` | New version with all fixes |
| `processed/metadata.json` | Added `lgbm_min_origin_idx`, updated version to 2.1 |
| `processed/P1_deep/scaler.pkl` | Now contains both `input_scaler` and `target_scaler` |
| `processed/P1_deep/scaler_params.json` | Added target scaler params |
| `processed/P2_simple/*.npz` | Now includes `Y_mask` |
| `processed/README.md` | Updated with all clarifications |

---

## Running the Pipeline

```bash
# Default mode (SCALE_TARGETS=False)
python preprocessing_pipeline_v2.1.py

# The script automatically runs both modes for verification
```

## Expected Output

```
RUN 1: SCALE_TARGETS=False (default)
  train: 26113 samples
  val: 5689 samples  
  test: 2689 samples
  PM2.5 mean (observed): ~70 (raw scale)
  Tests passed: True

RUN 2: SCALE_TARGETS=True
  train: 26113 samples
  val: 5689 samples
  test: 2689 samples
  PM2.5 mean (observed): ~0 (scaled)
  Target scaler centers: [~70, ~115, ~15, ~45, ~1000, ~55]
  Tests passed: True
```

---

## Backward Compatibility

- P1 outputs: Compatible (same keys, scaler.pkl now has dict structure)
- P2 outputs: **Breaking change** - now includes Y_mask (but this is a fix, not a regression)
- LightGBM: Fewer rows due to stricter valid_start (correct behavior)
- metadata.json: New fields added, version bumped to 2.1
