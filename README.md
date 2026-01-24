# Trend Analyzer (Nifty) — Local Runner

Computes a **daily 0–100 score** for:
- Nifty 50
- Nifty Midcap 100
- Nifty Smallcap 100

Scores incorporate equity trend/overbought-oversold/stress plus cross-asset **risk-off** signals (gold, silver, FX, volatility, US bond yields) and explicitly models **divergences** between asset classes.

This is meant to be a **local decision-support tool** (not a broker / execution system).

**Version note:** this is **v1** of the model/logic and there is plenty of scope to improve and change thresholds, features, and model design as you iterate.

## Strengths and limitations (v1)
This v1 is **especially helpful for detecting stress, crash regimes, and potential bottoms**, because it is intentionally **sensitive to fast selloffs and fast rebounds** (impulse + stress + cross-asset confirmation).

Why:
- Fast downside moves create strong “impulse” + stress signals (and cross-asset risk-off confirmation) that the score reacts to quickly.
- Market tops can be **slow and choppy**, with fewer clear stress signals until late, so the model may lag or stay neutral/bullish longer.

Practical takeaway:
- Use this tool primarily for **risk management, spotting stress, and spotting potential bottoms/bear-market-bounce conditions**.
- For “top detection”, you’ll likely want additional modules later (breadth, valuation, sentiment, longer-horizon divergences, etc.).

## Setup

Create a virtualenv, then install deps:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

```bash
.venv\Scripts\python -m trend_analyzer.run --config config.yaml --out-dir outputs --refresh
```

Outputs:
- `outputs/scores.csv` (daily scores + labels + risk-off + divergence fields)
- `outputs/aligned_prices.csv` (aligned raw price series used to compute everything)
- `outputs/features_*.csv` (optional; enable via `output.write_features: true` in `config.yaml`)
  - `features_<index>.csv` (per-index engineered features)
  - `features_cross_asset.csv` (cross-asset/divergence features)
  - `features_model_X.csv` (the model input table used for `risk_off_prob`)

## How the score works (logic)
Each index gets a **final 0–100 score** built from these parts:

- **TrendScore** (0..60): strength of the medium-term up/down trend (EMA slope/relationships) and stress (drawdown).
- **ReversionAdjustment** (-20..+20): overbought/oversold adjustment from RSI and price extremes.
- **ImpulseAdjustment** (-20..+20): reacts to **fast selloffs and fast buying** (short-horizon momentum vs volatility).
- **RiskPenalty** (0..20): penalty from a learned “risk-off regime” probability.

Final:

- `score = clamp( trend_score + neutral_shift + reversion_adj_eff + impulse_adj_eff - risk_penalty, 0, 100 )`

Where:
- `neutral_shift` defaults to **20**
- `reversion_adj_eff` and `impulse_adj_eff` are the **effective** adjustments after applying a “risk context” damping factor during strong risk-off backdrops (to avoid treating every bounce as a healthy recovery).

### Labels (how to interpret score levels)
The tool maps score into a human-readable bucket:

- 0–20: `oversold_panic`
- 20–40: `bearish_below_trend`
- 40–60: `neutral_fair_vs_trend`
- 60–80: `bullish_overboughtish`
- 80–100: `euphoric_very_overbought`

### What the “risk-off regime model” is
The tool learns a daily **`risk_off_prob`** using a **walk-forward logistic regression** trained on the Nifty 50.

- **Target/label** (what it tries to predict): whether the next ~21 trading days for Nifty 50 are “risk-off” (based on forward loss/drawdown thresholds in `config.yaml`).
- **Inputs** (what it learns from): Nifty 50 trend/stress context plus cross-asset signals (gold/silver, USDINR, VIX, US yields) and explicit divergence features.

This probability becomes a penalty so high-risk environments reduce the score even if the index is still trending up.

Important:
- **`risk_off_prob`** is a *trained probability* (learned from history).
- **`riskoff_composite`** is a *hand-built strength gauge* (standardized average of risk-off/divergence signals).
- They can disagree on some days; use them together for context.

## Inputs and tickers (high level)
Configured in `config.yaml`. By default the pipeline uses:

- **Equities**
  - Nifty 50: `^NSEI`
  - Nifty Midcap 100: `^CRSMID` (fallback: `NIFTY_MIDCAP_100.NS`)
  - Smallcap proxy: one of the configured Smallcap 250 ETFs (fallback list)
- **Risk-off assets**
  - Gold: `GC=F`
  - Silver: `SI=F`
  - USD/INR: `INR=X`
  - VIX: `^VIX`
  - US yields: `^TNX` (10Y), `^IRX` (13W/3M proxy)

## How to read `outputs/scores.csv` (what to focus on)
Each row is a date + an index (`index` column).

### Primary decision columns
- **`score`**: the main 0–100 number you track.
- **`label`**: bucketed interpretation of the score (oversold/neutral/overbought etc.).
- **`risk_off_prob`** (0..1): probability the environment looks like historical “risk-off” regimes.
- **`riskoff_composite`**: a standardized composite of risk-off + divergence signals (higher = more risk-off signals firing together).
- **`divergence_state`**: categorical regime/divergence label (more granular than the boolean flag).
- **`divergence_flag`** (True/False): True only for the specific divergence case **trend up + risk-off composite high + Nifty not down on the day**, and **not** a “bear-bounce” day.

### Safe-haven (new) columns
These are **per-asset mean-reversion stretch scores** for safe-haven assets (shared across indices each day):

- **`safe_gold_score` / `safe_gold_label`**
- **`safe_silver_score` / `safe_silver_label`**
- **`safe_usdinr_score` / `safe_usdinr_label`**
- **`safe_us10y_score` / `safe_us10y_label`**
- **`safe_basket_score` / `safe_basket_label`** (average of enabled safe-haven scores)

Interpretation:
- These are **NOT “fair value”**; they indicate **overbought/oversold stretch vs the asset’s own history**.
- Score scale: 0–20 `oversold`, 40–60 `neutral`, 80–100 `overbought`.

How they’re computed (v1):
- Inputs are **RSI** and **rolling z-scores** of **log(price)**:
  - `RSI(rsi_period)` (default 14)
  - `z(log(price), z_windows[0])` (default 60 trading days)
  - `z(log(price), z_windows[1])` (default 1260 trading days \(\approx\) 5 years)
- These components are normalized to \([-1, +1]\), combined using `safe_haven.weights` (default `[0.50, 0.30, 0.20]`), then mapped to 0–100.
- If the long window isn’t available yet, the score uses the available components (so you don’t get a long run of NaNs early on).

You can tune all of this under the `safe_haven:` section in `config.yaml`.

### Supporting columns (why the score is where it is)
- **`trend_score`**: how strong the trend component is (0..60).
- **`reversion_adj`**: overbought/oversold adjustment (-20..+20).
- **`risk_penalty`**: the score penalty (0..20) derived from `risk_off_prob`.

## What `riskoff_composite` means (detailed)
`riskoff_composite` is a **single “pressure” number** built as the average of multiple standardized (rolling z-score) signals. It is designed so that:

- **Higher = more risk-off signals aligning**
- **Lower = more risk-on / supportive signals**

It includes (when the required inputs exist):

- **Gold vs Nifty** (level + 60D momentum): gold outperforming equities is often risk-off.
- **USD/INR stress** (20D + 60D momentum, 20D FX vol): INR weakness/stress often increases in risk-off phases.
- **VIX** (20D + 60D changes): rising fear.
- **US yields** (20D + 60D changes): fast rate moves often coincide with risk/risk-off shifts.
- **US curve slope** (10Y − 3M): curve inversion is treated as risk-off (sign flipped so inversion pushes the composite higher).
- **Intra-equity breadth/leadership**: mid/small vs Nifty 60D relative momentum (mid/small lagging is treated as risk-off; sign flipped).

Interpretation guideline:
- Around **0**: mixed/normal
- Above **+0.75**: clearly risk-off leaning
- Above **+1.5**: strong risk-off backdrop
- Below **−0.75**: clearly risk-on leaning

## What `divergence_state` / `divergence_flag` mean (detailed)
These are designed to separate two different situations that can look similar if you only use medium-term trend:

1) **“Hidden risk under an uptrend”** (divergence)  
2) **“Risk-off selloff already happening”** (not a divergence)

### `divergence_state` values
- **`divergence_trendup_riskoff`**
  - `riskoff_composite > 0.75`
  - Nifty medium-term trend is up (EMA-based)
  - Nifty is **not down on the day**
  - Meaning: price trend still looks healthy, but other asset classes are flashing risk-off.

- **`riskoff_selloff`**
  - `riskoff_composite > 0.75`
  - Nifty is down on the day (even if the longer trend is still up)
  - Meaning: risk-off is showing up *in price already*.

- **`riskoff_bear_bounce`**
  - `riskoff_composite > 0.75`
  - Nifty is **up on the day**, but the **previous day was a heavy down day**
  - Meaning: this is often a **bear market bounce** / reflex rally while risk-off flows persist.

- **`riskoff_crash_day`**
  - `riskoff_composite > 0.75`
  - Nifty is down very sharply (vol-adjusted or > ~2% log drop)
  - Meaning: acute risk-off shock day.

- **`riskoff_downtrend`**
  - `riskoff_composite > 0.75`
  - Nifty is not in “trend up” (EMA-based), and not classified as a down-day selloff

- **`selloff_without_riskoff`**
  - Nifty is down very sharply while `riskoff_composite < -0.75`
  - Meaning: idiosyncratic selloff (not confirmed by typical risk-off assets).

- **`normal`**
  - none of the above triggered.

### `divergence_flag`
This is simply:

- `divergence_flag = (divergence_state == "divergence_trendup_riskoff")`

Use it as a **binary caution marker**.

### Practical reading guide (quick)
- **Bullish + confirmed**: high `score`, low `risk_off_prob`, `riskoff_composite` not rising, `divergence_state = normal`.
- **Caution / late-cycle**: decent `score` but `divergence_flag = True` (hidden risk under trend) or steadily rising `riskoff_composite`.
- **Risk-off selloff**: `divergence_state = riskoff_selloff` or `riskoff_crash_day` (treat as higher risk regardless of medium-term trend).
- **Bear market bounce**: `divergence_state = riskoff_bear_bounce` (equities up after a heavy down day while risk-off flows persist).
- **Bearish regime**: low `score` and elevated `risk_off_prob` (risk management matters most).

## Data window / how much history is downloaded
Controlled by `config.yaml`:
- `data.start` (default: `2015-01-01`)
- `data.end` (default: null = today)

Using `--refresh` downloads the full window and overwrites cached CSVs in `data/cache/`. Without `--refresh`, cached data is reused.

Note on “insufficient_data”:
- Early rows can show `label=insufficient_data` because long rolling windows (e.g., 200D EMA and 252D z-scores) need time to warm up.
- `risk_off_prob` starts only after `risk_model.min_train_years` of history exists.

## Notes
- Data is fetched from Yahoo Finance via `yfinance` and cached to `data/cache/`.
- Smallcap is typically sourced via a liquid Smallcap ETF proxy (configured in `config.yaml`) because Yahoo index tickers can be unreliable.

