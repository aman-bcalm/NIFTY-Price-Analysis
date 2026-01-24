# Trend Analyzer (Nifty) — Final Plan (Option A)

## Goal
Build a Python tool that computes a **daily 0–100 score** for:
- **Nifty 50**
- **Nifty Midcap 100**
- **Nifty Smallcap 100**

The score should clearly label each index as **oversold / neutral (“fair” vs trend) / overbought**, and incorporate **cross‑asset risk-off signals** (gold, silver, FX, volatility, and **bond yields**).

Target horizon: **medium-term (1–6 months)**.

## Data (public/free)
### Primary equity data (daily)
Use Yahoo Finance via `yfinance`:
- Nifty 50: `^NSEI`
- Nifty Midcap 100: `^CRSMID` (fallback `NIFTY_MIDCAP_100.NS`)
- Nifty Smallcap 100: `NIFTYSMLCAP100.NS` (fallback: a Smallcap ETF proxy if needed)

### Cross-asset risk-off inputs (daily)
Prefer no-auth Yahoo symbols; keep fallbacks:
- Gold: `GC=F` (optionally convert to INR using USD/INR)
- Silver: `SI=F`
- USD/INR: `INR=X` (or `USDINR=X` depending on provider availability)
- Volatility: `^VIX` (optional: India VIX if available)

### Bond yields (daily, no-auth preference)
US yields (usually available on Yahoo):
- US 10Y: `^TNX`
- US 13W/3M proxy: `^IRX` (if available)

India 10Y yield (try, but may be unreliable via Yahoo):
- Attempt: `IN10Y` / `^IN10Y` (verify in code; if missing, treat as unavailable)
- Fallback (still no-auth): use an **Indian bond ETF price proxy** as a yield-condition signal

## Option A algorithm (final)
The score = **equity health (trend + stretch + stress)** adjusted by a learned **RiskOffProbability** from cross‑asset + rates signals.

### 1) Data prep
- Download 5–10 years daily series.
- Align all series on a common date index.
- Handle missing data with limited forward-fill (e.g., <= 3 days).
- Require minimum history (e.g., >= 3 years) before outputting a score.

### 2) Equity “health” features (computed per index)
Medium-term, explainable features:
- **TrendStrength**
  - EMA(200) slope (annualized)
  - EMA(50) vs EMA(200)
- **DistanceFromTrend**
  - \(d200 = P/EMA200 - 1\) (optionally normalized by volatility)
- **Overbought/Oversold**
  - RSI(14)
  - z-score of price vs 60D rolling mean
- **Stress**
  - drawdown from 252D high
  - realized vol (20D, 60D)

### 3) Cross-asset + bond yield “risk-off” features (shared)
Signals historically associated with equity stress:
- Gold vs equities: z-score of log(Gold / Nifty50)
- Silver vs gold: z-score of log(Silver / Gold)
- FX stress: USD/INR momentum + volatility
- Volatility shock: VIX level + 20D change
- Bond yield stress:
  - US 10Y momentum + z-scored moves
  - Optional curve slope (10Y − 3M or 10Y − 2Y) if available
  - India 10Y momentum if available; otherwise bond-ETF proxy momentum/vol

### 4) Define the “risk-off” target (for training)
Anchor training on **Nifty 50** (most stable index series).

For each day \(t\), define RiskOff(t)=1 if, over the next 21 trading days:
- forward return < −5%, OR
- forward max drawdown worse than −7%

Otherwise RiskOff(t)=0.

### 5) Regime model = RiskOffProbability(t)
Default model: **walk-forward logistic regression** (explainable, stable).
- Train using only data up to time \(t-1\) (no leakage).
- Use rolling/expanding window.
- Output probability \(p(t)\in[0,1]\).

Optional (later): **XGBoost** model as an experiment toggle
- Must still be walk-forward time-series training
- Add probability calibration (XGBoost can be miscalibrated)
- Use SHAP for explanations if needed

### 6) Map to a daily 0–100 score (per index)
Compute:
- **TrendScore** in [0, 60] from trend/stress features
- **ReversionAdjustment** in [−20, +20] from RSI + stretch (overbought reduces, oversold increases)
- **RiskPenalty** in [0, 20] = \(20 * RiskOffProbability(t)\)

Final:
- Score = clamp( TrendScore + 20 + ReversionAdjustment − RiskPenalty, 0, 100 )

### 7) Labels
- 0–20: Oversold / panic
- 20–40: Bearish / below trend
- 40–60: Neutral / “fair” vs trend (statistical)
- 60–80: Bullish / overbought-ish
- 80–100: Euphoric / very overbought

## Implementation deliverables
- **Core library**:
  - Data loader + caching + alignment
  - Feature builder (equity + cross-asset + yields)
  - Regime model (logistic; XGBoost optional)
  - Score engine (0–100 + label + driver breakdown)
- **Outputs**:
  - CSV export of scores + features + RiskOffProbability
  - Simple dashboard (later) showing history and “why score changed”
- **Backtest / calibration**:
  - Score deciles vs forward returns/drawdowns
  - Tune RiskOff thresholds (−5%, −7%) and score weights if needed

## Suggested project structure
- `trend_analyzer/data/`
- `trend_analyzer/features/`
- `trend_analyzer/models/`
- `trend_analyzer/scoring/`
- `trend_analyzer/app/`
- `config.yaml`

