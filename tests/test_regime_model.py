import numpy as np
import pandas as pd

from trend_analyzer.regime_model import RiskModelConfig, walkforward_logistic_probabilities


def test_walkforward_logistic_outputs_probabilities():
    idx = pd.date_range("2018-01-01", periods=900, freq="B")
    rng = np.random.default_rng(0)

    # Synthetic features with some signal
    x1 = rng.normal(size=len(idx))
    x2 = rng.normal(size=len(idx))
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)

    # Create y with both classes and some relationship to x1
    logits = 0.8 * x1 - 0.2 * x2
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=len(idx)) < p).astype(int)
    y = pd.Series(y, index=idx, dtype="float64")

    cfg = RiskModelConfig(
        horizon_days=21,
        fwd_return_threshold=-0.05,
        fwd_max_drawdown_threshold=-0.07,
        min_train_years=1,
        retrain_frequency="ME",
        regularization_C=1.0,
    )

    probs = walkforward_logistic_probabilities(X=X, y=y, cfg=cfg)
    assert probs.index.equals(idx)
    finite = probs.dropna()
    assert len(finite) > 0
    assert (finite >= 0).all()
    assert (finite <= 1).all()

