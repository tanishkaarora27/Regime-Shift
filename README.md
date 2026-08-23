

# RegimeShift

Does an ML-predicted **volatility regime** improve a tactical SPY / TLT / GLD allocation?

A single-notebook experiment that predicts whether SPY’s realized volatility over the next 5 trading days will land in the top third of its historical distribution, and uses that signal to switch between growth (70/20/10) and defensive (25/55/20) weights. The target is **volatility, not direction** — nothing here forecasts returns.

## What makes this backtest honest

Most retail ML backtests fail in the same few places. This one is structured around avoiding them:

- **Three chronological chunks, strictly separated.** Model selection (5 classifiers, TimeSeriesSplit AUC) happens only on the first ~60% of data. The winner is walked forward on an expanding window through the next ~25%. The final ~15% is touched exactly once — one retrain, one prediction pass, final metrics. No decision anywhere is made by looking at holdout performance.
- **Embargo everywhere the label leaks.** The target looks 5 days ahead, so the last 5 rows before every fit boundary are dropped — inside CV folds, before each walk-forward refit, and before the holdout fit.
- **Expanding tercile threshold.** The “top third” label threshold at date *t* uses only vol history available at *t*. A full-sample threshold would leak the future distribution into early labels.
- **Signal lag + costs.** Yesterday’s prediction sets today’s weights (`shift(1)`); 5 bps per unit of one-way turnover on every rebalance; total turnover reported as its own number.
- **Predictive quality and economic value reported separately.** Table 1 gives classification diagnostics (score-based AUC, precision, recall, confusion matrices, base rates); Table 2 gives portfolio metrics. A classifier can rank regimes well and still fail to make money — conflating the two hides which one failed.
- **The central benchmark is a no-ML vol rule** (20d realized vol vs. its expanding median, same weights, lag, and costs). Beating buy-and-hold is not evidence of ML skill; adding value beyond simple volatility persistence would be. Static 60/30/10, equal-weight, and buy-and-hold SPY round out the comparisons.
- **Robustness checks**: a 500-permutation shuffled-signal test with an explicit empirical p-value, and a lagged-realized-regime test (does the model beat yesterday’s realized regime used as the signal?).
- **Negative findings are preserved.** No weights, thresholds, features, or model choices are revisited after seeing walk-forward or holdout results. The pre-specified claim under test: volatility regimes may be partially predictable out of sample, but whether that predictability delivers economic improvement over simple rules — after turnover and costs — is a separate question, and the answer stands whichever way it comes out.

## Quick start

**Colab (zero setup):** upload `RegimeShift_Notebook.ipynb` to Colab and run all cells.

**Local:**

```bash
pip install -r requirements.txt
jupyter lab RegimeShift_Notebook.ipynb
```

Data downloads automatically via yfinance (SPY, TLT, GLD, ^VIX, 2007–today). Set `USE_SYNTHETIC = True` in the config cell to run the pipeline on simulated 2-state Markov regime data instead — useful for verifying the machinery without a network connection.

## Outputs

Equity curves (ML strategy vs. all four baselines, walk-forward and holdout plotted separately), a CAGR / Sharpe / MaxDD / Turnover table per chunk, feature importances from the final model, and the two robustness plots. Save exported charts to `outputs/`.

## Disclaimer

One asset trio, one cost assumption, fixed weight pairs, default hyperparameters. This is a structured experiment in backtest hygiene, not investment advice.
