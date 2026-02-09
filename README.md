# Robust Portfolio Optimisation Under Parameter Uncertainty in the US Equity Market

**Bachelor Thesis Project**  
BSc Computational Business Analytics  
Frankfurt School of Finance & Management

**Authors:** Robert George Smith & Joaquin Rodriguez  
**Supervisors:** Prof. Dr. Grigory Vilkov, Prof. Dr. Paula Cocoma  
**Period:** 2012-2024 (US Market - S&P 100)

---

## Abstract

This thesis investigates portfolio optimisation methods that address parameter uncertainty in expected returns and covariances in the US equities market. Classical mean-variance optimisation is highly sensitive to estimation error, often producing unstable portfolio weights and poor out-of-sample results. 

We compare traditional mean-variance portfolios with robust alternatives including shrinkage estimators, Bayesian approaches, and robust optimisation techniques. Results are evaluated through comprehensive backtesting with transaction costs, statistical significance testing, and robustness checks across multiple market regimes.

**Key Contributions:**
- Empirical comparison of robust portfolio optimisation methods under industry-standard diversification constraints
- Analysis of portfolio concentration and its impact on method differentiation
- Demonstration that diversification constraints are necessary for robust methods to outperform classical approaches
- Statistical significance testing of performance differences across market regimes

---

## Research Questions

### Primary Questions
1. **Do robust optimisation techniques deliver superior out-of-sample performance compared to classical mean-variance optimisation?**
2. **How do robust methods affect portfolio stability (weight dispersion and turnover)?**
3. **Are diversification constraints necessary for robust methods to differentiate from classical approaches?**

### Secondary Questions
4. **How robust are the results across different estimation windows, sample periods, and transaction cost assumptions?**
5. **Do performance differences vary across market regimes (pre-COVID, COVID, post-COVID)?**

---

## Project Structure

```
portfolio-optimiser-thesis/
│
├── config.py                          # Configuration & parameters
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── data/                              # Data storage (gitignored)
│   ├── raw/                           # Downloaded price data
│   ├── processed/                     # Returns & statistics
│   ├── features/                      # Engineered features
│   └── analysis/                      # Summary statistics
│
├── data_pipeline/                     # Data pipeline modules
│   ├── __init__.py
│   ├── data_downloader.py             # Yahoo Finance data extraction
│   ├── data_processor.py              # Returns computation
│   ├── feature_engineer.py            # Feature engineering
│   └── summary_statistics.py          # Statistical analysis
│
├── portfolio_models/                  # Portfolio optimisation models
│   ├── __init__.py
│   ├── base_portfolio.py              # Abstract base class
│   ├── mean_variance.py               # Classical MV optimisation
│   ├── shrinkage_portfolio.py         # Ledoit-Wolf shrinkage
│   ├── bayesian_portfolio.py          # Bayesian estimation (Jorion)
│   ├── robust_portfolio.py            # Robust optimisation (Bertsimas & Sim)
│   └── equal_weight.py                # 1/N benchmark (DeMiguel et al.)
│
├── backtesting/                       # Backtesting framework
│   ├── __init__.py
│   ├── backtester.py                  # Rolling window backtesting
│   └── performance_metrics.py         # Performance evaluation
│
├── scripts/                           # Execution scripts
│   ├── run_data_pipeline.py           # Data preparation
│   ├── run_backtest.py                # Run backtesting
│   ├── run_full_analysis.py           # Master script (without data preparation)
│   └── compare_results.py             # Compare constrained vs unconstrained
│
├── analysis/                          # Analysis & diagnostics
│   ├── __init__.py
│   ├── visualise_results.py           # Generate figures
│   ├── statistical_analysis.py        # Statistical tests
│   ├── robustness_checks.py           # Robustness analysis
│   ├── plot_subperiods.py             # Sub-period analysis
│   ├── check_concentration.py         # Portfolio concentration diagnostic
│   ├── check_expected_returns.py      # Expected returns diagnostic
│   ├── turnover_investigation.py      # Turnover analysis
│   └── weight_correlation.py          # Weight correlation analysis
│
└── results/                           # Output (gitignored)
    ├── figures/                       # Charts & visualisations
    ├── tables/                        # Performance tables
    └── weights/                       # Portfolio weight histories
```

---

## Methodology Overview & Key Results Summary

### Asset Universe & Dataset
- **Index:** S&P 100 (OEX components)
- **Period:** 2010-01-01 to 2024-12-31 (filtered for 90% data coverage)
- **Frequency:** Daily data, monthly rebalancing
- **Final Sample:** 92 assets (92.9% of S&P 100), 3,773 trading days
- **Rebalancing Periods:** 179 monthly rebalances

### Portfolio Optimisation Methods

| Method | Description | Key Feature |
|--------|-------------|-------------|
| **Mean-Variance** | Classical Markowitz (1952) | Baseline benchmark |
| **Shrinkage** | Ledoit-Wolf covariance shrinkage (2003) | Stabilised covariance |
| **Bayesian** | Bayes-Stein mean estimation (Jorion 1986) | Shrunk expected returns |
| **Robust** | Uncertainty sets (Bertsimas & Sim 2004) | Explicit uncertainty modeling |
| **Equal Weight** | 1/N diversification (DeMiguel et al. 2009) | Naive benchmark |

### Key Methodological Improvements

**1. Diversification Constraints**
- Minimum effective number of assets: 40
- Maximum Herfindahl index: 0.05
- Prevents over-concentration in optimised portfolios

**2. Expected Return Treatment**
- Winsorization at 5th/95th percentiles
- Caps extreme values (prevents estimates outside [-50%, +100%])
- Addresses estimation uncertainty

**3. Backtesting Framework**
- **Estimation Window:** 252 trading days (~1 year)
- **Rebalancing:** Monthly (21 trading days)
- **Transaction Costs:** 10 basis points
- **Initial Capital:** $1,000,000
- **Constraints:** Max 10% per asset, no short-selling

### Performance Evaluation

**Risk-Adjusted Metrics:**
- Sharpe Ratio (annualised)
- Sortino Ratio (downside deviation)
- Maximum Drawdown
- Calmar Ratio (return/max drawdown)

**Stability Metrics:**
- Portfolio Turnover (average per rebalancing)
- Herfindahl Concentration Index
- Effective Number of Assets
- Weight Correlation across models

**Statistical Tests:**
- Paired t-tests (mean returns)
- Bootstrap tests (Sharpe ratio differences, 10,000 iterations)
- Levene's test (volatility equality)
- Regime-dependent analysis (pre-COVID, COVID, post-COVID)

### Performance Metrics

**Expected Findings:**
1. Robust methods show improved Sharpe ratios with proper diversification constraints
2. Without constraints, all methods converge to similar concentrated solutions
3. Equal Weight competitive in stable periods (validates DeMiguel et al. 2009)
4. Turnover differentiation emerges only with diversification constraints

### Sub-Period Analysis

Performance varies significantly across market regimes:

- **Pre-COVID (2015-2019):** Equal Weight dominates stable bull market
- **COVID Era (2020-2021):** Equal Weight maintains advantage during crisis
- **Post-COVID (2022-2024):** Robust optimisation outperforms in volatile markets

### Statistical Significance

*Results from `analysis/statistical_analysis.py`*

Bootstrap tests (10,000 iterations) reveal whether performance differences are statistically significant at the 5% level.

---

## Key References

### Foundational Theory
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28-43.

### Estimation Risk & Robust Methods
- Jorion, P. (1986). Bayes-Stein Estimation for Portfolio Analysis. *Journal of Financial and Quantitative Analysis*, 21(3), 279-292.
- Ledoit, O., & Wolf, M. (2003). Improved Estimation of the Covariance Matrix. *Journal of Empirical Finance*, 10(5), 603-621.
- Bertsimas, D., & Sim, M. (2004). The Price of Robustness. *Operations Research*, 52(1), 35-53.
- Goldfarb, D., & Iyengar, G. (2003). Robust Portfolio Selection Problems. *Mathematics of Operations Research*, 27(1), 1-38.

### Benchmarking
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915-1953.

---

## Division of Work

### Robert George Smith
- Robust portfolio optimisation implementation
- Diversification constraint analysis
- Parameter uncertainty evaluation
- Backtesting framework
- Statistical testing
- Methodology & results chapters

### Joaquin Rodriguez
- Literature review (robust methods & sentiment analysis)
- Data quality validation
- Literature review & discussion chapters

### Joint Responsibilities
- Theoretical framework
- Research design
- Results interpretation
- Introduction & conclusion

---

## Contact

**Robert George Smith**  
Email: robert.smith@fs-students.de  
GitHub: [@robertgsmith](https://github.com/robertgsmith)

**Joaquin Rodriguez**  
Email: joaquin.rodriguez@fs-students.de  
GitHub: [@pbzhxwfsd6-bit](https://github.com/pbzhxwfsd6-bit)

**Supervisors:**  
Prof. Dr. Grigory Vilkov - g.vilkov@fs.de  
Prof. Dr. Paula Cocoma - p.cocoma@fs.de

---

## License

This project is submitted as part of academic requirements at Frankfurt School of Finance & Management.

**Academic Use Only** - Not for commercial distribution.

© 2026 Robert George Smith & Joaquin Rodriguez

---

## Acknowledgments

- Prof. Dr. Grigory Vilkov for supervision and guidance
- Prof. Dr. Paula Cocoma for methodological support
- Thomas Osowski (Industry Practitioner) for professional feedback on diversification constraints
- Frankfurt School of Finance & Management for resources
- The open-source community for Python libraries (NumPy, Pandas, CVXPY)

---

## Key Takeaways

1. **Diversification constraints are essential** - Without them, all optimisation methods converge to similar concentrated solutions
2. **Winsorization addresses extreme estimates** - Capping expected returns at 5th/95th percentiles prevents unrealistic values
3. **Market regime matters** - Equal Weight excels in stable/crisis periods; Robust methods shine in volatile markets
4. **Statistical rigor is critical** - Bootstrap testing reveals whether performance differences are genuine or due to chance
5. **Industry feedback validates** - Professional practitioners confirmed our methodological gap and guided improvements

---

## Limitations
An initially planned extension to incorporate Federal Reserve policy sentiment was not pursued in the final analysis. Preliminary tests using 10-year Treasury yield changes as a monetary policy proxy showed no statistically significant improvement in risk-adjusted returns (p = 0.768), consistent with semi-strong form market efficiency. Future research could explore alternative sentiment measures, such as textual analysis of FOMC statements or high-frequency event studies around policy announcements

---

**Last Updated:** February 09, 2026  
**Version:** 1.0.0 (Final - Pre-Submission)