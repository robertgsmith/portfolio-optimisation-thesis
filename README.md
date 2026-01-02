# Robust Portfolio Optimisation Under Parameter Uncertainty

**Bachelor Thesis Project**  
BSc Computational Business Analytics  
Frankfurt School of Finance & Management

**Authors:** Robert George Smith & Joaquin Rodriguez  
**Supervisors:** Prof. Dr. Grigory Vilkov, Prof. Dr. Paula Cocoma  
**Period:** 2010-2024 (US Market - S&P 100)

---

## 📋 Abstract

This thesis investigates portfolio optimisation methods that address parameter uncertainty in expected returns and covariances in the US equities market. Classical mean-variance optimisation is highly sensitive to estimation error, often producing unstable portfolio weights and poor out-of-sample results. 

We compare traditional mean-variance portfolios with robust alternatives including shrinkage estimators, Bayesian approaches, and robust optimisation techniques. As a secondary extension, we test whether monetary policy sentiment extracted from Federal Reserve announcements can enhance short-term allocation decisions.

**Key Contributions:**
- Empirical comparison of robust portfolio optimisation methods
- Analysis of portfolio weight stability and turnover
- Comprehensive backtesting framework with transaction costs
- Statistical significance testing of performance differences

---

## 🎯 Research Questions

### Primary Questions
1. **Do robust optimisation techniques deliver superior out-of-sample performance compared to classical mean-variance optimisation?**
2. **How do robust methods affect portfolio stability (weight dispersion and turnover)?**

### Secondary Questions
3. **How robust are the results across different estimation windows, sample periods, and transaction cost assumptions?**
4. **Does a central bank sentiment indicator provide incremental value for short-term portfolio allocation?** *(Extension - Optional)*

---

## 📁 Project Structure

```
portfolio-optimiser-thesis/
│
├── data/                          # Data storage
│   ├── raw/                       # Raw price and volume data
│   ├── processed/                 # Returns, rolling statistics
│   ├── features/                  # Engineered features
│   └── analysis/                  # Summary statistics, correlations
│
├── data_components/               # Data pipeline modules
│   ├── data_downloader.py         # Yahoo Finance data extraction
│   ├── data_processor.py          # Returns and signal computation
│   ├── feature_engineer.py        # Feature engineering
│   └── summary_statistics.py      # Statistical analysis
│
├── portfolio_models/              # Portfolio optimisation models
│   ├── base_portfolio.py          # Abstract base class
│   ├── mean_variance.py           # Classical MV optimisation (benchmark)
│   ├── shrinkage_portfolio.py     # Ledoit-Wolf shrinkage covariance
│   ├── bayesian_portfolio.py      # Bayesian mean estimation (Jorion)
│   ├── robust_portfolio.py        # Robust optimisation (Bertsimas & Sim)
│   ├── equal_weight.py            # 1/N benchmark (DeMiguel et al.)
│   └── solver_utils.py            # CVXPY solver utilities
│
├── backtesting/                   # Backtesting framework
│   ├── backtester.py              # Rolling window backtesting engine
│   └── performance_metrics.py     # Sharpe, Sortino, drawdown, turnover
│
├── sentiment/                     # Fed sentiment analysis (optional)
│   ├── fed_scraper.py             # FOMC announcement scraping
│   └── sentiment_analyzer.py      # Lexicon-based sentiment scoring
│
├── results/                       # Output directory
│   ├── figures/                   # Charts and plots
│   ├── tables/                    # Performance tables
│   └── weights/                   # Portfolio weight histories
│
├── main.py                        # Data pipeline orchestration
├── run_backtest.py                # Execute backtesting
├── visualise_results.py           # Generate figures for thesis
├── statistical_analysis.py        # Statistical significance tests
├── robustness_checks.py           # Robustness analysis
├── config.py                      # Configuration parameters
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- ~2GB free disk space for data

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/robertgsmith/portfolio-optimiser-thesis.git
   cd portfolio-optimiser-thesis
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install optimisation solvers:**
   ```bash
   pip install scs osqp
   ```

### Quick Start

**Step 1: Run the complete data pipeline**
```bash
python main.py
```

This will:
- Download S&P 100 constituent data from Yahoo Finance (2010-2024)
- Determine optimal date range (90% coverage threshold)
- Compute returns, momentum signals, and rolling statistics
- Generate expected return estimates and market features
- Create summary statistics and correlation matrices
- Save all outputs to `data/` subdirectories

**Expected runtime:** ~15-20 minutes

**Step 2: Run backtesting**
```bash
python run_backtest.py
```

This will:
- Test all 5 portfolio optimisation models
- Use rolling window (252-day estimation, monthly rebalancing)
- Apply transaction costs (10 basis points)
- Save results to `results/`

**Expected runtime:** ~2-3 minutes

**Step 3: Generate visualizations**
```bash
python visualise_results.py
```

Creates thesis-ready figures in `results/figures/`

**Step 4: Statistical analysis**
```bash
python statistical_analysis.py
```

Tests statistical significance of performance differences

**Step 5: Robustness checks**
```bash
python robustness_checks.py
```

Tests sensitivity to parameters (transaction costs, estimation windows, sub-periods)

---

## 📊 Methodology Overview

### Asset Universe
- **Index:** S&P 100 (OEX components)
- **Period:** 2010-2024 (filtered to 2012-2024 for 90% coverage)
- **Frequency:** Daily data, monthly rebalancing
- **Assets:** 92 liquid, large-cap US equities

### Portfolio optimisation Methods

| Method | Description | Key Reference |
|--------|-------------|---------------|
| **Mean-Variance (MV)** | Classical Markowitz optimisation | Markowitz (1952) |
| **Shrinkage Covariance** | Ledoit-Wolf shrinkage estimator | Ledoit & Wolf (2003) |
| **Bayesian Portfolio** | Bayes-Stein mean estimation | Jorion (1986) |
| **Robust optimisation** | Uncertainty sets for parameters | Bertsimas & Sim (2004) |
| **Equal Weight (1/N)** | Naive diversification benchmark | DeMiguel et al. (2009) |

### Backtesting Framework
- **Estimation Window:** 252 trading days (~1 year)
- **Rebalancing Frequency:** 21 trading days (monthly)
- **Transaction Costs:** 10 basis points (with robustness checks at 5, 15, 25 bps)
- **Initial Capital:** $1,000,000
- **Constraints:** Max 10% per asset, no short selling

### Expected Return Estimation
- Historical mean (annualised from estimation window)
- Sample covariance matrix (annualised)
- Shrinkage covariance (Ledoit-Wolf)
- Bayesian shrinkage of means (Jorion)

### Performance Evaluation

**Risk-Adjusted Metrics:**
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio

**Stability Metrics:**
- Portfolio Turnover
- Weight Concentration (Herfindahl Index)
- Effective Number of Assets

**Statistical Tests:**
- Paired t-tests for mean returns
- Bootstrap tests for Sharpe ratio differences
- Levene's test for volatility equality
- Drawdown comparison

---

## 📈 Results Summary

### Dataset Characteristics
- **Final Assets:** 92 stocks (after filtering)
- **Date Range:** 2012-01-03 to 2024-12-30
- **Trading Days:** 3,773 observations
- **Backtesting Periods:** 179 rebalancing periods
- **Coverage:** 92.9% of S&P 100

### Performance Overview

All models tested over 13-year out-of-sample period (2012-2024) with monthly rebalancing and 10bp transaction costs.

**Key Findings:**
1. **Robust methods reduce portfolio volatility** compared to classical mean-variance
2. **Shrinkage methods show lower turnover** - reduced trading costs
3. **Equal weight benchmark is competitive** - validates DeMiguel et al. (2009) findings
4. **Statistical significance varies** - bootstrap tests confirm some differences

*See `results/backtest_metrics.csv` for complete performance metrics*

### Generated Outputs

After running the pipeline, you'll have:

**Data Files:**
- `data/raw/sp100_prices.csv` - Historical prices
- `data/processed/log_returns.csv` - Daily returns
- `data/features/expected_returns_*.csv` - Return estimates
- `data/analysis/return_statistics.csv` - Summary statistics

**Backtest Results:**
- `results/backtest_returns.csv` - Daily portfolio returns
- `results/backtest_metrics.csv` - Performance metrics table
- `results/backtest_cumulative_returns.csv` - Cumulative performance
- `results/weights/*.csv` - Portfolio weight histories

**Statistical Tests:**
- `results/test_mean_returns.csv` - Mean return comparisons
- `results/test_sharpe_ratios.csv` - Sharpe ratio bootstrap tests
- `results/test_volatility.csv` - Volatility equality tests
- `results/test_turnover.csv` - Portfolio stability analysis

**Visualizations:**
- `results/figures/cumulative_returns.png` - Performance over time
- `results/figures/drawdowns.png` - Drawdown comparison
- `results/figures/risk_return_scatter.png` - Risk-return profile
- `results/figures/performance_metrics.png` - Bar chart comparison

---

## ⚙️ Configuration

Edit `config.py` to customise parameters:

```python
# Date range
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# Backtesting parameters
ESTIMATION_WINDOW = 252           # Trading days (1 year)
REBALANCING_FREQUENCY = 21        # Monthly rebalancing
TRANSACTION_COST = 0.001          # 10 basis points

# Portfolio constraints
MAX_WEIGHT = 0.10                 # Maximum 10% per asset
MIN_WEIGHT = 0.00                 # No short selling

# Risk parameters
RISK_FREE_RATE = 0.00             # For Sharpe ratio calculation
RISK_AVERSION_DEFAULT = 1.0       # Risk aversion parameter

# Robust optimisation
ROBUST_EPSILON = 0.5              # Uncertainty set size
UNCERTAINTY_SET_SIZE = 0.5        # Parameter uncertainty

# optimisation solvers
CVXPY_SOLVER = "SCS"              # Primary solver (SCS, ECOS, OSQP)
CVXPY_SOLVER_FALLBACK = "OSQP"   # Backup solver
```

---

## 📚 Key References

### Foundational Theory
- **Markowitz, H. (1952).** Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- **Black, F., & Litterman, R. (1992).** Global Portfolio optimisation. *Financial Analysts Journal*, 48(5), 28-43.

### Estimation Risk
- **Jorion, P. (1986).** Bayes-Stein Estimation for Portfolio Analysis. *Journal of Financial and Quantitative Analysis*, 21(3), 279-292.
- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915-1953.

### Robust Methods
- **Ledoit, O., & Wolf, M. (2003).** Improved Estimation of the Covariance Matrix of Stock Returns with an Application to Portfolio Selection. *Journal of Empirical Finance*, 10(5), 603-621.
- **Bertsimas, D., & Sim, M. (2004).** The Price of Robustness. *Operations Research*, 52(1), 35-53.
- **Goldfarb, D., & Iyengar, G. (2003).** Robust Portfolio Selection Problems. *Mathematics of Operations Research*, 27(1), 1-38.

### Sentiment Analysis (Extension)
- **Tetlock, P. (2007).** Giving Content to Investor Sentiment. *Journal of Finance*, 62(3), 1139-1168.
- **Loughran, T., & McDonald, B. (2011).** When is a Liability not a Liability? *Journal of Finance*, 66(1), 35-65.
- **Bernanke, B., & Kuttner, K. (2005).** What Explains the Stock Market's Reaction to Federal Reserve Policy? *Journal of Finance*, 60(3), 1221-1257.

---

## 🔄 Complete Workflow

### ✅ Phase 1: Data Preparation (COMPLETE)
1. ✅ Download S&P 100 price data from Yahoo Finance
2. ✅ Clean and consolidate data (forward fill, filter dates)
3. ✅ Compute returns and signals (momentum, volatility)
4. ✅ Engineer features (expected returns, market features)
5. ✅ Generate summary statistics and correlation matrices

### ✅ Phase 2: Model Implementation (COMPLETE)
1. ✅ Implement abstract base portfolio class
2. ✅ Code mean-variance benchmark
3. ✅ Implement shrinkage covariance (Ledoit-Wolf)
4. ✅ Implement Bayesian approach (Jorion)
5. ✅ Implement robust optimisation (Bertsimas & Sim)
6. ✅ Implement equal weight (1/N) benchmark

### ✅ Phase 3: Backtesting (COMPLETE)
1. ✅ Set up rolling window framework (252-day estimation)
2. ✅ Run out-of-sample backtests (monthly rebalancing)
3. ✅ Apply transaction costs (10 basis points)
4. ✅ Compute performance metrics (Sharpe, Sortino, drawdown, turnover)
5. ✅ Generate comparison tables and save results

### ✅ Phase 4: Analysis (COMPLETE)
1. ✅ Create visualizations (cumulative returns, drawdowns, risk-return)
2. ✅ Statistical significance tests (bootstrap Sharpe, t-tests)
3. ✅ Robustness checks (transaction costs, estimation windows, sub-periods)
4. ✅ Generate thesis-ready tables and figures

### 📝 Phase 5: Extension (OPTIONAL)
1. ⏸️ Scrape Fed FOMC announcements
2. ⏸️ Build sentiment scores (Loughran-McDonald lexicon)
3. ⏸️ Integrate into portfolio allocation
4. ⏸️ Evaluate incremental value of sentiment signals

*Note: Fed sentiment analysis is optional and can be cited as future work if time is limited*

### 📖 Phase 6: Thesis Writing (IN PROGRESS)
1. 📝 Write Introduction chapter
2. 📝 Write Literature Review
3. ✅ Write Methodology (documented in code)
4. ✅ Write Results (tables and figures ready)
5. 📝 Write Discussion and robustness analysis
6. 📝 Write Conclusion

---

## 👥 Division of Work

### Robert George Smith
- ✅ Implement robust portfolio optimisation models
- ✅ Analyze parameter uncertainty effects
- ✅ Evaluate portfolio stability metrics
- ✅ Backtesting framework implementation
- 📝 Write methodology and results chapters

### Joaquin Rodriguez
- ⏸️ Develop Fed sentiment signal (optional extension)
- ⏸️ Integrate sentiment into allocation
- ⏸️ Evaluate sentiment impact on performance
- 📝 Literature review on sentiment analysis

### Joint Responsibilities
- ✅ Theoretical framework development
- ✅ Data preprocessing and pipeline
- ✅ Backtesting framework design
- ✅ Statistical analysis
- 📝 Results interpretation and thesis writing

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'cvxpy'`
- **Solution:** Run `pip install -r requirements.txt`

**Issue:** `The solver ECOS is not installed`
- **Solution:** Install solvers: `pip install scs osqp ecos`
- **Alternative:** Update `config.py` to use `CVXPY_SOLVER = "SCS"`

**Issue:** All models show identical results
- **Solution:** optimisation failed - check solver installation
- **Verify:** Run `python -c "import cvxpy as cp; print(cp.installed_solvers())"`

**Issue:** Yahoo Finance download fails
- **Solution:** Script automatically handles failures and logs them. Check logs for details.
- **Retry:** Delete `data/raw/` and run `python main.py` again

**Issue:** Missing data in returns
- **Solution:** Script uses forward-fill (max 5 days). Longer gaps are dropped.
- **Check:** Review `data/processed/log_returns.csv` for completeness

**Issue:** Memory errors during backtesting
- **Solution:** Reduce date range in `config.py` or increase system RAM

---

## 📊 Data Sources

| Data Type | Source | Access Method | Frequency |
|-----------|--------|---------------|-----------|
| **Equity Prices** | Yahoo Finance | `yfinance` API | Daily |
| **Trading Volume** | Yahoo Finance | `yfinance` API | Daily |
| **Fed Announcements** | Federal Reserve Website | Web scraping | Event-based |
| **Risk-Free Rate** | Assumed 0% | Configuration | Annual |

**Data Quality:**
- Missing data: Forward fill (max 5 days)
- Survivorship bias: Minimal (using current S&P 100 constituents)
- Coverage requirement: 90% of tickers must have data from start date
- Outliers: Handled through portfolio constraints (max 10% weight)

---

## 📧 Contact

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

## 📄 License

This project is submitted as part of academic requirements at Frankfurt School of Finance & Management. 

**Academic Use Only** - Not for commercial distribution.

© 2026 Robert George Smith & Joaquin Rodriguez

---

## 🙏 Acknowledgments

- Prof. Dr. Grigory Vilkov for supervision and guidance on portfolio theory
- Prof. Dr. Paula Cocoma for methodological support and feedback
- Frankfurt School of Finance & Management for resources and infrastructure
- The open-source community for excellent Python libraries (NumPy, Pandas, CVXPY, scikit-learn)
- DeMiguel et al. (2009) for inspiring the equal-weight benchmark

---

## 📅 Project Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Literature Review | Week 1 | ✅ Complete |
| Data Collection | Week 2 | ✅ Complete |
| Model Implementation | Week 3-4 | ✅ Complete |
| Backtesting Framework | Week 5 | ✅ Complete |
| Statistical Analysis | Week 5 | ✅ Complete |
| Robustness Checks | Week 6 | ✅ Complete |
| Thesis Writing | Week 7-8 | 📝 In Progress |

**Expected Submission:** February 2026

---

## 📝 Version History

- **v0.1.0** (Dec 2025) - Initial project setup and data pipeline
- **v0.2.0** (Jan 2026) - Portfolio models implementation
- **v0.3.0** (Jan 2026) - Backtesting framework complete
- **v0.4.0** (Jan 2026) - Statistical analysis and visualizations
- **v1.0.0** (Feb 2026) - Final thesis submission

---

## 🎓 Citation

If you use this code or methodology, please cite:

```bibtex
@thesis{smith2026robust,
  title={Robust Portfolio optimisation Under Parameter Uncertainty},
  author={Smith, Robert George and Rodriguez, Joaquin},
  year={2026},
  school={Frankfurt School of Finance \& Management},
  type={Bachelor's Thesis},
  note={BSc Computational Business Analytics}
}
```

---

**Last Updated:** January 2, 2026