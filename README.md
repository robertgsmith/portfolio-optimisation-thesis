# Robust Portfolio Optimization Under Parameter Uncertainty

**Bachelor Thesis Project**  
BSc Computational Business Analytics  
Frankfurt School of Finance & Management

**Authors:** Robert George Smith & Joaquin Rodriguez  
**Supervisors:** Prof. Dr. Grigory Vilkov, Prof. Dr. Paula Cocoma  
**Period:** 2010-2024 (US Market - S&P 100)

---

## Abstract

This thesis investigates portfolio optimization methods that address parameter uncertainty in expected returns and covariances in the US equities market. Classical mean-variance optimization is highly sensitive to estimation error, often producing unstable portfolio weights and poor out-of-sample results. 

We compare traditional mean-variance portfolios with robust alternatives including shrinkage estimators, Bayesian approaches, and robust optimization techniques. As a secondary extension, we test whether monetary policy sentiment extracted from Federal Reserve announcements can enhance short-term allocation decisions.

**Key Contributions:**
- Empirical comparison of robust portfolio optimization methods
- Analysis of portfolio weight stability and turnover
- Integration of Fed sentiment signals for tactical allocation
- Comprehensive backtesting framework with risk-adjusted performance metrics

---

## Research Questions

### Primary Questions
1. **Do robust optimization techniques deliver superior out-of-sample performance compared to classical mean-variance optimization?**
2. **How do robust methods affect portfolio stability (weight dispersion and turnover)?**

### Secondary Questions
3. **How robust are the results across different estimation windows, sample periods, and transaction cost assumptions?**
4. **Does a central bank sentiment indicator provide incremental value for short-term portfolio allocation?**

---

## Project Structure

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
│   ├── data_downloader.py        # Yahoo Finance data extraction
│   ├── data_processor.py         # Returns and signal computation
│   ├── feature_engineer.py       # Feature engineering
│   └── summary_statistics.py     # Statistical analysis
│
├── portfolio_models/              # Portfolio optimization models
│   ├── base_portfolio.py         # Abstract base class
│   ├── mean_variance.py          # Classical MV optimization (benchmark)
│   ├── shrinkage_portfolio.py    # Ledoit-Wolf shrinkage covariance
│   ├── bayesian_portfolio.py     # Bayesian mean estimation (Jorion)
│   └── robust_portfolio.py       # Robust optimization (Bertsimas & Sim)
│
├── backtesting/                   # Backtesting framework
│   ├── backtester.py             # Rolling window backtesting engine
│   └── performance_metrics.py    # Sharpe, Sortino, drawdown, turnover
│
├── sentiment/                     # Fed sentiment analysis
│   ├── fed_scraper.py            # FOMC announcement scraping
│   └── sentiment_analyzer.py     # Lexicon-based sentiment scoring
│
├── utils/                         # Utility functions
│   └── visualization.py          # Plotting and visualization
│
├── results/                       # Output directory
│   ├── figures/                  # Charts and plots
│   └── tables/                   # Performance tables
│
├── notebooks/                     # Jupyter notebooks (exploratory)
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_results_analysis.ipynb
│
├── main.py                        # Main pipeline orchestration
├── config.py                      # Configuration parameters
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

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

### Quick Start

**Run the complete data pipeline:**
```bash
python main.py
```

This will:
1. Download S&P 100 constituent data from Yahoo Finance
2. Determine optimal date range (90% coverage threshold)
3. Compute returns, momentum signals, and rolling statistics
4. Generate expected return estimates and market features
5. Create summary statistics and correlation matrices
6. Save all outputs to `data/` subdirectories

**Expected runtime:** ~15-20 minutes (depending on network speed)

---

## Methodology Overview

### Asset Universe
- **Index:** S&P 100 (OEX components)
- **Period:** 2010-2024
- **Frequency:** Daily data, monthly rebalancing
- **Assets:** ~95 liquid, large-cap US equities

### Portfolio Optimization Methods

| Method | Description | Key Reference |
|--------|-------------|---------------|
| **Mean-Variance (MV)** | Classical Markowitz optimization | Markowitz (1952) |
| **Shrinkage Covariance** | Ledoit-Wolf shrinkage estimator | Ledoit & Wolf (2003) |
| **Bayesian Portfolio** | Bayes-Stein mean estimation | Jorion (1986) |
| **Robust Optimization** | Uncertainty sets for parameters | Bertsimas & Sim (2004) |

### Expected Return Estimation
- Historical mean (expanding window)
- Momentum-based estimates (12-month)
- Combined weighted approach

### Covariance Estimation
- Sample covariance matrix
- Ledoit-Wolf shrinkage
- Robust covariance estimation

### Sentiment Signal (Extension)
- **Source:** Federal Reserve FOMC announcements
- **Method:** Lexicon-based sentiment scoring (Loughran-McDonald)
- **Integration:** Short-term tactical tilts

### Performance Evaluation

**Risk-Adjusted Metrics:**
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio

**Stability Metrics:**
- Portfolio Turnover
- Weight Concentration (Herfindahl Index)
- Weight Stability (correlation between periods)

**Transaction Costs:**
- Base case: 10 basis points
- Robustness: 5 bps, 15 bps, 25 bps

---

## Key Results (Preliminary)

> **Note:** Final results will be updated after complete backtesting.

### Dataset Summary
- **Final Assets:** 95 stocks (after filtering)
- **Date Range:** 2012-01-03 to 2024-12-31
- **Trading Days:** ~3,260 observations
- **Coverage:** 90%+ data availability

### Top Performers (by Sharpe Ratio)
*Based on individual stock performance, 2012-2024*

| Ticker | Ann. Return | Ann. Vol | Sharpe Ratio |
|--------|-------------|----------|--------------|
| NVDA   | TBD         | TBD      | TBD          |
| META   | TBD         | TBD      | TBD          |
| AAPL   | TBD         | TBD      | TBD          |

*Results to be updated after pipeline execution*

---

## Data Sources

| Data Type | Source | Access Method |
|-----------|--------|---------------|
| **Equity Prices** | Yahoo Finance | `yfinance` Python API |
| **Fed Announcements** | Federal Reserve Website | Web scraping |
| **Macro Indicators** | IMF Data / FRED | API (optional) |

**Data Quality Checks:**
- Missing data handling: Forward fill (max 5 days)
- Survivorship bias: Using S&P 100 constituents as of start date
- Outlier detection: Winsorization at 99th percentile

---

## Configuration

Edit `config.py` to customize:

```python
# Date range
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# Estimation parameters
ESTIMATION_WINDOW = 252  # Trading days (1 year)
REBALANCING_FREQUENCY = 21  # Monthly

# Transaction costs
TRANSACTION_COST = 0.001  # 10 basis points

# Risk-free rate
RISK_FREE_RATE = 0.00  # Assuming 0% for Sharpe calculation

# Optimization constraints
MAX_WEIGHT = 0.10  # Maximum 10% per asset
MIN_WEIGHT = 0.00  # No short selling
```

---

## Key References

### Foundational Theory
- **Markowitz, H. (1952).** Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- **Black, F., & Litterman, R. (1992).** Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28-43.

### Estimation Risk
- **Jorion, P. (1986).** Bayes-Stein Estimation for Portfolio Analysis. *Journal of Financial and Quantitative Analysis*, 21(3), 279-292.
- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915-1953.

### Robust Methods
- **Ledoit, O., & Wolf, M. (2003).** Improved Estimation of the Covariance Matrix of Stock Returns with an Application to Portfolio Selection. *Journal of Empirical Finance*, 10(5), 603-621.
- **Bertsimas, D., & Sim, M. (2004).** The Price of Robustness. *Operations Research*, 52(1), 35-53.

### Sentiment Analysis
- **Tetlock, P. (2007).** Giving Content to Investor Sentiment. *Journal of Finance*, 62(3), 1139-1168.
- **Loughran, T., & McDonald, B. (2011).** When is a Liability not a Liability? *Journal of Finance*, 66(1), 35-65.

---

## Workflow

### Phase 1: Data Preparation ✅
1. Download S&P 100 price data
2. Clean and consolidate data
3. Compute returns and signals
4. Engineer features
5. Generate summary statistics

### Phase 2: Model Implementation (In Progress)
1. Implement base portfolio class
2. Code mean-variance benchmark
3. Implement shrinkage covariance
4. Implement Bayesian approach
5. Implement robust optimization

### Phase 3: Backtesting (Upcoming)
1. Set up rolling window framework
2. Run out-of-sample backtests
3. Compute performance metrics
4. Generate comparison tables

### Phase 4: Extension (Upcoming)
1. Scrape Fed announcements
2. Build sentiment scores
3. Integrate into allocation
4. Evaluate incremental value

### Phase 5: Analysis & Writing (Upcoming)
1. Robustness checks
2. Statistical significance tests
3. Results interpretation
4. Thesis writing and revision

---

## Division of Work

### Robert George Smith
- Implement robust portfolio optimization models
- Analyze parameter uncertainty effects
- Evaluate portfolio stability metrics
- Use Bloomberg data for S&P 100 analysis

### Joaquin Rodriguez
- Develop Fed sentiment signal
- Integrate sentiment into allocation
- Evaluate sentiment impact on performance
- Analyze FOMC announcement effects

### Joint Responsibilities
- Theoretical framework development
- Data preprocessing and pipeline
- Backtesting framework design
- Results interpretation and thesis writing

---

## Usage Examples

### Example 1: Run Data Pipeline Only
```python
from data_components import DataDownloader, DataProcessor

# Download data
downloader = DataDownloader(start_date="2010-01-01")
prices, volumes, coverage = downloader.download_all_tickers()

# Process data
processor = DataProcessor()
returns = processor.compute_returns(prices)
```

### Example 2: Compute Portfolio (After Implementation)
```python
from portfolio_models import MeanVariancePortfolio, ShrinkagePortfolio

# Initialize models
mv_model = MeanVariancePortfolio(returns, max_weight=0.10)
shrink_model = ShrinkagePortfolio(returns, max_weight=0.10)

# Compute optimal weights
mv_weights = mv_model.optimize()
shrink_weights = shrink_model.optimize()
```

### Example 3: Run Backtest (After Implementation)
```python
from backtesting import Backtester

# Initialize backtester
backtester = Backtester(
    returns=returns,
    models=[mv_model, shrink_model],
    rebalance_freq=21,
    transaction_cost=0.001
)

# Run backtest
results = backtester.run()
performance = backtester.compute_metrics(results)
```

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'yfinance'`
- **Solution:** Run `pip install -r requirements.txt`

**Issue:** Yahoo Finance download fails for certain tickers
- **Solution:** Script automatically handles this and logs failed tickers. Check logs for details.

**Issue:** Missing data in final dataset
- **Solution:** Script uses forward-fill (max 5 days). If >5 days missing, that period is dropped.

**Issue:** Memory errors with large datasets
- **Solution:** Reduce date range or process in chunks. Consider using `dask` for larger datasets.

---

## Contact

**Robert George Smith**  
Email: robert.smith@student.fs.de  
GitHub: [@robertsmith](https://github.com/robertsmith)

**Joaquin Rodriguez**  
Email: joaquin.rodriguez@student.fs.de  
GitHub: [@joaquinrodriguez](https://github.com/joaquinrodriguez)

---

## License

This project is submitted as part of academic requirements at Frankfurt School of Finance & Management. 

**Academic Use Only** - Not for commercial distribution.

© 2025 Robert George Smith & Joaquin Rodriguez

---

## Acknowledgments

- Prof. Dr. Grigory Vilkov for supervision and guidance
- Prof. Dr. Paula Cocoma for methodological support
- Frankfurt School of Finance & Management for resources and infrastructure
- The open-source community for excellent Python libraries

---

## Project Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Literature Review | Week 1 | Complete |
| Data Collection | Week 2 | Complete |
| Model Implementation | Week 3-4 | In Progress |
| Backtesting | Week 5 | Planned |
| Robustness Checks | Week 6 | Planned |
| Thesis Writing | Week 7-8 | Planned |

**Expected Submission:** [Insert Date]

---

## Version History

- **v0.1.0** (Dec 2025) - Initial project setup and data pipeline
- **v0.2.0** (Jan 2026) - Model implementation (planned)
- **v0.3.0** (Jan 2026) - Backtesting framework (planned)
- **v1.0.0** (Feb 2026) - Final thesis submission (planned)

---

**Last Updated:** December 27, 2025