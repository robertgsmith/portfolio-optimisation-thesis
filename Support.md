# Quickstart Overview
This Document (not in README.md) contains:
- Getting Started
- Quick Start
- Troubleshooting
- Configuration Overview
- Generated Outputs

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- ~1GB free disk space for data

### Installation

1. **Clone or download the repository:**
   ```bash
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

4. **Verify installation:**
   ```bash
   python -c "import cvxpy as cp; print(f'Solvers: {cp.installed_solvers()}')"
   ```

### Quick Start

**Complete Pipeline (20-25 minutes):**
```bash
# Run everything
python main.py
```

**Step-by-Step Execution:**
```bash
# Step 1: Prepare data (5-15 minutes)
python scripts/run_data_pipeline.py

# Step 2: Run backtesting (2-3 minutes)
python scripts/run_backtest.py

# Step 3: Generate visualisations (1 minute)
python analysis/visualise_results.py

# Step 4: Statistical tests (1 minute)
python analysis/statistical_analysis.py

# Step 5: Robustness checks (5 minutes)
python analysis/robustness_checks.py

# Step 6: Sub-period analysis (1 minute)
python analysis/plot_subperiods.py
```

---

## Troubleshooting

### Common Issues

**Q: Solver errors (ECOS not installed)**  
A: Install additional solvers:
```bash
pip install scs osqp ecos
```

**Q: All models show identical results**  
A: Check that diversification constraints are enabled in `config.py`:
```python
ENABLE_DIVERSIFICATION = True
MIN_EFFECTIVE_ASSETS = 40
```

**Q: Herfindahl warnings in concentration check**  
A: If Herfindahl = 0.0500 exactly, this is **correct** - you're at the constraint boundary. The warnings are overly strict for values exactly at 0.05.

**Q: Missing data errors**  
A: Run data pipeline first:
```bash
python scripts/run_data_pipeline.py
```

**Q: Import errors in analysis scripts**  
A: Ensure all scripts are in correct folders with proper import fixes at the top of each file.

---

## Configuration

Edit `config.py` to customise parameters:

```python
# Date Range
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# Backtesting Parameters
ESTIMATION_WINDOW = 252           # 1 year
REBALANCING_FREQUENCY = 21        # Monthly
TRANSACTION_COST = 0.001          # 10 basis points

# Portfolio Constraints
MAX_WEIGHT = 0.10                 # Max 10% per asset
MIN_WEIGHT = 0.00                 # No short-selling

# Diversification Constraints (Industry Best Practice)
MIN_EFFECTIVE_ASSETS = 40         # Minimum 40 effective assets
ENABLE_DIVERSIFICATION = True     # Toggle for comparison

# Expected Return Treatment
WINSORIZE_EXPECTED_RETURNS = True # Cap extreme values
WINSORIZE_LOWER_PERCENTILE = 0.05 # 5th percentile
WINSORIZE_UPPER_PERCENTILE = 0.95 # 95th percentile

# Risk Parameters
RISK_AVERSION_DEFAULT = 1.0       # Risk aversion (λ)
ROBUST_EPSILON = 0.5              # Uncertainty set size
```

---

## Generated Outputs

After running the pipeline, you'll have:

### Data Files
- `data/raw/sp100_prices.csv` - Historical prices
- `data/processed/log_returns.csv` - Daily returns
- `data/features/expected_returns_*.csv` - Return estimates
- `data/analysis/return_statistics.csv` - Summary statistics

### Results Files
- `results/backtest_returns.csv` - Daily portfolio returns
- `results/backtest_metrics.csv` - Performance metrics
- `results/backtest_cumulative_returns.csv` - Cumulative performance
- `results/weights/*.csv` - Portfolio weight histories

### Statistical Tests
- `results/test_mean_returns.csv` - Mean return comparisons
- `results/test_sharpe_ratios.csv` - Sharpe ratio bootstrap tests
- `results/test_volatility.csv` - Volatility equality tests
- `results/test_turnover.csv` - Turnover analysis
- `results/test_drawdowns.csv` - Maximum drawdown comparison

### Visualisations
- `results/figures/cumulative_returns.png` - Performance over time
- `results/figures/drawdowns.png` - Drawdown comparison
- `results/figures/risk_return_scatter.png` - Risk-return profile
- `results/figures/rolling_sharpe.png` - Rolling Sharpe ratios
- `results/figures/performance_metrics.png` - Metric comparison
- `results/figures/turnover_comparison.png` - Turnover analysis
- `results/figures/weight_evolution.png` - Weight changes over time
- `results/figures/subperiod_performance.png` - Regime-dependent performance

### LaTeX Tables
- `results/tables/performance_table.tex` - Main results table
- `results/tables/performance_table.csv` - CSV version

---