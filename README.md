# Thesis Code

**Authors**: Robert George Smith & Joaquin Rodriguez


## Project Overview

This project ... Robust Portfolio Optimisation Under Parameter Uncertainty (S&P 100)


## Features
...


## Project Structure

```
portfolio_optimiser/
│
├── main.py                    # Main entry point
├── data_preparation.py        # Prepares data
├── data/                      # Data directory
│   ├── raw/                   # Raw data before processing
│   ├── processed/             # Data for analysis
│   └── results/               # Results of portfolio optimisation
├── portfolio_optimiser.py     # Creates portfolio
├── portfolio_components/      # Portfolio components used by portfolio_optimisation.py
│   ├── ....py                 # ...
│   └── ....py                 # ...
└── README.md                  # This file
```

### Configuration
Edit the following parameters in `main.py`:

```python
DATA_FILES = "data/raw/"  # Raw Data file path
# ...
```

### Real-time vs Fast Mode
In `simulation_engine.py`, line in `main()`:


```python
# Fast mode (recommended)
simulation.run_simulation(real_time=False)

# Real-time mode with delays
simulation.run_simulation(real_time=True)
```


## Algorithm Parameters
...


## Performance Metrics
...

## Output

The simulation generates:

...



### Object-Oriented Design
- `dataDownloader`: Downloads and stores raw data (not set up yet)
...

