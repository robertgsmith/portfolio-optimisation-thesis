"""
S&P 100 Portfolio Data Downloader Module

Download US equity price data from Yahoo Finance (via yfinance),
construct month-end prices and monthly returns, and save to disk.

Universe default: current S&P 100 constituents scraped from Wikipedia.
Caveat: this is NOT historical membership (survivorship bias).
"""
import yfinance as yf
import pandas as pd


def _data_downloader():
    # S&P 100 tickers (OEX components as of recent composition)
    SP100_TICKERS = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT',
        'AMZN', 'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY',
        'BRK-B', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST',
        'CRM', 'CSCO', 'CVS', 'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR',
        'EXC', 'F', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS',
        'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT',
        'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK',
        'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE',
        'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG',
        'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS',
        'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
    ]

    for ticker in SP100_TICKERS:
        ticker_instance = yf.Ticker(ticker)
        ticker_data = ticker_instance.history(period="max")
        pd.set_option('display.max_rows', None)
        ticker_data.to_csv(f'data/raw/{ticker}.csv')