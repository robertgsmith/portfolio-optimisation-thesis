"""
Data Pipeline

Downloading and preparing data for analysis and backtesting

Authors: Robert George Smith & Joaquin Rodriguez
"""

from .data_downloader import DataDownloader
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .summary_statistics import SummaryStatistics


__all__ = [
    'DataDownloader',
    'DataProcessor',
    'FeatureEngineer',
    'SummaryStatistics'
]