# File to prepare the data

from data_components.data_downloader import _sp100_downloader

def prepare_data():
    _sp100_downloader()


# run data preparation
prepare_data