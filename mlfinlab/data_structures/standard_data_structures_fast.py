"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

# Imports
from collections import namedtuple
import pandas as pd
import numpy as np
from numba import jit

from mlfinlab.data_structures.base_bars_fast import BaseBars


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, file_path, metric, threshold=50000, batch_size=20000000):

        BaseBars.__init__(self, file_path, metric, batch_size)

        # Threshold at which to sample
        self.threshold = threshold
        # Named tuple to help with the cache
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'cum_volume', 'cum_dollar'])

    def _extract_bars(self, data, inverse=False):

        dt_arr = pd.to_datetime(data.iloc[:, 0]).astype(np.int64).values
        price_arr = data.iloc[:, 1].astype(float).values
        volume_arr = data.iloc[:, 2].astype(int).values

        new_bar, cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = self._update_counters()

        kwarg_dict = {
            'metric': self.metric,
            'threshold': self.threshold,
            'inverse': inverse,
            'dt_arr': dt_arr,
            'price_arr': price_arr,
            'volume_arr': volume_arr,
            'new_bar': new_bar,
            'cum_ticks': cum_ticks,
            'cum_dollar_value': cum_dollar_value,
            'cum_volume': cum_volume,
            'high_price': high_price,
            'low_price': low_price
        }

        arr_bars, cache_tuple = self._iterate_bars(**kwarg_dict)

        self.cache = () if np.inf in cache_tuple else self.cache_tuple(*cache_tuple)
        return arr_bars

    @staticmethod
    @jit(nopython=True)
    def _iterate_bars(metric, threshold, inverse, dt_arr, price_arr, volume_arr, new_bar, cum_ticks, cum_dollar_value, cum_volume, high_price, low_price):
        """
        For loop which compiles the various bars: dollar, volume, or tick.

        :param data: Contains 3 columns - date_time, price, and volume.
        """

        # Iterate over rows
        arr_bars = np.zeros((len(dt_arr), 6))
        n_bars = 0

        for i in range(len(dt_arr)):
            # Set variables
            date_time = dt_arr[i]
            price = price_arr[i]
            volume = volume_arr[i]

            # Update open high low prices
            if new_bar:
                open_price = price
                new_bar = False

            high_price = max(high_price, price)
            low_price = min(low_price, price)

            # Calculations
            cum_ticks += 1
            dollar_value = volume / price if inverse else price * volume
            cum_dollar_value = cum_dollar_value + dollar_value
            cum_volume += volume

            # If threshold reached then take a sample
            if metric == 'cum_ticks':
                compare = cum_ticks
            elif metric == 'cum_dollar_value':
                compare = cum_dollar_value
            elif metric == 'cum_volume':
                compare = cum_volume

            if compare >= threshold:  # pylint: disable=eval-used
                high_price = max(open_price, high_price)
                low_price = min(open_price, low_price)

                arr_bars[n_bars, 0] = date_time
                arr_bars[n_bars, 1] = open_price
                arr_bars[n_bars, 2] = high_price
                arr_bars[n_bars, 3] = low_price
                arr_bars[n_bars, 4] = price
                arr_bars[n_bars, 5] = cum_volume

                # Reset counters
                cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf

                n_bars += 1
                new_bar = True

        cache_tuple = (date_time, price, low_price, high_price,
                       cum_ticks, cum_volume, cum_dollar_value)

        return arr_bars[:n_bars], cache_tuple

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: Updated counters - cum_ticks, cum_dollar_value, cum_volume, high_price, low_price
        """
        # Check flag
        if self.flag and self.cache:
            new_bar = False
            last_entry = self.cache

            # Update variables based on cache
            cum_ticks = int(last_entry.cum_ticks)
            cum_dollar_value = np.float(last_entry.cum_dollar)
            cum_volume = last_entry.cum_volume
            low_price = np.float(last_entry.low)
            high_price = np.float(last_entry.high)
        else:
            # Reset counters
            new_bar = True
            cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf

        return new_bar, cum_ticks, cum_dollar_value, cum_volume, high_price, low_price


def get_dollar_bars(file_path, inverse=False, threshold=70000000, batch_size=20000000, verbose=True, to_csv=False, output_path=None):
    """
    Creates the dollar bars: date_time, open, high, low, close.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path: File path pointing to csv data.
    :param threshold: A cumulative value above this threshold triggers a sample to be taken.
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param output_path: Path to csv file, if to_csv is True
    :return: Dataframe of dollar bars
    """

    bars = StandardBars(file_path=file_path, metric='cum_dollar_value',
                        threshold=threshold, batch_size=batch_size)
    dollar_bars = bars.batch_run(
        inverse=inverse, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars


def get_volume_bars(file_path, threshold=28224, batch_size=20000000, verbose=True, to_csv=False, output_path=None):
    """
    Creates the volume bars: date_time, open, high, low, close.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

    :param file_path: File path pointing to csv data.
    :param threshold: A cumulative value above this threshold triggers a sample to be taken.
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param output_path: Path to csv file, if to_csv is True
    :return: Dataframe of volume bars
    """
    bars = StandardBars(file_path=file_path, metric='cum_volume',
                        threshold=threshold, batch_size=batch_size)
    volume_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars


def get_tick_bars(file_path, threshold=2800, batch_size=20000000, verbose=True, to_csv=False, output_path=None):
    """
    Creates the tick bars: date_time, open, high, low, close.

    :param file_path: File path pointing to csv data.
    :param threshold: A cumulative value above this threshold triggers a sample to be taken.
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param output_path: Path to csv file, if to_csv is True
    :return: Dataframe of tick bars
    """
    bars = StandardBars(file_path=file_path, metric='cum_ticks',
                        threshold=threshold, batch_size=batch_size)
    tick_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars
