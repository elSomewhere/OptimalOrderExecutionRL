from typing import Tuple, Dict, Generator

import pandas as pd


class DataFeed:
    '''
    Parses raw L2 data into a generator that can be passed around.
    '''
    def __init__(self, data: pd.DataFrame, levels: int, price_norm: int = 1, volume_norm: int = 1):
        self.data = data
        self.levels = levels
        self.price_norm = price_norm
        self.volume_norm = volume_norm

    def raw_data_yield_fun(self, start_time_at: int) -> Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None]:
        data_trimmed = self.data[self.data.index > start_time_at]
        for index, row in data_trimmed.iterrows():
            new_bids, new_asks = self.parse_raw_order_book_row(row)
            yield index, new_bids, new_asks
    def parse_raw_order_book_row(self, row) -> Tuple[Dict[int, int], Dict[int, int]]:
        # Extract bids and asks
        bids_data = row.iloc[:(self.levels * 2)]
        asks_data = row.iloc[(self.levels * 2):]
        # Parse bids
        new_bids = {}
        for i in range(0, len(bids_data), 2):
            price = int(bids_data.iloc[i] / self.price_norm)
            volume = int(bids_data.iloc[i + 1] / self.volume_norm)
            new_bids[price] = volume
        # Parse asks
        new_asks = {}
        for i in range(0, len(asks_data), 2):
            price = int(asks_data.iloc[i] / self.price_norm)
            volume = int(asks_data.iloc[i + 1] / self.volume_norm)
            new_asks[price] = volume
        return new_bids, new_asks