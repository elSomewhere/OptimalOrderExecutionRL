from typing import Optional, Generator, Tuple, Dict, List
from BacktesterL2 import Backtester, Trade, LimitOrder, MarketOrder
from DataProvision import DataFeed


#=======================================================================================================================
# Market Execution classes & methods.
#=======================================================================================================================

class OrderExecutor:
    '''
    Executes a given job, bucket by bucket
    Step executes next bucket
    '''

    def __init__(self, levels):
        self.backtester: Backtester = Backtester()
        self.levels = levels

    def reset(self):
        self.backtester.reset()

    def simulate_limit_order(self, data_gen: Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None], order_volume: int, side: str, max_time: Optional[int], resubmit_each_tick: bool =True, price_depth_posted = 1) -> Tuple[Dict[int, List[Trade]], int, Dict[int, Dict[int, int]], Dict[int, Dict[int, int]], Dict[int, LimitOrder]]:
        '''
        This variant re-submits a limit order every tick with a new price, which is pretty conservative,
        because for example given a limit bid close to the top and a downward market trajectory, the bid is not allowed to rest
        but instead gets continuously pushed downwards. In other words, it only gets filled if the downward trajectory is big enough in a single given tick.
        '''
        placed_orders = {}
        tick_data_bids = {}
        tick_data_asks = {}
        trade_data = {}
        remaining_volume = order_volume
        # get initial data
        time_start, first_bids, first_asks = next(data_gen)
        tick_data_bids.update({time_start: first_bids})
        tick_data_asks.update({time_start: first_asks})
        # init the LOBs with first data point (we need it to generate price)
        _ = self.backtester.step(first_bids, first_asks, time_start)
        # calculate the price
        best_bid = self.backtester.lob.get_best_bid()
        best_ask = self.backtester.lob.get_best_ask()
        order_price = best_bid + self.backtester.lob.get_spread() * price_depth_posted if side == 'bid' else best_ask - self.backtester.lob.get_spread() * price_depth_posted if side == 'ask' else None
        # place the order
        limit_order_id = self.backtester.place_virtual_limit_order(order_price, remaining_volume, side)
        placed_orders.update({self.backtester.timestep: self.backtester.get_limit_order(limit_order_id)})
        # start the backtest for this step. Note this is indeed actually re-placing limit orders, i.e. modifying the price unlike in real life
        empty_generator = True
        time_passed = 0
        for index, first_bids, first_asks in data_gen:
            tick_data_bids.update({index: first_bids})
            tick_data_asks.update({index: first_asks})
            # calc time passed since start of bucket
            time_passed = index - time_start
            # check if we exhausted our available time, if so break
            if max_time is not None:
                if time_passed > max_time:
                    # break loop, move on to finalization
                    empty_generator = False
                    deleted_limit_order = self.backtester.delete_limit_order(limit_order_id)
                    remaining_volume = deleted_limit_order.volume
                    break
            # execute timestep
            executed_trades, to_remove_limit_orders, _ = self.backtester.step(first_bids, first_asks, index)
            # parse the results
            trade_data.update({index: executed_trades})
            # check if our orders are exhausted, we could then do early stop
            if len(self.backtester.virtual_limit_orders) == 0 and len(self.backtester.virtual_limit_orders) == 0:
                remaining_volume = to_remove_limit_orders[0].volume
                assert (remaining_volume == 0)
                empty_generator = False
                break
            if resubmit_each_tick:
                # if the limit order still rests, delete the limit order and re-place it according to the new price and remaining volume
                deleted_limit_order = self.backtester.delete_limit_order(limit_order_id)
                remaining_volume = deleted_limit_order.volume
                # place the orders, first the RL order
                best_bid = self.backtester.lob.get_best_bid()
                best_ask = self.backtester.lob.get_best_ask()
                order_price = best_bid + self.backtester.lob.get_spread() * price_depth_posted if side == 'bid' else best_ask - self.backtester.lob.get_spread() * price_depth_posted if side == 'ask' else None
                limit_order_id = self.backtester.place_virtual_limit_order(order_price, remaining_volume, side)
                placed_orders.update({self.backtester.timestep: self.backtester.get_limit_order(limit_order_id)})
        # make sure we didn't exit the loop cus available data was exhausted
        if empty_generator:
            raise Exception("Ran out of data")
        for index, first_bids, first_asks in data_gen:
            tick_data_bids.update({index: first_bids})
            tick_data_asks.update({index: first_asks})
            time_passed = index - time_start
            if time_passed > max_time:
                break
        return trade_data, remaining_volume, tick_data_bids, tick_data_asks, placed_orders

    def simulate_market_order(self, data_gen: Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None], order_volume: int, side: str, max_time: Optional[int]) -> Tuple[Dict[int, List[Trade]], int, Dict[int, Dict[int, int]], Dict[int, Dict[int, int]], Dict[int, MarketOrder]]:
        placed_orders = {}
        tick_data_bids = {}
        tick_data_asks = {}
        trade_data = {}
        remaining_volume = order_volume
        # get initial data
        time_start, first_bids, first_asks = next(data_gen)
        tick_data_bids.update({time_start: first_bids})
        tick_data_asks.update({time_start: first_asks})
        # init the LOBs with first data point (we need it to generate price)
        _ = self.backtester.step(first_bids, first_asks, time_start)
        # place the order
        market_order_id = self.backtester.place_virtual_market_order(remaining_volume, side)
        placed_orders.update({self.backtester.timestep: self.backtester.get_market_order(market_order_id)})
        # start the backtest for this step. Note this is indeed actually re-placing market orders, so as to be sure all gets exhausted

        empty_generator = True
        time_passed = 0
        for index, first_bids, first_asks in data_gen:
            tick_data_bids.update({index: first_bids})
            tick_data_asks.update({index: first_asks})
            # calc time passed since start of bucket
            time_passed = index - time_start
            # check if we exhausted our available time, if so break
            if max_time is not None:
                if time_passed > max_time:
                    # break loop, move on to finalization
                    empty_generator = False
                    break
            # execute timestep
            executed_trades, _, to_remove_market_orders = self.backtester.step(first_bids, first_asks, index)
            # parse the results
            trade_data.update({index: executed_trades})
            # check if our orders are exhausted, we could then do early stop
            remaining_volume = to_remove_market_orders[0].volume
            if remaining_volume == 0:
                empty_generator = False
                break
            # if the limit order still rests, delete the limit order and re-place it according to the new price and remaining volume
            market_order_id = self.backtester.place_virtual_market_order(remaining_volume, side)
            placed_orders.update({self.backtester.timestep: self.backtester.get_market_order(market_order_id)})
        # make sure we didn't exit the loop cus available data was exhausted
        if empty_generator:
            raise Exception("Ran out of data")
        for index, first_bids, first_asks in data_gen:
            tick_data_bids.update({index: first_bids})
            tick_data_asks.update({index: first_asks})
            time_passed = index - time_start
            if time_passed > max_time:
                break
        return trade_data, remaining_volume, tick_data_bids, tick_data_asks, placed_orders




if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv("LOBL2.csv")
    data = data.set_index(data.iloc[:, 0]).iloc[:, 1::]
    max_time = (data.index[-1] - data.index[0]) / 3 * 2
    env = OrderExecutor(10)
    backtester = Backtester()
    data_feed = DataFeed(data, levels=10)
    data_gen = data_feed.raw_data_yield_fun(0)
    order_volume = 61000000# 1 too much: 5129844392    1 before: 5129844390
    side = "bid"
    trade_data, remaining_volume, tick_data_bids, tick_data_asks, placed_orders = env.simulate_limit_order(data_gen, order_volume, side, max_time)