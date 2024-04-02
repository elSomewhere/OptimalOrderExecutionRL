from typing import Dict, Generator, Tuple, List, Optional
import numpy as np
from BacktesterL2 import Trade
from DataProvision import DataFeed
from OrderExecutor import OrderExecutor


class Event:
    def __init__(self, side: str, max_time: int):
        self.side = side
        self.max_time = max_time

class LimitEvent(Event):
    def __init__(self, side: str, max_time: int, volume: int):
        super().__init__(side, max_time)
        self.volume = volume


class MarketEvent(Event):
    def __init__(self, side: str, max_time: int):
        super().__init__(side, max_time)


class ScheduleExecutor:
    def __init__(self, resubmit_each_tick=True, price_depth_penalty=-0.1):
        self.schedule_gen = None
        self.carryover_vol_rl = None
        self.carryover_vol_twap = None
        self.trades_collect_rl = {}
        self.trades_collect_twap = {}
        self.data_feed = None
        self.levels = 10
        self.resubmit_each_tick = resubmit_each_tick
        self.price_depth_penalty = price_depth_penalty
        self.market_executor_rl: OrderExecutor = OrderExecutor(self.levels)
        self.market_executor_twap: OrderExecutor = OrderExecutor(self.levels)
        self.index = 0

    def reset(self, schedule, data_feed, side, start_time_at=0):
        self.total_volume_twap = sum([i.volume if isinstance(i, LimitEvent) else 0 for i in schedule.values()])
        self.total_volume_rl = sum([i.volume if isinstance(i, LimitEvent) else 0 for i in schedule.values()])
        self.schedule_gen = self.get_next_event(schedule)
        self.data_gen_rl = data_feed.raw_data_yield_fun(start_time_at)
        self.data_gen_twap = data_feed.raw_data_yield_fun(start_time_at)
        self.trades_collect_rl = {}
        self.trades_collect_twap = {}
        self.carryover_vol_rl = 0
        self.carryover_vol_twap = 0
        self.market_executor_rl.reset()
        self.market_executor_twap.reset()
        self.index = 0
        self.length = len(schedule)
        self.total_remaining_vol_rl = self.total_volume_rl
        self.total_remaining_vol_twap = self.total_volume_twap
        self.side = side
        # update the backtester lob with an initial state the RL agent has a meaningful state to begin with
        time_start, first_bids, first_asks = next(self.data_gen_rl)
        _ = self.market_executor_rl.backtester.step(first_bids, first_asks, time_start)
        time_start, first_bids, first_asks = next(self.data_gen_twap)
        _ = self.market_executor_twap.backtester.step(first_bids, first_asks, time_start)


    def _step(self, action):
        index, event, done = next(self.schedule_gen)
        self.index = index
        if isinstance(event, LimitEvent):
            volume_twap = event.volume
            max_time = event.max_time
            volume_twap_plus_carryover = volume_twap + self.carryover_vol_twap
            volume_rl = action * volume_twap
            volume_rl_plus_carryover = volume_rl + self.carryover_vol_rl
            volume_to_place_rl = min(self.total_remaining_vol_rl, volume_rl_plus_carryover)
            volume_to_place_twap = min(self.total_remaining_vol_twap, volume_twap_plus_carryover)
            # simulates a limit order placement for both RL and TWAP
            # trades_rl, remaining_vol_rl, trades_twap, remaining_vol_twap = self.env.simulate_limit(self.data_gen_rl, self.data_gen_twap, volume_to_place_rl, volume_to_place_twap, self.side, max_time)
            trades_rl, remaining_vol_rl, _, _, _ = self.market_executor_rl.simulate_limit_order(self.data_gen_rl, volume_to_place_rl, self.side, max_time, self.resubmit_each_tick, self.price_depth_penalty)
            trades_twap, remaining_vol_twap, _, _, _ = self.market_executor_twap.simulate_limit_order(self.data_gen_twap, volume_to_place_twap, self.side, max_time, self.resubmit_each_tick, self.price_depth_penalty)
            filled_volume_rl = volume_to_place_rl - remaining_vol_rl
            filled_volume_twap = volume_to_place_twap - remaining_vol_twap
            # print("Limit Order: filled_volume_rl: " + str(filled_volume_rl) + " filled_volume_twap: " + str(filled_volume_twap))
            self.total_remaining_vol_rl -= filled_volume_rl
            self.total_remaining_vol_twap -= filled_volume_twap
            return trades_rl, remaining_vol_rl, trades_twap, remaining_vol_twap, done
        elif isinstance(event, MarketEvent):
            volume_twap_plus_carryover = self.carryover_vol_twap
            volume_rl_plus_carryover = self.carryover_vol_rl
            volume_to_place_rl = max(self.total_remaining_vol_rl, volume_rl_plus_carryover)
            volume_to_place_twap = max(self.total_remaining_vol_twap, volume_twap_plus_carryover)
            # simulates a market order for both RL and TWAP - note we will also make sure to account the RLAgent having gone short
            if volume_to_place_rl >= 0:
                final_side_rl = "bid"
            else:
                final_side_rl = "ask"
            if volume_to_place_twap >= 0:
                final_side_twap = "bid"
            else:
                final_side_twap = "ask"
            trades_rl, remaining_vol_rl, _, _, _ = self.market_executor_rl.simulate_market_order(self.data_gen_rl, volume_to_place_rl, final_side_rl, np.Inf)
            trades_twap, remaining_vol_twap, _, _, _ = self.market_executor_twap.simulate_market_order(self.data_gen_twap, volume_to_place_twap, final_side_twap, np.Inf)
            filled_volume_rl = volume_to_place_rl - remaining_vol_rl
            filled_volume_twap = volume_to_place_twap - remaining_vol_twap
            # print("Market Order: filled_volume_rl: "+str(filled_volume_rl)+" filled_volume_twap: "+str(filled_volume_twap))
            self.total_remaining_vol_rl -= filled_volume_rl
            self.total_remaining_vol_twap -= filled_volume_twap
            return trades_rl, remaining_vol_rl, trades_twap, remaining_vol_twap, done

    def simulate_market(self, data_gen_rl: Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None], data_gen_twap: Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None], target_bucket_volume_rl: int, target_bucket_volume_twap: int, side: str, max_time: int) -> Tuple[Dict[int, List[Trade]], int, Dict[int, List[Trade]], int]:
        trades_rl, remaining_vol_rl, _, _, _  = self.market_executor_rl.simulate_market_order(data_gen_rl, target_bucket_volume_rl, side, max_time)
        trades_twap, remaining_vol_twap, _, _, _ = self.market_executor_twap.simulate_market_order(data_gen_twap, target_bucket_volume_twap, side, max_time)
        return trades_rl, remaining_vol_rl, trades_twap, remaining_vol_twap

        # target_bucket_volume_rl should me mutated previously by an action...

    def simulate_limit(self, data_gen_rl: Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None], data_gen_twap: Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None], target_bucket_volume_rl: int, target_bucket_volume_twap: int, side: str, max_time: int, resubmit_each_tick: bool) -> Tuple[Dict[int, List[Trade]], int, Dict[int, List[Trade]], int]:
        resubmit_each_tick = True
        trades_rl, remaining_vol_rl, _, _, _ = self.market_executor_rl.simulate_limit_order(data_gen_rl, target_bucket_volume_rl, side, max_time, resubmit_each_tick)
        trades_twap, remaining_vol_twap, _, _, _ = self.market_executor_twap.simulate_limit_order(data_gen_twap, target_bucket_volume_twap, side, max_time, resubmit_each_tick)
        return trades_rl, remaining_vol_rl, trades_twap, remaining_vol_twap

    def step(self, action):
        trades_rl, remaining_vol_rl, trades_twap, remaining_vol_twap, done = self._step(action)
        self.carryover_vol_twap = remaining_vol_twap
        self.carryover_vol_rl = remaining_vol_rl
        self.trades_collect_rl = {**self.trades_collect_rl, **trades_rl}
        self.trades_collect_twap = {**self.trades_collect_twap, **trades_twap}
        return done, trades_rl, trades_twap

    def get_next_event(self, schedule: Dict[int, Event]):
        max_index = len([i for i in schedule.keys()])
        done = False
        for index, event in schedule.items():
            if index == max_index-1:
                done = True
            yield index, event, done

    def get_reward(self, trades_rl, trades_twap):
        if len(trades_rl) == 0 and len(trades_twap) == 0:
            return 0
        exec_price_rl = 0
        exec_volume_rl = 0
        for t in trades_rl.values():
            for trade in t:
                for p, v in trade.price_levels.items():
                    exec_price_rl += p * v
                    exec_volume_rl += v
        if exec_price_rl > 0 and exec_volume_rl > 0:
            rl_vwap = exec_price_rl / exec_volume_rl
        else:
            rl_vwap = 0
        exec_price_twap = 0
        exec_volume_twap = 0
        for t in trades_twap.values():
            for trade in t:
                for p, v in trade.price_levels.items():
                    exec_price_twap += p * v
                    exec_volume_twap += v
        if exec_price_twap > 0 and exec_volume_twap > 0:
            twap_vwap = exec_price_twap / exec_volume_twap
        else:
            twap_vwap = 0
        reward = twap_vwap - rl_vwap
        return reward

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv("LOBL2.csv")
    data = data.set_index(data.iloc[:, 0]).iloc[:, 1::]
    data_feed = DataFeed(data, levels=10)
    action = 1.5
    max_time = int(3 * 1e6)
    max_volume = int(3 * 1e6)
    side = "bid"
    data_gen_rl = data_feed.raw_data_yield_fun(0)
    data_gen_twap = data_feed.raw_data_yield_fun(0)
    num_schedule_elements = 3
    schedule = {}
    for i in range(num_schedule_elements):
        schedule.update({i: LimitEvent(side, max_time, max_volume)})
    schedule.update({len(schedule): MarketEvent(side, max_time)})
    env = ScheduleExecutor()
    env.reset(schedule, data_feed, side)
    done = False
    all_trades_rl = {}
    all_trades_twap = {}
    while not done:
        done, trades_rl, trades_twap = env.step(action)
        all_trades_rl = {**all_trades_rl, **trades_rl}
        all_trades_twap = {**all_trades_twap, **trades_twap}
    print(env.get_reward(all_trades_rl, all_trades_twap))
