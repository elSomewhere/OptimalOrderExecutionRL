from typing import Generator, Tuple, Dict
import random
from matplotlib.gridspec import GridSpec
from DataProvision import DataFeed
from OrderExecutor import OrderExecutor
from ScheduleExecutor import Event, LimitEvent, MarketEvent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_schedule(schedule: Dict[int, Event], data_gen: Generator[Tuple[int, Dict[int, int], Dict[int, int]], None, None], levels=10, resubmit_each_tick: bool = True, price_level_offset: float = -0.5):
    market_executor = OrderExecutor(levels)
    market_executor.reset()
    carryover_vol = 0
    trades_collect_all = {}
    tick_data_bids_all = {}
    tick_data_asks_all = {}
    placed_limit_orders_collect_all = {}
    placed_market_orders_collect_all = {}
    for index, event in schedule.items():
        if isinstance(event, LimitEvent):
            trades_collect, carryover_vol, tick_data_bids, tick_data_asks, placed_orders = market_executor.simulate_limit_order(data_gen, event.volume + carryover_vol, event.side, event.max_time, resubmit_each_tick, price_level_offset)
            trades_collect_all = {**trades_collect_all, **trades_collect}
            tick_data_bids_all = {**tick_data_bids_all, **tick_data_bids}
            tick_data_asks_all = {**tick_data_asks_all, **tick_data_asks}
            placed_limit_orders_collect_all = {**placed_limit_orders_collect_all, **placed_orders}
        elif isinstance(event, MarketEvent):
            trades_collect, carryover_vol, tick_data_bids, tick_data_asks, placed_orders = market_executor.simulate_market_order(data_gen, carryover_vol, event.side, event.max_time)
            trades_collect_all = {**trades_collect_all, **trades_collect}
            tick_data_bids_all = {**tick_data_bids_all, **tick_data_bids}
            tick_data_asks_all = {**tick_data_asks_all, **tick_data_asks}
            placed_market_orders_collect_all = {**placed_market_orders_collect_all, **placed_orders}
    if carryover_vol == 0:
        all_filled = True
    else:
        all_filled = False
    return trades_collect_all, tick_data_bids_all, tick_data_asks_all, carryover_vol, all_filled, placed_limit_orders_collect_all, placed_market_orders_collect_all


def parse_to_heatmap(tick_data_bids, tick_data_asks):
    # Combine all price levels from both bids and asks
    all_prices = set()
    for data in tick_data_bids.values():
        all_prices.update(data.keys())
    for data in tick_data_asks.values():
        all_prices.update(data.keys())

    # Create a sorted list of unique prices
    sorted_prices = sorted(list(all_prices))

    # Prepare an empty DataFrame with timestamps as index and prices as columns
    all_timestamps = set(tick_data_bids.keys()).union(set(tick_data_asks.keys()))
    index = sorted(list(all_timestamps))
    columns = sorted_prices
    heatmap_df = pd.DataFrame(index=index, columns=columns, dtype=float).fillna(0.0)

    # Fill the DataFrame with volumes
    for ts in all_timestamps:
        bid_data = tick_data_bids.get(ts, {})
        ask_data = tick_data_asks.get(ts, {})
        for price, volume in bid_data.items():
            heatmap_df.at[ts, price] += np.log(volume)
        for price, volume in ask_data.items():
            heatmap_df.at[ts, price] -= np.log(volume)  # Assuming asks are negative volumes

    return heatmap_df


def get_top_levels(tick_data_bids, tick_data_asks):
    top_levels_bids = {}
    for timestamp, levels in tick_data_bids.items():
        if not levels:
            continue  # Skip empty levels
        max_price = max(levels.keys())
        top_levels_bids[timestamp] = max_price
    top_levels_asks = {}
    for timestamp, levels in tick_data_asks.items():
        if not levels:
            continue  # Skip empty levels
        min_price = min(levels.keys())
        top_levels_asks[timestamp] = min_price
    return top_levels_bids, top_levels_asks


def plot_lob_data(tick_data_bids_all, tick_data_asks_all):
    # Example usage
    heatmap_data = parse_to_heatmap(tick_data_bids_all, tick_data_asks_all)

    # regular heatmap
    timestamps = heatmap_data.index.values
    prices = heatmap_data.columns.values
    T, P = np.meshgrid(timestamps, prices)

    # Assuming tick_data_bids_all and tick_data_asks_all are dictionaries with the proper timestamp format
    top_bids, top_asks = get_top_levels(tick_data_bids_all, tick_data_asks_all)

    # Assuming the timestamps are sorted, we can plot the lines directly
    sorted_bid_timestamps = sorted(top_bids.keys())
    sorted_ask_timestamps = sorted(top_asks.keys())

    # Extracting the top bid and ask prices in timestamp order
    top_bid_prices = [top_bids[ts] for ts in sorted_bid_timestamps]
    top_ask_prices = [top_asks[ts] for ts in sorted_ask_timestamps]

    # Setup figure and axes using GridSpec
    fig = plt.figure(figsize=(15, 9))
    fig.tight_layout()
    gs = GridSpec(2, 1, height_ratios=[6, 3])
    ax1 = fig.add_subplot(gs[0])

    # Plot heatmap in the first subplot
    cmap = ax1.pcolormesh(T, P, heatmap_data.T, shading='auto', cmap='viridis')
    # fig.colorbar(cmap, ax=ax1, label='Volume')
    ax1.set_ylabel('Price')
    ax1.set_title('Order Book Heatmap with Top Bids and Asks')
    ax1.plot(sorted_bid_timestamps, top_bid_prices, color='green', label='Top Bid')
    ax1.plot(sorted_ask_timestamps, top_ask_prices, color='orange', label='Top Ask')

    # Extend the x-axis limits
    x_min, x_max = ax1.get_xlim()
    x_buffer = int((x_max - x_min) * 0.05)  # Extend by 5% of the range on both sides
    ax1.set_xlim(x_min - x_buffer, x_max + x_buffer)
    return ax1



def plot_lob_data_with_volume(schedule, data_gen, resubmit_each_tick: bool = True, price_level_offset: float = -0.8):
    trades_collect_all, tick_data_bids_all, tick_data_asks_all, carryover_vol, all_filled, placed_limit_orders_collect_all, placed_market_orders_collect_all = run_schedule(schedule, data_gen, resubmit_each_tick=resubmit_each_tick, price_level_offset=price_level_offset)
    # Generate heatmap data
    heatmap_data = parse_to_heatmap(tick_data_bids_all, tick_data_asks_all)

    # Extract top bid and ask prices
    top_bids, top_asks = get_top_levels(tick_data_bids_all, tick_data_asks_all)

    # Setup figure and axes using GridSpec
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 1, height_ratios=[6, 3])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot heatmap in the first subplot
    timestamps = heatmap_data.index.values
    prices = heatmap_data.columns.values
    T, P = np.meshgrid(timestamps, prices)
    cmap = ax1.pcolormesh(T, P, heatmap_data.T, shading='auto', cmap='viridis')
    ax1.set_ylabel('Price')
    # ax1.set_title('Order Book Heatmap with Top Bids and Asks')

    # Plot top bids and asks
    sorted_bid_keys = sorted(top_bids.keys())
    sorted_ask_keys = sorted(top_asks.keys())
    ax1.plot(sorted_bid_keys, [top_bids[ts] for ts in sorted_bid_keys], color='green', label='Top Bid')
    ax1.plot(sorted_ask_keys, [top_asks[ts] for ts in sorted_ask_keys], color='orange', label='Top Ask')
    ax1.legend()

    # Overlay trades and orders on the heatmap
    # Overlay the trades on the heatmap
    for timestamp, trades in trades_collect_all.items():
        for trade in trades:
            for price, volume in trade.price_levels.items():
                ax1.scatter(timestamp, price, s=np.log(volume + 1) * 10, color='red', alpha=0.6)

    # Overlay the trades on the heatmap
    for timestamp, order in placed_limit_orders_collect_all.items():
        ax1.scatter(timestamp, order.price, s=np.log(order.volume + order.filled_volume + 1) * 10, color='blue', alpha=0.6)

    for timestamp, order in placed_market_orders_collect_all.items():
        # extract top level of the given side so we have a price to display at
        if order.side == "bid":
            price = min(list(tick_data_asks_all[timestamp].keys()))
        elif order.side == "ask":
            price = max(list(tick_data_bids_all[timestamp].keys()))
        ax1.scatter(timestamp, price, s=np.log(order.volume + order.filled_volume + 1) * 10, color='green', alpha=0.6)

    # Prepare data for the volume bar plot
    limit_order_timestamps = sorted(placed_limit_orders_collect_all.keys())
    limit_order_volumes = [placed_limit_orders_collect_all[ts].volume + placed_limit_orders_collect_all[ts].filled_volume for ts in limit_order_timestamps]

    market_order_timestamps = sorted(placed_market_orders_collect_all.keys())
    market_order_volumes = [placed_market_orders_collect_all[ts].volume + placed_market_orders_collect_all[ts].filled_volume for ts in market_order_timestamps]

    trade_timestamps = sorted(trades_collect_all.keys())
    trade_volumes = [sum(trade.filled_volume for trade in trades_collect_all[ts]) for ts in trade_timestamps]

    bar_width = (max(timestamps) - min(timestamps)) / len(timestamps) * 0.3

    # Create a small offset for the bars so they don't overlap if they have the same timestamp
    offset = bar_width / 2

    # Bar plot for limit order and trade volumes
    ax2.bar(np.array(limit_order_timestamps) - offset, limit_order_volumes, width=bar_width, align='center', color='blue', label='Limit Order Volume')
    ax2.bar(np.array(trade_timestamps) + offset, trade_volumes, width=bar_width, align='center', color='red', alpha=0.7, label='Trade Volume')
    ax2.bar(np.array(market_order_timestamps), market_order_volumes, width=bar_width, align='center', color='green', alpha=0.7, label='Market order Volume')

    ax2.legend(loc='upper right')
    # ax2.set_title('Limit Order and Trade Volumes Over Time')

    # Format the timestamp for x-axis
    # ax2.xaxis_date()

    plt.setp(ax1.get_xticklabels(), visible=False)  # Hide x-axis labels for the top plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)  # Adjust the space between the plots
    return ax1


def plot_lob_and_executed_schedule(tick_data_bids_all, tick_data_asks_all, trades_collect_all, placed_limit_orders_collect_all, placed_market_orders_collect_all):
    ax1 = plot_lob_data(tick_data_bids_all, tick_data_asks_all)
    # Overlay the trades on the heatmap
    for timestamp, trades in trades_collect_all.items():
        for trade in trades:
            for price, volume in trade.price_levels.items():
                ax1.scatter(timestamp, price, s=np.log(volume + 1) * 10, color='red', alpha=0.6)

    # Overlay the trades on the heatmap
    for timestamp, order in placed_limit_orders_collect_all.items():
        ax1.scatter(timestamp, order.price, s=np.log(order.volume + order.filled_volume + 1) * 10, color='blue', alpha=0.6)

    for timestamp, order in placed_market_orders_collect_all.items():
        # extract top level of the given side so we have a price to display at
        if order.side =="bid":
            price = min(list(tick_data_asks_all[timestamp].keys()))
        elif order.side == "ask":
            price = max(list(tick_data_bids_all[timestamp].keys()))
        ax1.scatter(timestamp, price, s=np.log(order.volume + order.filled_volume + 1) * 10, color='green', alpha=0.6)
    return ax1

def execute_and_plot_schedule(schedule, data_gen, resubmit_each_tick: bool = True, price_level_offset: float = -0.8):
    trades_collect_all, tick_data_bids_all, tick_data_asks_all, carryover_vol, all_filled, placed_limit_orders_collect_all, placed_market_orders_collect_all = run_schedule(schedule, data_gen, resubmit_each_tick=resubmit_each_tick, price_level_offset=price_level_offset)
    ax1 = plot_lob_and_executed_schedule(tick_data_bids_all, tick_data_asks_all, trades_collect_all, placed_limit_orders_collect_all, placed_market_orders_collect_all)
    return ax1

def plot_dataset(data_feed):
    data_gen = data_feed.raw_data_yield_fun(0)
    tick_data_bids = {}
    tick_data_asks = {}
    for index, first_bids, first_asks in data_gen:
        tick_data_bids.update({index: first_bids})
        tick_data_asks.update({index: first_asks})
    plot_lob_data(tick_data_bids, tick_data_asks)






if __name__ == '__main__':
    # get the data
    data = pd.read_csv("LOBL2.csv")
    data = data.set_index(data.iloc[:, 0]).iloc[:, 1::]


    # data metrics to get some good training params, 1§e6 = 1s
    num_ticks = data.shape[0]
    average_tick_duration = np.round(np.nanmean(data.index.diff()))
    average_tick_duration_s = average_tick_duration / 1e6
    max_time = int(3 * 1e6)
    max_volume = int(3 * 1e6)
    side = "bid"
    # other params
    num_schedule_elements = 3
    schedule = {}
    for i in range(num_schedule_elements):
        schedule.update({i: LimitEvent(side, max_time, max_volume)})
    schedule.update({len(schedule): MarketEvent(side, max_time)})
    # run
    plt.close()
    start_index = random.choice(data.index[0:(len(data.index) // 3) * 2])
    print(start_index)
    start_index =1710498181404675
    data_gen = DataFeed(data, levels=10).raw_data_yield_fun(start_index)
    trades_collect_all, tick_data_bids_all, tick_data_asks_all, carryover_vol, all_filled, placed_limit_orders_collect_all, placed_market_orders_collect_all = run_schedule(schedule, data_gen, 10)
    plot_lob_data_with_volume(schedule, data_gen, resubmit_each_tick=False, price_level_offset=0.0)
    plt.show()