from typing import Dict, List, Optional, Tuple
from DataProvision import DataFeed

class Order:
    '''
    Baseline Order class
    '''
    order_count = 0  # static var for generating unique order IDs
    def __init__(self, side: str):
        self.order_id = LimitOrder.order_count
        Order.order_count += 1
        self.side: str = side


class LimitOrder(Order):
    '''
    Baseline Limit Order class
    '''
    def __init__(self, volume: int, price: int, side: str):
        super().__init__(side)
        self.order_id = Order.order_count
        self.volume: int = volume
        self.price: int = price
        self.filled_volume: int = 0


class MarketOrder(Order):
    '''
    Baseline Market Order class
    '''
    def __init__(self, volume: int, side: str):
        super().__init__(side)
        self.order_id = Order.order_count
        self.volume: int = volume
        self.filled_volume: int = 0


class LOB:
    '''
    A simple Limit Order Book for L2 granularity.
    Note: the dicts are not sorted.
    this is a really quick and dirty implementation....
    A proper implementation would use some ordered container for prices, i.e. std::set with comparators in c++.
    An implementation for L3 would be significantly more complex if we want it to be performant.
    '''

    def __init__(self):
        self.bids: Dict[int, int] = {}
        self.asks: Dict[int, int] = {}

    def update_lob(self, new_bids: Dict[int, int], new_asks: Dict[int, int]) -> None:
        '''
        Pass new L2 data to the notebook
        '''
        self.bids = new_bids
        self.asks = new_asks

    def get_best_bid(self) -> Optional[int]:
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[int]:
        return min(self.asks.keys()) if self.asks else None

    def get_mid(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def get_spread(self) -> Optional[int]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None


class Trade:
    '''
    This class keeps track of any executions that happen as a consequence of our Agents interaction
    '''

    def __init__(self, order_id: int, order_type: str, filled_volume: int, remaining_volume: int):
        self.order_type = order_type
        self.order_id = order_id
        self.filled_volume = filled_volume
        self.remaining_volume = remaining_volume
        self.price_levels = {}  # Dict[int, int] where key is price and value is volume

    def add_fill_at_price(self, price: int, volume: int):
        if price in self.price_levels:
            self.price_levels[price] += volume
        else:
            self.price_levels[price] = volume


class Backtester:
    def __init__(self):
        self.lob: LOB = LOB()
        self.virtual_limit_orders: List[LimitOrder] = []
        self.virtual_market_orders: List[MarketOrder] = []
        self.tick_size: int = 1
        self.timestep = None

    def reset(self):
        self.lob: LOB = LOB()
        self.virtual_limit_orders: List[LimitOrder] = []
        self.virtual_market_orders: List[MarketOrder] = []
        self.timestep = None

    def get_limit_order(self, order_id: int) -> Optional[LimitOrder]:
        for order in self.virtual_limit_orders:
            if order.order_id == order_id:
                return order
        return None

    def get_market_order(self, order_id: int) -> Optional[MarketOrder]:
        for order in self.virtual_market_orders:
            if order.order_id == order_id:
                return order
        return None

    def place_virtual_limit_order(self, price: int, volume: int, side: str) -> int:
        '''
        Places a virtual limit order.
        Gets continuously matched till empty in subsequent steps, then automatically deleted
        '''
        order = LimitOrder(volume, price, side)
        self.virtual_limit_orders.append(order)
        return order.order_id

    def place_virtual_market_order(self, volume: int, side: str) -> int:
        '''
        Places a virtual limit order.
        Gets once matched in next step, then automatically deleted
        '''
        order = MarketOrder(volume, side)
        self.virtual_market_orders.append(order)
        return order.order_id

    def place_and_match_virtual_limit_order(self, price: int, volume: int, side: str) -> Tuple[int, Trade]:
        '''
        Places a virtual limit order.
        Gets continuously matched till empty in subsequent steps
        If "immediate" then immediately matches (only affects if crossing the book)
        '''
        order = LimitOrder(volume, price, side)
        self.virtual_limit_orders.append(order)
        order, trade = self.match_virtual_limit_order(order)
        return order.order_id, trade

    def place_and_match_virtual_market_order(self, volume: int, side: str) -> Tuple[int, Trade]:
        '''
        Places a virtual limit order.
        Gets immediately matched and then deleted, no placement for subsequent steps
        '''
        order = MarketOrder(volume, side)
        order, trade = self.match_virtual_market_order(order)
        return order.order_id, trade

    def match_virtual_limit_order(self, order: LimitOrder) -> Tuple[LimitOrder, Trade]:
        trade = Trade(order.order_id, "limit", 0, order.volume)
        volume_to_be_matched = order.volume
        if order.side == 'bid':
            # Matching a buy limit order against the ask levels
            for price, price_volume in sorted(self.lob.asks.items()):
                if order.price < price or volume_to_be_matched <= 0:
                    # print("no ask that is >= the bid order price, order.price(): "+str(order.price)+" ask price: "+str(price))
                    break
                # Only simulate the trade, do not mutate the book
                trade_volume = min(volume_to_be_matched, price_volume)
                # print("Trade a bid limit order with price: "+str(order.price)+" volume:" + str(order.volume)+" at price level "+str(price) + " which has volume: "+str(price_volume))
                volume_to_be_matched -= trade_volume
                trade.filled_volume += trade_volume
                trade.add_fill_at_price(price, trade_volume)
                order.volume -= trade_volume
                order.filled_volume += trade_volume
                # print("remaining volume: " + str(order.volume))
        elif order.side == 'ask':
            # Matching a sell limit order against the bid levels
            for price, price_volume in sorted(self.lob.bids.items(), reverse=True):
                if order.price > price or volume_to_be_matched <= 0:
                    # print("no bid that is < the ask order price, order.price(): " + str(order.price) + " ask price: " + str(price))
                    break
                # Only simulate the trade, do not mutate the book
                trade_volume = min(volume_to_be_matched, price_volume)
                # print("Trade a bid limit order with price: " + str(order.price) + " volume:" + str(order.volume) + " at price level " + str(price) + " which has volume: " + str(price_volume))
                volume_to_be_matched -= trade_volume
                trade.filled_volume += trade_volume
                trade.add_fill_at_price(price, trade_volume)
                order.volume -= trade_volume
                order.filled_volume += trade_volume
                # print("remaining volume: " + str(order.volume))
        trade.remaining_volume = volume_to_be_matched
        return order, trade

    def match_virtual_market_order(self, order: MarketOrder) -> Tuple[MarketOrder, Trade]:
        '''
        Immediately matches against LOB and returns trade
        TODO: does not consider initial matching on orders crossing spread
        '''
        filled_volume = 0
        total_cost = 0.0
        trade = Trade(order.order_id, "market", 0, order.volume)
        if order.side == 'bid':
            for price in sorted(self.lob.asks.keys()):
                if order.volume <= 0:
                    break
                available_volume = self.lob.asks[price]
                volume_to_trade = min(order.volume, available_volume)
                order.volume -= volume_to_trade
                order.filled_volume += volume_to_trade
                filled_volume += volume_to_trade
                total_cost += volume_to_trade * price
                trade.add_fill_at_price(price, volume_to_trade)
        elif order.side == 'ask':
            for price in sorted(self.lob.bids.keys(), reverse=True):
                if order.volume <= 0:
                    break
                available_volume = self.lob.bids[price]
                volume_to_trade = min(order.volume, available_volume)
                order.volume -= volume_to_trade
                order.filled_volume += volume_to_trade
                filled_volume += volume_to_trade
                total_cost += volume_to_trade * price
                trade.add_fill_at_price(price, volume_to_trade)
        trade.filled_volume = filled_volume
        trade.remaining_volume = order.volume - filled_volume
        return order, trade

    def step(self, new_bids: Dict[int, int], new_asks: Dict[int, int], timestep: int) -> Tuple[List[Trade], List[LimitOrder], List[MarketOrder]]:
        '''
        Pass new external L2 data, match virtual orders against it.
        Housekeeps virtual orders by removing them if filled.
        '''
        # LOBs
        self.timestep = timestep
        self.lob.update_lob(new_bids, new_asks)
        executed_trades = []

        # manage limit orders
        to_remove_limit_orders = []
        for order in self.virtual_limit_orders:
            initial_volume = order.volume
            _, trade = self.match_virtual_limit_order(order)
            filled_volume = initial_volume - order.volume
            if filled_volume > 0:
                trade.filled_volume = filled_volume
                trade.remaining_volume = order.volume
                executed_trades.append(trade)
                # print("backtester: fill happened during the match, filled_volume "+str(filled_volume)+" remaining volume: "+str(order.volume))
            if order.volume <= 0:
                # print("backtester: no fill happened during the match")
                to_remove_limit_orders.append(order)
        # remove empty limit orders
        for order in to_remove_limit_orders:
            self.virtual_limit_orders.remove(order)

        # manage market orders
        to_remove_market_orders = []
        for order in self.virtual_market_orders:
            initial_volume = order.volume
            _, trade = self.match_virtual_market_order(order)
            filled_volume = initial_volume - order.volume
            if filled_volume > 0:
                trade.filled_volume = filled_volume
                trade.remaining_volume = order.volume
                executed_trades.append(trade)
            to_remove_market_orders.append(order)
        # remove market orders
        for order in to_remove_market_orders:
            self.virtual_market_orders.remove(order)

        # return
        return executed_trades, to_remove_limit_orders, to_remove_market_orders

    def delete_limit_order(self, order_id: int) -> Optional[LimitOrder]:
        '''
        Instantaneously remove a limit order from the LOB
        '''
        for order in self.virtual_limit_orders:
            if order.order_id == order_id:
                self.virtual_limit_orders.remove(order)
                return order
        return None

    def delete_market_order(self, order_id: int) -> Optional[MarketOrder]:
        '''
        Instantaneously remove a market order from the LOB
        '''
        for order in self.virtual_market_orders:
            if order.order_id == order_id:
                self.virtual_market_orders.remove(order)
                return order
        return None



if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv("LOBL2.csv")
    data = data.set_index(data.iloc[:, 0]).iloc[:, 1::]
    levels = 10
    init_time = 0
    backtester = Backtester()
    data_feed = DataFeed(data, levels)
    gen = data_feed.raw_data_yield_fun(init_time)
    # get first data point
    index, new_bids, new_asks = next(gen)
    # init book
    _, _, _ = backtester.step(new_bids, new_asks, index)
    # get a price near the top
    price = backtester.lob.get_best_bid() - backtester.tick_size
    # place at that price
    order_id = backtester.place_virtual_limit_order(price, 100000000, "bid")
    all_executed_trades = []
    for  index, new_bids, new_asks in gen:
        print("iter")
        executed_trades, to_remove_limit_orders, to_remove_market_orders = backtester.step(new_bids, new_asks, index)
        all_executed_trades += executed_trades
        if len(backtester.virtual_limit_orders) == 0:
            break
    print([trade.price_levels for trade in all_executed_trades])
    assert(sum([sum(list(trade.price_levels.values())) for trade in all_executed_trades]) == 100000000)







    data = pd.read_csv("LOBL2.csv")
    data = data.set_index(data.iloc[:, 0]).iloc[:, 1::]
    levels = 10
    init_time = 0

    backtester = Backtester()
    data_feed = DataFeed(data, levels)
    gen = data_feed.raw_data_yield_fun(init_time)
    index, new_bids, new_asks = next(gen)
    # init book
    _, _, _ = backtester.step(new_bids, new_asks, index)
    # get a price near the top
    price = backtester.lob.get_best_bid() - backtester.tick_size
    # place at that price
    order_id = backtester.place_virtual_limit_order(price, 10000, "bid")
    for  index, new_bids, new_asks in gen:
        executed_trades, to_remove_limit_orders, to_remove_market_orders = backtester.step(new_bids, new_asks, index)



    # test single limit order placement that will not get completely filled
    backtester = Backtester()
    data_feed = DataFeed(data, levels)
    gen = data_feed.raw_data_yield_fun(init_time)
    # get first data point
    index, new_bids, new_asks = next(gen)
    # init book
    _, _, _ = backtester.step(new_bids, new_asks, index)
    # get a price near the top
    price = backtester.lob.get_best_bid() - backtester.tick_size
    # place at that price
    order_id = backtester.place_virtual_limit_order(price, 2709129, "ask")
    # get new data
    index, new_bids, new_asks = next(gen)
    executed_trades, to_remove_limit_orders, to_remove_market_orders = backtester.step(new_bids, new_asks, index)
    assert (executed_trades[0].remaining_volume == 1)
    assert (executed_trades[0].price_levels[36912] == 2709128)
    assert (len(to_remove_limit_orders) == 0)
    index, new_bids, new_asks = next(gen)
    executed_trades, to_remove_limit_orders, to_remove_market_orders = backtester.step(new_bids, new_asks, index)
    assert (executed_trades[0].remaining_volume == 0)
    assert (executed_trades[0].price_levels[36908] == 1)
    assert (len(to_remove_limit_orders) == 1)


    # test single market order placement that will get completely filled
    backtester = Backtester()
    data_feed = DataFeed(data, levels)
    gen = data_feed.raw_data_yield_fun(init_time)
    # get first data point
    index, new_bids, new_asks = next(gen)
    # init book
    _, _, _ = backtester.step(new_bids, new_asks, index)
    # place at that price
    order_id = backtester.place_virtual_market_order(10000, "bid")
    # get new data
    index, new_bids, new_asks = next(gen)
    executed_trades, to_remove_limit_orders, to_remove_market_orders = backtester.step(new_bids, new_asks, index)
    assert (executed_trades[0].remaining_volume == 0)
    assert (executed_trades[0].price_levels[36912] == 10000)
    assert (len(to_remove_market_orders) == 1)
    assert (len(backtester.virtual_market_orders) == 0)



