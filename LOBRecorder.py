import pandas as pd
import websocket
import json
import numpy as np

# globals AMM
data = None
mode = "live" #"record", "live"
bitstamp_endpoint = 'wss://ws.bitstamp.net'
endpoint_type = 'order_book_ethusd'



book_deliver_size = 10
span = 500
time_span = 1000
rows = span * 2
cols = time_span
mid = 0



book_timestamp = 0
book = {}

def subscribe_marketdata(ws):
    params = {
        'event': 'bts:subscribe',
        'data': {
            'channel': endpoint_type
        }
    }
    market_depth_subscription = json.dumps(params)
    ws.send(market_depth_subscription)


def on_open(ws):
    print('web-socket connected.')
    subscribe_marketdata(ws)



def on_error(ws, msg):
    print(msg)

prices = np.array([0])
volumes = np.array([0])
top_bid = 0
top_ask = 0
span_min = 0
span_max = 0
price_mult = 10
vol_mult = 1e7

all_bids = {}
all_asks = {}
all = {}
def parse_book(ws, dat):
    global data
    global prices
    global volumes
    global mid
    dat = json.loads(dat)
    data = dat #dat = data
    book_timestamp = int(dat['data']["microtimestamp"])
    prices_bids = np.array([int(float(i)*price_mult) for i,j in dat['data']["bids"][0:book_deliver_size]])
    volumes_bids = np.array([int(float(j)*vol_mult) for i,j in dat['data']["bids"][0:book_deliver_size]])
    prices_asks = np.array([int(float(i) * price_mult) for i, j in dat['data']["asks"][0:book_deliver_size]])
    volumes_asks = np.array([int(float(j)*vol_mult) for i, j in dat['data']["asks"][0:book_deliver_size]])
    bids = np.empty((prices_bids.size + volumes_bids.size,), dtype=int)
    bids[0::2] = prices_bids
    bids[1::2] = volumes_bids
    asks = np.empty((prices_asks.size + volumes_asks.size,), dtype=int)
    asks[0::2] = prices_asks
    asks[1::2] = volumes_asks
    all_bids.update({book_timestamp: bids})
    all_asks.update({book_timestamp: asks})
    all.update({book_timestamp: np.concatenate([bids, asks], axis=0)})
    print(book_timestamp)


def on_message_live(ws, dat):
    data = json.loads(dat)
    parse_book(data)


if __name__ == '__main__':
    marketdata_ws = websocket.WebSocketApp(bitstamp_endpoint, on_open=on_open, on_message=parse_book, on_error=on_error)
    marketdata_ws.run_forever()
    data = pd.DataFrame.from_dict(all, orient='index')
    data.index.name = "timestamp"
    data.to_csv("LOB_L2_1.csv", index="timestamp")





