import os
from api_client import BinanceFuturesClient

def main():
    print("=== AntiFOMO BINANCE FUTURES TEST ===")
    client = BinanceFuturesClient(testnet=False)

    print("\n[1] Ping:")
    print(client.ping())

    print("\n[2] Server time:")
    print(client.get_server_time())

    print("\n[3] BTC price:")
    print(client.get_price("BTCUSDT"))

    print("\n[4] Orderbook check:")
    ob = client.get_order_book("BTCUSDT", limit=10)
    print(ob)

if __name__ == "__main__":
    main()
