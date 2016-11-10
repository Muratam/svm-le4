import pandas as pd
import sys
import datetime
import os
from pprint import pprint
from collections import OrderedDict
import numpy as np

# id000{1,5} : 2500ずつ
# id0002 : 一日目9000 : 二日目1000(擦り切れ)
# id000{3,4} : 100程度
# 一日目のデータで評価し、二日目の入札者データと合わせてオークションする


def get_datas_by_date(filename):
    df = pd.read_csv(filename)
    prices = OrderedDict()
    for i, row in df.iterrows():
        ts = datetime.datetime.strptime(row["TIMESTAMP"], "%m/%d/%Y %H:%M:%S")
        day_key = ts.strftime("%Y/%m/%d")
        proc_time = int((ts - datetime.datetime(ts.year,
                                                ts.month, ts.day)).total_seconds()) // 900
        prices[day_key] = prices.get(day_key, []) + [[proc_time, row["PRICE"]]]
    for k, v in prices.items():
        prices[k] = np.array(v)
    return prices


def simple_agent(prices, max_money):
    # 一日目の中間値で二日目を投稿し続ける簡単なエージェント
    firsts = prices[list(prices.keys())[0]]
    seconds = prices[list(prices.keys())[1]]
    first_mean = firsts[:, 1].mean()
    boughts = []
    left_money = max_money
    for price in seconds[:, 1]:
        if left_money - price < 0:
            continue
        if price > first_mean:
            continue
        elif price < first_mean:
            left_money -= price
            boughts += [price]
        else:
            print("!!")
    return boughts, len(seconds), left_money


def greedy_agent(prices, max_money):
    # 一日目は無視して二日目は持ってるお金の最高額を言い続けるエージェント
    seconds = prices[list(prices.keys())[1]]
    boughts = []
    left_money = max_money
    for price in seconds[:, 1]:
        if left_money - price < 0:
            continue
        left_money -= price
        boughts += [price]
    return boughts, len(seconds), left_money


def saikyou_agent(prices, max_money):
    # 未来予知ができるので二日目安い順に全て買える理論上最強のエージェント
    second_prices = prices[list(prices.keys())[1]][:, 1]
    second_prices.sort()
    boughts = []
    left_money = max_money
    for price in second_prices:
        if left_money - price < 0:
            continue
        left_money -= price
        boughts += [price]
    return boughts, len(second_prices), left_money


if __name__ == "__main__":
    assert len(sys.argv) > 1
    prices = get_datas_by_date(sys.argv[1])
    print("(day1:{} items)".format(len(prices[list(prices.keys())[0]])))
    agents = simple_agent, greedy_agent, saikyou_agent
    for agent in agents:
        boughts, chance, left_money = agent(prices, 10000)
        print("{} / {} | left {} \nmax{} / min{}".format(
            len(boughts), chance, left_money,
            max(boughts), min(boughts)
        ))
