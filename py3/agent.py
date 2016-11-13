import pandas as pd
import sys
import datetime
import os
from pprint import pprint
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotdata
import random
import subprocess
import math

# 一日目のデータで評価し、二日目の入札者データと合わせてオークションする
# 購入機構 I/Oを分離したい


class Buyer:

    def __init__(self, max_money):
        self.left_money = max_money
        self.boughts = []

    def buy(self, real_price, detected_price):
        eps = 0.001
        if real_price - detected_price > eps:
            return False
        if real_price - self.left_money > eps:
            return False
        if abs(real_price - detected_price) < eps:
            if random.randint(0, 1) == 0:
                return False
        self.left_money -= real_price
        self.boughts += [real_price]
        return True

    def can_buy(self, price):
        return price <= self.left_money

    def __str__(self):
        return "buy {} | left {}$ \nmax{} | min{}".format(
            len(self.boughts), self.left_money,
            max(self.boughts), min(self.boughts)
        )


class Auction:

    def __init__(self, filename):
        self.prices = Auction.get_datas_by_date(filename)

    def get_datas_by_date(filename):
        df = pd.read_csv(filename)
        prices = OrderedDict()
        for i, row in df.iterrows():
            ts = datetime.datetime.strptime(
                row["TIMESTAMP"], "%m/%d/%Y %H:%M:%S")
            day_key = ts.strftime("%Y/%m/%d")
            proc_time = int((ts - datetime.datetime(ts.year,
                                                    ts.month, ts.day)).total_seconds()) // 900
            prices[day_key] = prices.get(
                day_key, []) + [[proc_time, row["PRICE"]]]
        for k, v in prices.items():
            prices[k] = np.array(v)
        return prices

    def get_pairs(self, index):
        return self.prices[list(self.prices.keys())[index]]

    def get_prices(self, index):
        return self.get_pairs(index)[:, 1]

    def get_timestamps(self, index):
        return self.get_pairs(index)[:, 0]


def simple_agent(auction, buyer):
    "一日目の中間値で二日目を投稿し続ける簡単なエージェント"
    first_mean = auction.get_prices(0).mean()
    for price in auction.get_prices(1):
        buyer.buy(price, first_mean)
    return buyer


def greedy_agent(auction, buyer):
    "一日目の平均の5倍の額で二日目を書い続ける貪欲に買いまくりたいエージェント"
    first_mean = auction.get_prices(0).mean()
    for price in auction.get_prices(1):
        val = min(first_mean * 5, buyer.left_money)
        buyer.buy(price, val)
    return buyer


def saikyou_agent(auction, buyer):
    "未来予知ができるので二日目安い順に全て買える理論上最強のエージェント"
    prices = auction.get_prices(1)
    prices.sort()
    for price in prices:
        buyer.buy(price, buyer.left_money)
    return buyer


def sorena_agent(auction, buyer):
    "一つ前の値段を言うエージェント"
    pre_val = 0
    for price in auction.get_prices(1):
        buyer.buy(price, pre_val)
        pre_val = price * 1.01
    return buyer


def svr_agent(prices, max_money):
    "SVRの予測値そのままでやる単純エージェント"
    # データ可視化の結果、連続する一次元の時間の方が関連性がありそう
    # 毎度過去n=20件のデータの回帰分析から毎度 σ,C を作成して計測
    # 関係のない過去の σ,Cを使いまわすことはよくないため。
    pass


def visualize_price_data(prices, savefilename=None):
    "x:日程,y:時間,z:価格 として可視化"
    xs, ys, zs = [], [], []
    for i, (_, ts_prices) in enumerate(prices.items()):
        for j, (ts, price) in enumerate(ts_prices):
            xs += [i]
            ys += [ts + j / 100.0]
            zs += [price]
    print(ys)
    Axes3D(plt.figure()).plot3D(xs, ys, zs)
    if savefilename:
        plt.savefig(save_file_name)
    else:
        plt.show()


def make_anary_data(f, ranges, grid_num=50):
    "[0,1]の範囲のsample_datasからsvrを作成し、SVR自体の能力を確認する"
    xs, ys = [], []
    rxs, rys = [], []
    for i in range(grid_num + 1):
        x = i / grid_num
        for l, r in ranges:
            if l <= x and x <= r:
                xs.append([x])
                ys.append(f(x))
                break
        rxs.append([x])
        rys.append(f(x))
    plotdata.write_spaced_data("a.dat", xs, ys)
    closs = False
    if closs:
        subprocess.call(["./svr", "a.dat", "--cross", "4",
                         "--plot", "b.dat", "non-normalize"])
    else:
        subprocess.call(["./svr", "a.dat", "--c", "776", "--p", "0.757",
                         "--plot", "b.dat", "non-normalize"])
    svr_x, svr_y = plotdata.read_spaced_data("b.dat")
    plotdata.plot1d(svr_x, svr_y, rxs, rys, save_file_name=None)


def test_make_anary_data():
    "一次元SVRの推定能力をテストしまくる => 結構すごいことが分かる"
    f_ranges = [
        (lambda x: math.sin(x * 10.0), [[0.0, 0.9]])
    ]
    for f, r in f_ranges:
        make_anary_data(f, r)

if __name__ == "__main__":
    test_make_anary_data()
    assert len(sys.argv) > 1
    auction = Auction(sys.argv[1])
    # visualize_price_data(auction.prices)
    # exit()
    print("day1:{} items".format(len(auction.get_prices(0))))
    print("day2:{} items".format(len(auction.get_prices(1))))
    agents = [simple_agent, greedy_agent, saikyou_agent, sorena_agent]
    for agent in agents:
        boughter = agent(auction, Buyer(10000))
        print(boughter)
