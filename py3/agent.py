import pandas as pd
import sys
import datetime
import os
from pprint import pprint
from collections import OrderedDict, deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotdata
import random
import subprocess
import math
import time

# 一日目のデータで評価し、二日目の入札者データと合わせてオークションする
# id意識したら精度変わるかも？


class Buyer:

    def __init__(self, max_money):
        self.left_money = max_money
        self.boughts = []

    def buy(self, real_price, detected_price):
        eps = 0.0000001
        if detected_price <= 0:
            return False
        if real_price - detected_price > eps:
            return False
        if real_price - self.left_money > eps:
            return False
        self.left_money -= real_price
        self.boughts += [real_price]
        return True

    def can_buy(self, price):
        return price <= self.left_money

    def buy_wisely(self, price, predict, left):
        expected = self.left_money / left
        if predict < expected:
            self.buy(price, expected)  # ある程度安いやつは安く買う
        elif predict < expected * 2:
            self.buy(price, predict * 1.001)  # 普通のやつは普通に買う
        else:
            self.buy(price, predict * 0.5)  # 高すぎるやつは買う気を見せない

    def __str__(self):
        if len(self.boughts) == 0:
            return ""
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
            if i == 0:
                first_ts = int(time.mktime(ts.timetuple()))
            proc_time = (int(time.mktime(ts.timetuple())) -
                         first_ts) // 900 + i / 50.0
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
    prices = auction.get_prices(1).copy()
    prices.sort()
    for price in prices:
        buyer.buy(price, buyer.left_money)
    return buyer


def sorena_agent(auction, buyer):
    "一つ前の値段を言うエージェント"
    pre_val = 0
    error_num = 0
    seconds = auction.get_prices(1)
    for i, price in enumerate(seconds):
        if pre_val / price < 0.8 or pre_val / price > 1.2:
            error_num += 1
        buyer.buy_wisely(price, pre_val, (len(seconds) - i))
        pre_val = price
    print(str(error_num) + " times sharp diff")
    return buyer


def yochi_agent(auction, buyer):
    "完全予測ができるエージェント"
    seconds = auction.get_prices(1)
    for i, price in enumerate(seconds):
        buyer.buy_wisely(price, price, (len(seconds) - i))
    return buyer


def svr_agent(auction, buyer):
    "SVRの予測値そのままでやる単純エージェント"
    # データ可視化の結果、連続する一次元の時間の方が関連性がありそう
    # 毎度過去n=40件のデータの回帰分析から毎度 σ,C を作成して計測
    # 関係のない過去の σ,Cを使いまわしても精度は上がらないため
    # データ作成 -> SVR -> get
    # 545 ~ 627 | 直前n個 + 1,2,4,8,16...個前のデータも使用してみる => 微妙だった
    # 20個のみ : 557,148tsd 736$
    # 30個,36grid : 539,129tsd,834$
    # 20+10(1.1^)個 : 553, 191tsd,796$
    # 20+10(2^)個 : 522, 169tsd,1392$
    # sorena : 542, 127tsd, 635$
    firsts = auction.get_pairs(0)
    log_xs = [[_[0]] for _ in firsts]
    log_ys = [_[1] for _ in firsts]
    seconds = auction.get_pairs(1)
    error_num = 0
    for i, (t, price) in enumerate(seconds):
        just_before = 20
        xs, ys = log_xs[-just_before:-1], log_ys[-just_before:-1]
        """
        for pre_i in range(10):
            index = int(len(log_xs) - just_before - (1.0 ** pre_i))
            if index < 0:
                break
            xs.append(log_xs[index])
            ys.append(log_ys[index])
        """
        plotdata.write_spaced_data("a.dat", xs, ys)
        plotdata.write_spaced_data("b.dat", [[t]], [""])
        subprocess.call(["./svr", "a.dat", "--cross", "4", "--silent",
                         "--test", "b.dat", "--plot", "c.dat", "non-normalize"])
        svr_x, svr_y = plotdata.read_spaced_data("c.dat")
        predict = svr_y[0]
        if predict > 1000:  # svr作成に失敗した時は前回の値を使う
            predict = ys[-1]
        buyer.buy_wisely(price, predict, (len(seconds) - i))
        log_xs.append([t])
        log_ys.append(price)
        if predict / price < 0.8 or predict / price > 1.2:
            pre = ys[-2]
            if predict / pre < 0.8 or predict / pre > 1.2:
                error_num += 1
                print(str(t) + " : " + str(predict) + "/" +
                      str(price) + " / " + str(pre))
    print(str(error_num) + " times sharp diff")
    return buyer


def visualize_price_data(prices, savefilename=None):
    "x:日程,y:時間,z:価格 として可視化"
    xs, ys, zs = [], [], []
    for i, (_, ts_prices) in enumerate(prices.items()):
        for j, (ts, price) in enumerate(ts_prices):
            xs += [i]
            ys += [ts]
            zs += [price]
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
    if True:
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
        (lambda x: 14.0 + (0.1 if int(x * 50) % 2 == 0 else 0.0), [[0.5, 0.9]])
        #(lambda x: math.sin(x * 10.0), [[0.0, 0.9]])
    ]
    for f, r in f_ranges:
        make_anary_data(f, r)
    exit()

if __name__ == "__main__":
    assert len(sys.argv) > 1
    auction = Auction(sys.argv[1])
    # visualize_price_data(auction.prices)
    print("day1:{} items".format(len(auction.get_prices(0))))
    print("day2:{} items".format(len(auction.get_prices(1))))
    agents = [svr_agent]
    # agents = [simple_agent, greedy_agent,
    #          sorena_agent, yochi_agent, saikyou_agent]
    for agent in agents:
        boughter = agent(auction, Buyer(10000))
        print(boughter)
