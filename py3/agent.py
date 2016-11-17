import pandas as pd
import sys
import datetime
import os
import visualize
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

    def buy_wisely(self, price, predict, left, allow_level=4):
        "完璧に予測できるほどよくなる買い方"
        exp = self.left_money / left
        allow_exp = exp * allow_level
        if predict < allow_exp:  # ちょっと高い時は許容期待値までなら出す
            val = predict * 1.001
        else:  # 高すぎるやつは許容期待値まで
            val = allow_exp
        return self.buy(price, val)
        # 2: 541|248.39$, 627|15.41, 587|407.96$
        # 3: 553|0.74$, 628|1.13$, 603|75.27$
        # 4: 550|8.28$, 624|9.62$, 605|11.87$

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
            prices[day_key] = prices.get(day_key, []) + [
                [proc_time, row["PRICE"], int(row["ACCOUNT_ID"])]
            ]
        for k, v in prices.items():
            prices[k] = np.array(v)
        return prices

    def get_all(self, index):
        return self.prices[list(self.prices.keys())[index]]

    def get_pairs(self, index):
        return self.get_all(index)[:, 0:2]

    def get_prices(self, index):
        return self.get_all(index)[:, 1]

    def get_timestamps(self, index):
        return self.get_all(index)[:, 0]

    def get_ids():
        return self.get_all(index)[:, 2]


def simple_agent(auction, buyer):
    "一日目の中間値で二日目を投稿し続ける簡単なエージェント"
    first_mean = auction.get_prices(0).mean()
    seconds = auction.get_prices(1)
    for i, price in enumerate(auction.get_prices(1)):
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
    firsts = auction.get_all(0)
    log_xs = list(firsts[:, [0, 2]])
    log_ys = list(firsts[:, 1])
    seconds = auction.get_all(1)
    error_num = 0
    for i, (t, price, a_id) in enumerate(seconds):
        just_before = 20
        xs = log_xs[-just_before:-1]
        ys = log_ys[-just_before:-1]
        plotdata.write_spaced_data("a.dat", xs, ys)
        b_lists = [[t, _] for _ in list(set([_[1] for _ in xs]))]
        plotdata.write_spaced_data("b.dat", b_lists, [""] * len(b_lists))
        subprocess.call(["./svr", "a.dat", "--cross", "4", "--silent",
                         "--test", "b.dat", "--plot", "c.dat", "non-normalize"])
        svr_x, svr_y = plotdata.read_spaced_data("c.dat")
        predict = max(svr_y)
        if predict > 1000:  # svr作成に失敗した時は前回の値を使う
            predict = ys[-1]
        buyer.buy_wisely(price, predict, (len(seconds) - i))
        log_xs.append([t, a_id])
        log_ys.append(price)
        if predict / price < 0.8 or predict / price > 1.2:
            pre = ys[-2]
            if predict / pre < 0.8 or predict / pre > 1.2:
                error_num += 1
                print(str(t) + " : " + str(predict) + "/" +
                      str(price) + " / " + str(pre) + "(pre)")
    print(str(error_num) + " times sharp diff")
    return buyer


if __name__ == "__main__":
    assert len(sys.argv) > 1
    auction = Auction(sys.argv[1])
    print("day1:{} items".format(len(auction.get_prices(0))))
    print("day2:{} items".format(len(auction.get_prices(1))))
    agents = [simple_agent, greedy_agent,
              sorena_agent, yochi_agent, saikyou_agent,
              svr_agent
              ]
    for agent in agents:
        boughter = agent(auction, Buyer(10000))
        print("### " + agent.__name__ + " ###")
        print(boughter)
