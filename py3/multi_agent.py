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
from auction import Auction

# 一日目のデータで評価し、二日目の入札者データと合わせてオークションする


class Buyer:

    def __init__(self, max_money):
        self.left_money = max_money
        self.boughts = []
        self.pre_val = 0

    def expected(self, predict, left, allow_level=3):
        "完璧に予測できるほどよくなる買い方"
        exp = self.left_money / left
        allow_exp = exp * allow_level
        if predict < exp:  # ちょっと高い時は許容期待値までなら出す
            val = exp
        elif predict < exp * allow_level:
            val = predict * 1.1
        else:  # 高すぎるやつは許容期待値まで
            val = allow_exp
        return val

    def buy(self, price):
        eps = 0.0000001
        if price - self.left_money > eps:
            return False
        self.left_money -= price
        self.boughts += [price]
        return True

    def tell_price(self, price):
        self.pre_val = price

    def __str__(self):
        if len(self.boughts) == 0:
            return ""
        return "buy {} | left {}$ \nmax{} | min{}".format(
            len(self.boughts), self.left_money,
            max(self.boughts), min(self.boughts)
        )

# id0001,一人対戦のときは 理論上628個買えることがわかっている


class Agent:

    def __init__(self, firsts, first_mean, buyer):
        self.firsts = firsts
        self.first_mean = first_mean
        self.buyer = buyer
        self.log_xs = list(self.firsts[:, [0, 2]])
        self.log_ys = list(self.firsts[:, 1])

    def do_multi_auction(methods, auction, max_money, show_process=False):
        agents = []
        first_mean = auction.get_prices(0).mean()
        firsts = auction.get_all(0)
        seconds = auction.get_all(1)
        for method in methods:
            agent = Agent(firsts, first_mean, Buyer(max_money))
            agents.append([agent, method])
        for i, (t, price, a_id) in enumerate(seconds):
            left = len(seconds) - i
            vals = [method(agent, left, t) for agent, method in agents]
            vals = [int(_ * 1000) / 1000 for _ in vals]
            prices = vals.copy()
            prices.append(price)
            prices.sort()
            result_price = prices[-2]
            buyer_count = 0
            boughters = [(agent, method, j)
                         for j, (agent, method) in enumerate(agents)
                         if vals[j] == max(prices)]
            if len(boughters) > 0:
                random.shuffle(boughters)
                agent, method, j = boughters[0]
                agent.buyer.buy(price)
                a_id = j
                buyer_count += 1
            for j, (agent, method) in enumerate(agents):
                agent.teach_result(t, a_id, result_price)
            if a_id > len(agents):
                a_id = -1
            if show_process:
                # price
                print(" ".join([str(_) for _ in [a_id, result_price, t]]))
        return agents

    def buy_simple(self, left, t):
        "一日目の中間値で二日目を投稿し続ける簡単なエージェント"
        return self.first_mean

    def buy_greedy(self, left, t):
        "一日目の平均の5倍の額で二日目を書い続ける貪欲に買いまくりたいエージェント"
        val = min(self.first_mean * 5, self.buyer.left_money)
        return val

    def buy_sorena(self, left, t):
        "一つ前の値段を言うエージェント"
        return self.buyer.expected(self.buyer.pre_val, left)

    def buy_svr(self, left, t):
        "SVRの予測値そのままでやるエージェント"
        just_before = 20
        xs = self.log_xs[-just_before:-1]
        ys = self.log_ys[-just_before:-1]
        plotdata.write_spaced_data("a.dat", xs, ys)
        b_lists = [[t, _] for _ in list(set([_[1] for _ in xs]))]
        plotdata.write_spaced_data("b.dat", b_lists, [""] * len(b_lists))
        subprocess.call(["./svr", "a.dat", "--cross", "4", "--silent",
                         "--test", "b.dat", "--plot", "c.dat", "non-normalize"])
        svr_x, svr_y = plotdata.read_spaced_data("c.dat")
        predict = max(svr_y)
        if predict > 1000:  # svr作成に失敗した時は前回の値を使う
            predict = ys[-1]
        return self.buyer.expected(predict, left)

    def teach_result(self, t, a_id, price):
        self.log_xs.append([t, a_id])
        self.log_ys.append(price)
        self.buyer.tell_price(price)

    agent_dict = {
        "@simple": buy_simple,
        "@greedy": buy_greedy,
        "@sorena": buy_sorena,
        "@svr": buy_svr,
    }


def main(args):
    allowed_agents = " ".join(Agent.agent_dict.keys())
    if len(args) <= 1:
        return print(
            "Usage:\n    python3 " + __file__ + " <filename> " +
            "(" + allowed_agents + " ...) " +
            "[--visualize-prices] [--show-process]"
        )
    methods = [Agent.agent_dict[_] for _ in args if _ in Agent.agent_dict]
    if len(methods) < 1:
        return print("no method selected !! (" + allowed_agents + ")")
    auction = Auction(args[1])
    if "--visualize-prices" in args:
        return auction.visualize_prices()
    show_process = "--show-process" in args
    agents = Agent.do_multi_auction(methods, auction, 10000, show_process)
    if not show_process:
        print("day1:{} items".format(len(auction.get_prices(0))))
        print("day2:{} items".format(len(auction.get_prices(1))))
        for agent, method in agents:
            print(method.__name__)
            print(agent.buyer)

if __name__ == "__main__":
    main(sys.argv)
