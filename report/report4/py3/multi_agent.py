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
import visualize
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
        if predict < exp:
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
        self.log_ys = list(self.firsts[:, 1])
        self.c = None
        self.p = None

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
            vals = [method(agent, left) for agent, method in agents]
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
                agent.teach_result(result_price)
            if a_id > len(agents):
                a_id = -1
            if show_process:
                print(" ".join([str(_) for _ in [a_id, t, result_price]]))
        return agents

    def buy_simple(self, left):
        "一日目の中間値で二日目を投稿し続ける簡単なエージェント"
        return self.first_mean

    def buy_greedy(self, left):
        "一日目の平均の5倍の額で二日目を書い続ける貪欲に買いまくりたいエージェント"
        val = min(self.first_mean * 5, self.buyer.left_money)
        return val

    def buy_sorena(self, left):
        "一つ前の値段を言うエージェント"
        return self.buyer.expected(self.buyer.pre_val, left)

    def buy_svr(self, left):
        "SVRの予測値そのままでやるエージェント"
        dim = 10
        teacher_num = 100
        # or (len(self.log_ys) % sample_dim == 0):
        if (self.c == None or self.p == None):
            self.c, self.p = visualize.create_svr(
                self.log_ys, dim, "a.dat", teacher_num)
        predict = visualize.test_svr(
            self.log_ys, dim, self.c, self.p, "a.dat", teacher_num)
        return self.buyer.expected(predict, left)

    def teach_result(self, price):
        self.log_ys.append(price)
        self.buyer.tell_price(price)

    agent_dict = {
        "@simple": buy_simple,
        "@greedy": buy_greedy,
        "@sorena": buy_sorena,
        "@svr": buy_svr,
    }


def max_expected(prices, left_price):
    prices = prices.copy()
    prices.sort()
    boughts = 0
    for price in prices:
        if left_price - price < 0:
            break
        left_price -= price
        boughts += 1
    return boughts, prices[-1]


def main(args):
    allowed_agents = " ".join(Agent.agent_dict.keys())
    if len(args) <= 1:
        return print(
            "Usage:\n    python3 " + __file__ + " <filename> " +
            "(" + allowed_agents + " ...) " +
            "[--visualize-prices] [--show-process]"
        )
    methods = [Agent.agent_dict[_] for _ in args if _ in Agent.agent_dict]
    auction = Auction(args[1])
    if "--visualize-prices" in args:
        return auction.visualize_prices()
    if len(methods) < 1:
        return print("no method selected !! (" + allowed_agents + ")")
    show_process = "--show-process" in args
    agents = Agent.do_multi_auction(methods, auction, 10000, show_process)
    if not show_process:
        print("day1:{} items".format(len(auction.get_prices(0))))
        print("day2:{} items".format(len(auction.get_prices(1))))
        m1, m2 = max_expected(auction.get_prices(1), 10000)
        print("data max buy:" + str(m1) + "items")
        print("data max price:" + str(m2))
        for agent, method in agents:
            print(method.__name__)
            print(agent.buyer)

if __name__ == "__main__":
    main(sys.argv)
