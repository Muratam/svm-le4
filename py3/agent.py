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

# 一日目のデータで評価し、二日目の入札者データと合わせてオークションする
# 購入機構 I/Oを分離したい


class Buyer:

    def __init__(self, max_money):
        self.left_money = max_money
        self.boughts = []

    def buy(self, real_price, detected_price):
        if detected_price < real_price:
            return False
        if real_price > self.left_money:
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
    "一日目は無視して二日目は持ってるお金の最高額を言い続けるエージェント"
    for price in auction.get_prices(1):
        buyer.buy(price, buyer.left_money)
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
    pass


def svr_agent(prices, max_money):
    "SVRの予測値そのままでやる単純エージェント"
    # 一日目のデータを遡ってmin(一日目のデータ数,500)個分で検定して o,c を決定
    # その件数毎回過去 n 個のデータ,σ C から毎回評価機を作成する
    # 検定の方法は様々
    # まずどのくらいの精度で予測できるかを計測してみるべき
    # 連続的な一次元データとする
    # 先にデータを可視化してみるべき => 分析の結果、連続する時間の方が関連性がありそう
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


def visualize_svr_anary_function(sample_datas, real_datas, savefilename=None):
    """ datas := [[x1,y1],...,[xn,yn]]
    [0,1]の範囲のsample_datasからsvrを作成し、
    real_datasの各xに対するf(x) と 実際の y をプロットする。
    こうすることでSVR自体の能力を確認する"""


if __name__ == "__main__":
    assert len(sys.argv) > 1
    auction = Auction(sys.argv[1])
    visualize_price_data(auction.prices)
    exit()
    print("day1:{} items".format(len(auction.get_prices(0))))
    print("day2:{} items".format(len(auction.get_prices(1))))
    agents = [simple_agent, greedy_agent, saikyou_agent]
    for agent in agents:
        boughter = agent(auction, Buyer(10000))
        print(boughter)
