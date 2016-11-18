import pandas as pd
import numpy as np
import time
import datetime
from collections import OrderedDict
import visualize

class Auction:

    def __init__(self, filename):
        self.prices = Auction.__get_datas_by_date(filename)

    def __get_datas_by_date(filename):
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
            row = [proc_time, row["PRICE"], int(row["ACCOUNT_ID"])]
            prices[day_key] = prices.get(day_key,[]) + [row]
        for k, v in prices.copy().items():
            prices[k] = np.array(v)
        return prices

    def visualize_prices(self):
        prices,tss,days =[],[],[]
        for day,(_,vs) in enumerate(self.prices.items()):
            for i,(ts,price,a_id) in enumerate(vs):
                prices.append(price)
                tss.append(ts)
                days.append(day)
        visualize.visualize_3d_data(days,tss,prices)

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
