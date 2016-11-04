import pandas as pd
import sys
import datetime
import os


def csv2tsprice(filename):
    df = pd.read_csv(filename)
    outputfile = os.path.splitext(filename)[0] + ".dat"
    with open(outputfile, "w") as f:
        prices_dict = {}
        deltaday = 60 * 60 * 24
        for i, row in df.iterrows():
            ts = datetime.datetime.strptime(
                row["TIMESTAMP"], "%m/%d/%Y %H:%M:%S").timestamp()
            key = ts % deltaday
            prices_dict[key] = prices_dict.get(key, []) + [row["PRICE"]]
            #f.write(str(ts % deltaday) + " " + str(row["PRICE"]) + "\n")
        prices = []
        for k, v in prices_dict.items():
            v.sort()
            prices += [(k, v[len(v) // 2])]
        prices.sort(key=lambda x: x[0])
        for k, v in prices:
            f.write(str(k) + " " + str(v) + "\n")

if __name__ == "__main__":
    assert len(sys.argv) > 1
    csv2tsprice(sys.argv[1])
