import pandas as pd
import sys
import datetime
import os


def csv2tsprice(filename):
    df = pd.read_csv(filename)
    outputfile = os.path.splitext(filename)[0] + ".dat"
    with open(outputfile, "w") as f:
        deltaday = 60 * 60 * 24
        for i, row in df.iterrows():
            ts = datetime.datetime.strptime(
                row["TIMESTAMP"], "%m/%d/%Y %H:%M:%S").timestamp()
            f.write(str(ts % deltaday) + " " + str(row["PRICE"]) + "\n")

if __name__ == "__main__":
    assert len(sys.argv) > 1
    csv2tsprice(sys.argv[1])
