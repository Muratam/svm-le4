#! /usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
from mpl_toolkits.mplot3d import Axes3D

__doc__ = """{f}
Usage:
    {f} <filename> [<basefilename>] [--save <savefilename>] [--3d | --2d | --1d]
    {f} (-h | --help)
Options:
    --save      save output as png file
    --3d        plot 3d data
    --2d        plot 2d data
    --1d        plot 1d data
    -h --help   show this help.
""".format(f=__file__)


def load_spaced_data(lines):
    x, y = [], []
    for line in lines:
        xk = [_ for _ in line.split(" ") if _ != ""]
        x.append([float(x) for x in xk[0:-1]])
        y.append(float(xk[-1]))
    assert len(x) == len(y)
    return np.array(x), np.array(y)


def plot2d(x, y, base_x, base_y, save_file_name=None):
    def separate(xs, ys):
        x1p = xs[ys[:] > 0, 0]
        x2p = xs[ys[:] > 0, 1]
        x1m = xs[ys[:] < 0, 0]
        x2m = xs[ys[:] < 0, 1]
        return x1p, x2p, x1m, x2m
    x1p, x2p, x1m, x2m = separate(base_x, base_y)
    x1pg, x2pg, x1mg, x2mg = separate(x, y)
    x1s = [x1pg, x1mg, x1p, x1m]
    x2s = [x2pg, x2mg, x2p, x2m]
    cs = ["red", "blue", "red", "blue"]
    ss = [10, 10, 30, 30]
    markers = ["x", "x", "o", "o"]
    labels = ["+1", "-1", "+1", "-1"]
    for i in range(len(x1s)):
        plt.scatter(x1s[i], x2s[i], c=cs[i], s=ss[i],
                    marker=markers[i], label=labels[i])
    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.title(result_str)
    if save_file_name:
        plt.savefig(save_file_name)
    else:
        plt.show()


def plot1d(x, y, base_x, base_y, save_file_name=None):
    x = [_[0] for _ in x]
    base_x = [_[0] for _ in base_x]
    #f = interp1d(x, y, kind="cubic")
    plt.plot(x, y)
    plt.plot(base_x, base_y)
    plt.show()


def plot3d(x, y, base_x, base_y, save_file_name=None, plot_type3d="contour"):
    # 散布図以外は mesh 変換処理を書く必要があるため保留。
    X1 = np.r_[x[:, 0], base_x[:, 0]]
    X2 = np.r_[x[:, 1], base_x[:, 1]]
    Y = np.r_[y, base_y]
    s = np.r_[[10] * len(y), [40] * len(base_y)]
    Axes3D(plt.figure()).scatter3D(X1, X2, Y,
                                   marker='x', c='b', s=s, alpha=0.4)
    plt.show()


if __name__ == "__main__":
    args = docopt(__doc__)
    with open(args["<filename>"], "r") as f:
        x, y = load_spaced_data(f.readlines())
    if args["<basefilename>"]:
        with open(args["<basefilename>"], "r") as f:
            base_x, base_y = load_spaced_data(f.readlines())
        minval, maxval = base_x.min(0), base_x.max(0)
        base_x = (base_x - minval) / (maxval - minval)  # normalize
    else:
        base_x, base_y = np.array([x[0]]), np.array([y[0]])
    if args["--3d"]:
        plot3d(x, y, base_x, base_y, args["<savefilename>"])
    elif args["--1d"]:
        plot1d(x, y, base_x, base_y, args["<savefilename>"])
    else:
        plot2d(x, y, base_x, base_y, args["<savefilename>"])