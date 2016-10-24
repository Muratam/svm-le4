import sys
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

__doc__ = """{f}
Usage:
    {f} [<filename>] [--save <savefilename>]
    {f} (-h | --help)
Options:
    --save      save output as png file
    -h --help   show this help.
""".format(f=__file__)


def load_spaced_data(lines):
    x, y = [], []
    for line in lines:
        xk = line.split(" ")
        x.append([float(x) for x in xk[0:-1]])
        y.append(float(xk[-1]))
    assert len(x) == len(y)
    return np.array(x), np.array(y)


def plot(x, y, base_x, base_y, save_file_name=None):
    def separate(xs, ys):
        n = len(ys)
        assert len(xs) == n
        x1p = [xs[i][0] for i in range(n) if ys[i] > 0]
        x2p = [xs[i][1] for i in range(n) if ys[i] > 0]
        x1m = [xs[i][0] for i in range(n) if ys[i] < 0]
        x2m = [xs[i][1] for i in range(n) if ys[i] < 0]
        return x1p, x2p, x1m, x2m
    x1p, x2p, x1m, x2m = separate(base_x, base_y)
    x1pg, x2pg, x1mg, x2mg = separate(x, y)
    x1s = [x1pg, x1mg, x1p, x1m]
    x2s = [x2pg, x2mg, x2p, x2m]
    cs = ["red", "blue", "red", "blue"]
    ss = [10, 10, 30, 30]
    markers = ["x", "x", "o", "o"]
    labels = ["+1", "-1", "+1", "-1"]
    plt.cla()
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

if __name__ == "__main__":
    args = docopt(__doc__)
    if len(sys.argv) > 1:
        with open(args["<filename>"], "r") as f:
            base_x, base_y = load_spaced_data(f.readlines())
        minval, maxval = base_x.min(0), base_x.max(0)
        base_x = (base_x - minval) / (maxval - minval)  # normalize
    else:
        base_x, base_y = [], []
    x, y = load_spaced_data(sys.stdin.readlines())
    plot(x, y, base_x, base_y, args["<savefilename>"])
