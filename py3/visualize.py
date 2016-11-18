import sys
import datetime
import os
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotdata
import subprocess
import math


def visualize_3d_data(xs,ys,zs,savefilename = None):
    "各xs,ys,zs つながり"
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
