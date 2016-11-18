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
import random


def visualize_3d_data(xs, ys, zs, savefilename=None):
    "各xs,ys,zs つながり"
    Axes3D(plt.figure()).plot3D(xs, ys, zs)
    if savefilename:
        plt.savefig(save_file_name)
    else:
        plt.show()


def create_current_data(ys, dim, filename, teacher_num):
    res_xs, res_ys = [], []
    assert teacher_num <= len(ys)
    for i in range(len(ys) - teacher_num, len(ys)):
        res_x = [ys[i - dim + j] for j in range(dim)]
        res_xs.append(res_x)
        res_ys.append(ys[i])
    plotdata.write_spaced_data(filename, res_xs, res_ys)


def create_svr(ys, dim, filename, teacher_num):
    "各データからSVR計算用の形式に変換し、c,pを得る"
    create_current_data(ys, dim, filename, teacher_num)
    subprocess.call(["./svr", filename, "--cross", "10", "--silent",
                     "--plot-c-p", "tmp.dat", "non-normalize"])
    c, p = plotdata.read_spaced_data("tmp.dat")
    return c[0][0], p[0]


def test_svr(ys, dim, c, p, svr_filename, teacher_num):
    create_current_data(ys, dim, svr_filename, teacher_num)
    plotdata.write_spaced_data("tmp.dat", [ys[-dim:]], [""])
    subprocess.call(["./svr", svr_filename, "--c", str(c), "--p", str(p), "--silent",
                     "--test", "tmp.dat", "--plot", "tmp2.dat", "non-normalize"])
    _, svr_y = plotdata.read_spaced_data("tmp2.dat")
    return svr_y[0]


def make_anary_data(f, end, grid_num=200):
    "[0,1]の範囲のsample_datasからsvrを作成し、SVR自体の能力を確認する"
    ys = [f(i / grid_num) for i in range(grid_num) if i / grid_num <= end]
    dim = min(10, grid_num / 10)
    teacher_num = len(ys)
    c, p = create_svr(ys, dim, "a.dat", teacher_num)
    svr_ys = []
    rxs, rys = [], []
    for i in range(grid_num + 1):
        x = i / grid_num
        rxs.append([x])
        rys.append(f(x))
        if x <= end:
            svr_ys.append(f(x))
        else:
            y = test_svr(rys, dim, c, p, "a.dat", teacher_num)
            print(str(y), end=" ", flush=True)
            svr_ys.append(y)
    plotdata.plot1d(rxs, svr_ys, rxs, rys, save_file_name=None)


def test_make_anary_data():
    "一次元SVRの推定能力をテストしまくる => 結構すごいことが分かる"
    f_ranges = [
        (lambda x: x + 0.2 * (round(x) + random.random()), 0.3)
    ]
    for f, end in f_ranges:
        make_anary_data(f, end)

if __name__ == "__main__":
    test_make_anary_data()
