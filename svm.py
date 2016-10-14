import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import time
import matplotlib
import random
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

kernels = {
    "linear": lambda: lambda x, y: x.dot(y),
    "polynomial": lambda d=2: lambda x, y: (1 + x.dot(y)) ** d,
    "sigmoid": lambda a=3, b=4: lambda x, y: np.tanh(a * x.dot(y) + b),
    "gauss": lambda sigma=10:
        (lambda x, y: np.exp(-0.5 * np.square(np.linalg.norm(x - y) / sigma))),
}


def solve(x, y, kernel):
    x = np.array(x)
    y = np.array(y)
    n = len(y)
    assert n == len(x)
    P = cvxopt.matrix([[0.0] * n] * n)
    for k in range(n):
        for l in range(n):
            P[k, l] = y[k] * y[l] * kernel(x[k], x[l]) if k >= l else P[l, k]
    q = cvxopt.matrix([-1.0] * n)
    G = cvxopt.spdiag([-1.0] * n)
    h = cvxopt.matrix([0.0] * n)
    A = cvxopt.matrix([[0.0] * 1] * n)
    A[0, :] = y
    b = cvxopt.matrix(0.0)
    sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={"show_progress": False})
    a = np.array([(x if x > 0.0000001 else 0) for x in sol["x"]])
    ok_indexes = [i for i in range(n) if abs(a[i]) > 0]
    ok_coes = [(a[i] * y[i], x[i]) for i in ok_indexes]
    kernel_dot_to_w = lambda n_x: sum(
        [co[0] * kernel(co[1], n_x) for co in ok_coes])
    max_index = a.argmax()
    theta = kernel_dot_to_w(x[max_index]) - y[max_index]
    print("α : " + str(a))
    print("θ : " + str(theta))
    f = lambda n_x: 1.0 if kernel_dot_to_w(n_x) - theta > 0 else -1.0
    print("passed :" + str(sum([1 for i in range(n)
                                if f(x[i]) == y[i]])) + " / " + str(len(y)))
    f_str = "f(x) = "
    for i in ok_indexes:
        co = a[i] * y[i]
        f_str += (" +" if co > 0 else " ") + \
            str(co) + "*K(" + str(x[i].tolist()) + ",x)"
    print(f_str)
    return f


def plot_3D(f, x, num, plot_type3D="scatter"):
    # x is used for this range
    x1 = np.linspace(min([_[0] for _ in x]), max([_[0] for _ in x]), num)
    x2 = np.linspace(min([_[1] for _ in x]), max([_[1] for _ in x]), num)
    x1mesh, x2mesh = np.meshgrid(x1, x2)
    Z = x1mesh.copy()
    for i in range(num):
        for j in range(num):
            Z[i, j] = f([x1mesh[i, j], x2mesh[i, j]])
    if plot_type3D == "contourf":
        Axes3D(plt.figure()).contourf3D(x1mesh, x2mesh, Z)
    elif plot_type3D == "contour":
        Axes3D(plt.figure()).contourf3D(x1mesh, x2mesh, Z)
    else:
        Axes3D(plt.figure()).scatter3D(x1mesh.ravel(), x2mesh.ravel(),
                                       [f(x) for x in np.c_[x1mesh.ravel(), x2mesh.ravel()]])
    return plt.show()


def plot_f(f, x, y, num=50, plot_type3D="scatter"):
    def separate(xs, ys):
        n = len(ys)
        assert len(xs) == n
        x1p = [xs[i][0] for i in range(n) if ys[i] > 0]
        x2p = [xs[i][1] for i in range(n) if ys[i] > 0]
        x1m = [xs[i][0] for i in range(n) if ys[i] < 0]
        x2m = [xs[i][1] for i in range(n) if ys[i] < 0]
        return x1p, x2p, x1m, x2m
    if plot_type3D:
        return plot_3D(f, x, num, plot_type3D)
    x1 = np.linspace(min([_[0] for _ in x]), max([_[0] for _ in x]), num)
    x2 = np.linspace(min([_[1] for _ in x]), max([_[1] for _ in x]), num)
    x1mesh, x2mesh = np.meshgrid(x1, x2)  # x1ij x2ij
    xgs = np.c_[x1mesh.ravel(), x2mesh.ravel()]
    ygs = [f(x) for x in xgs]
    x1p, x2p, x1m, x2m = separate(x, y)
    x1pg, x2pg, x1mg, x2mg = separate(xgs, ygs)
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


def load_x_y(fileName):
    x, y = [], []
    try:
        with open(fileName, "r") as f:
            for line in f.readlines():
                xk = line.split(" ")
                x.append([float(x) for x in xk[0:-1]])
                y.append(float(xk[-1]))
    except:
        print("invalid file " + fileName + "!!!")
        exit(0)
    return x, y


def parse_argv():
    __doc__ = """{f}
Usage:
    {f} <filename> [-m | --method <method>] [--plot]
    {f} [-h | --help]
Options:
    -m --method    {methods} (default:gauss)
    --plot         show plotted graph (with matplotlib)
    -h --help      show this help.
""".format(f=__file__, methods=str(",".join(kernels.keys())))
    from docopt import docopt
    args = docopt(__doc__)
    if not args["<method>"]:
        args["<method>"] = "gauss"
    else:
        if args["<method>"] not in kernels:
            print("input valid method name!! ")
            exit(1)
    return args

# def cross_validation():


if __name__ == "__main__":
    args = parse_argv()
    x, y = load_x_y(args["<filename>"])
    f = solve(x, y, kernels[args["<method>"]]())
    if args["--plot"]:
        """
        def plot(index):
            plt.cla()
            if index < 1:
                return
            n = len(y)
            xi, yi = [], []
            for i in random.sample(range(n), min(index, n)):
                xi.append(x[i])
                yi.append(y[i])
            plot_f(f, xi, yi, plot_type3D="")
            print(index)
        ani = animation.FuncAnimation(plt.figure(), plot, interval=1000)
        # ani.save("output.gif", writer="imagemagick")
        plt.show()
        """

        plot_f(f, x, y, plot_type3D="")
        plt.show()
        plt.savefig("plotdata.png")
