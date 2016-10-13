import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import time
import matplotlib
import sys
import sympy
# http://www.ai.soc.i.kyoto-u.ac.jp/~matsubara/le4-2016/index.php?%28Step1%29%20SVM%E3%81%AE%E4%BD%9C%E6%88%90

kernels = {
    "dot": lambda: lambda x, y: x.dot(y),
    "polynomial": lambda d=2: lambda x, y: (1 + x.dot(y)) ** d,
    "sigmoid": lambda a=3, b=4: lambda x, y: np.tanh(a * x.dot(y) + b),
    "gauss": lambda sigma=10:
        (lambda x, y: np.exp(-0.5 * np.square(np.linalg.norm(x - y) / sigma))),
}

# レポート作成

"""
def plot3D():#a,y,x):
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    X, Y = np.mgrid[-100:100:2, -100:100:2]
    Z = X + Y
    Axes3D(plt.figure()).plot_wireframe(X, Y, Z)
    plt.show()
plot3D()
"""


def solve(x, y, kernel):
    x = np.array(x)
    y = np.array(y)
    n = len(y)
    assert(n == len(x))
    P = cvxopt.matrix([[0.0] * n] * n)
    for k in range(n):
        for l in range(n):
            P[k, l] = y[k] * y[l] * kernel(x[k], x[l]) if k >= l else P[l, k]
    q = cvxopt.matrix([-1.0] * n)
    G = cvxopt.spdiag([-1.0] * n)
    h = cvxopt.matrix([0.0] * n)
    A = cvxopt.matrix([[0.0] * 1] * n)
    for k in range(n):
        A[0, k] = y[k]
    b = cvxopt.matrix(0.0)
    sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={"show_progress": False})
    a = np.array([(x if x > 0.0000001 else 0) for x in sol["x"]])
    ok_indexes = [i for i in range(n) if abs(a[i]) > 0]
    ok_coes = [(a[i] * y[i], x[i]) for i in ok_indexes]

    def kernel_dot_to_w(n_x):
        return sum([co[0] * kernel(co[1], n_x) for co in ok_coes])
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


def plot_core(x1s, x2s, cs, ss, markers, labels):
    for i in range(len(x1s)):
        plt.scatter(x1s[i], x2s[i], c=cs[i], s=ss[i],
                    marker=markers[i], label=labels[i])
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig("plotdata.png")


def separate(xs, ys):
    n = len(ys)
    assert(len(xs) == n)
    x1p = [xs[i][0] for i in range(n) if ys[i] > 0]
    x2p = [xs[i][1] for i in range(n) if ys[i] > 0]
    x1m = [xs[i][0] for i in range(n) if ys[i] < 0]
    x2m = [xs[i][1] for i in range(n) if ys[i] < 0]
    return x1p, x2p, x1m, x2m


def plot_f(f, x, y, num=100):
    x1 = np.linspace(min([_[0] for _ in x]), max([_[0] for _ in x]), num)
    x2 = np.linspace(min([_[1] for _ in x]), max([_[1] for _ in x]), num)
    x1mesh, x2mesh = np.meshgrid(x1, x2)
    xgs = np.c_[x1mesh.ravel(), x2mesh.ravel()]
    ygs = [f(x) for x in xgs]
    x1p, x2p, x1m, x2m = separate(x, y)
    x1pg, x2pg, x1mg, x2mg = separate(xgs, ygs)
    x1s = [x1p, x1m, x1pg, x1mg]
    x2s = [x2p, x2m, x2pg, x2mg]
    cs = ["red", "blue", "red", "blue"]
    ss = [30, 30, 10, 10]
    markers = ["o", "o", "x", "x"]
    labels = ["+1", "-1", "+1", "-1"]
    plot_core(x1s, x2s, cs, ss, markers, labels)


def load_x_y(fileName):
    x, y = [], []
    with open(fileName, "r") as f:
        for line in f.readlines():
            xk = line.split(" ")
            x.append([float(x) for x in xk[0:-1]])
            y.append(float(xk[-1]))
    return x, y

if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print("#### support vector machine ####")
        print("python3 svm.py <filename> <method> --plot")
        print("method : gauss(default), polynomial, sigmoid, dot")
        print("--plot : show plot graph (with matplotlib)")
        exit(0)
    shold_plot = "--plot" in sys.argv
    args = [x for x in sys.argv[1:] if x != "--plot"]
    if len(args) == 0:
        print("please input filename !!")
        exit(1)
    filename = args[0]
    if len(args) == 1:
        method = "gauss"
    else:
        method = args[1]
        if method not in kernels:
            print("method must be gauss, polynomial, sigmoid, or dot")
            exit(1)
    x, y = load_x_y(filename)
    f = solve(x, y, kernels[method]())
    if shold_plot:
        plot_f(f, x, y)
