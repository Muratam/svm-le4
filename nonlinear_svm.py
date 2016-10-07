import numpy as np
from scipy import optimize

def opti_fun(a):
    sum = 0
    for k in range(len(y)):
        for l in range(len(y)):
            sum += a[k] * a[l] * y[k] * y[l] * X[k].dot(X[l])
    return -1 * (a.sum() - 0.5 * sum)


def solve(X,y):
    constraints = [{
        "type": "ineq",
        "fun": lambda a: a
    }, {
        "type": "eq",
        "fun": lambda a: a.dot(y)
    }]
    x0 = np.random.randn(len(y))  # 標準正規分布乱数 : 解の探索の始点
    res = optimize.minimize(
            opti_fun, x0,
            constraints=constraints,
            method="SLSQP")
    print(res)
    a = np.array( res["x"] )
    print("a:")
    print(a)
    w = 0
    for k in range(len(y)):
        w += a[k] * y[k] * X[k]
    print("w")
    print(w)

with open("sample_linear.dat","r") as f:
    for line in f.readlines():
        xk = line.split(" ")
        yk = x[-1]
        xk = x[0:-1]

y = np.array([1, 1, 1, -1, -1, -1])
X = np.array([[-1, 1], [1, 1], [-0.5, 0], [0.5, -0.5], [-1, -1], [1, -1]])
solve(X,y)

