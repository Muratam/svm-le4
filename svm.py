import numpy as np
import matplotlib.pyplot as plt
import cvxopt, time,matplotlib

def sign(x): return 1.0 if x >= 0 else -1.0

def calc_P(x,y,kernel):
    n = len(y)
    P = cvxopt.matrix([[0.0]*n]*n)
    for k in range(n):
        for l in range(n):
            P[k,l] = y[k] * y[l] * kernel(x[k],x[l]) if k >= l else P[l,k]
    return P

def calc_A(y):
    n = len(y)
    A = cvxopt.matrix([[0.0]*1]*n)
    for k in range(n): A[0,k] = y[k]
    return A

def solve(x,y,kernel):
    x = np.array(x)
    y = np.array(y)
    n = len(y)
    assert(n == len(x))
    P = calc_P(x,y,kernel)
    q = cvxopt.matrix([-1.0] * n)
    G = cvxopt.spdiag([-1.0] * n)
    h = cvxopt.matrix([0.0] * n)
    A = calc_A(y)
    b = cvxopt.matrix(0.0)
    sol = cvxopt.solvers.qp(P,q,G,h,A,b,options = {"show_progress" : False})
    a = np.array([(x if x > 0.0000001 else 0 ) for x in sol["x"]])
    w = np.array(sum([a[i] * y[i] * x[i] for i in range(n)]))
    max_index = a.argmax()
    def kernel_dot_to_w(n_x):
        return sum([a[i] * y[i] * kernel(x[i],n_x) for i in range(n)])
    theta = kernel_dot_to_w(x[max_index]) - y[max_index]
    print("α : " + str(a))
    print("max_index : " + str(max_index))
    print("max_num : " + str(a[max_index]))
    print("w : " + str(w))
    print("θ : " + str(theta))
    f = lambda n_x : sign(kernel_dot_to_w(n_x) - theta)
    print("passed :" + str(sum([1 for i in range(n) if f(x[i]) == y[i]])) + " / " + str(len(y)))

def kernel_dot():
    return (lambda x,y : x.dot(y))
def kernel_polynomial(d = 2):
    return (lambda x,y : (1 + x.dot(y)) ** d)
def kernel_gauss(rho = 10):
    return (lambda x,y : np.exp(-0.5 * np.square(np.linalg.norm(x-y) / rho)))
def kernel_sigmoid(a = 2,b = 2):
    return (lambda x,y : np.tanh(a * x.dot(y) + b))

def plot(x,y):
    x1,y1,x2,y2 = [],[],[],[]
    for i in range(len(y)):
        (x1 if y[i] > 0 else x2).append(x[i][0])
        (y1 if y[i] > 0 else y2).append(x[i][1])
    plt.scatter(x1,y1, c='red', s=30, marker='x', label='+1')
    plt.scatter(x2,y2, c='blue',s=30, marker='x', label='-1')
    #plt.title('plot')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig("test.png")


def main(fileName):
    x,y = [],[]
    with open(fileName,"r") as f:
        for line in f.readlines():
            xk = line.split(" ")
            x.append([float(x) for x in xk[0:-1]])
            y.append(float(xk[-1]))
    #x = [[0,1],[0,2],[0,5],[0,7],[0.0,8.0]]
    #y = [1.0,1,-1,-1,1]
    #plot(x,y)
    solve(x,y,kernel_gauss())
    #solve(x,y,kernel_gauss())
if __name__ == "__main__" :
    main("sample_data/sample_circle.dat")