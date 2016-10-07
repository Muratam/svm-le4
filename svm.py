import numpy as np
import matplotlib.pyplot as plt
import cvxopt, time, matplotlib, sys
import sympy

kernels = {
    "dot" : lambda : lambda x,y: x.dot(y),
    "polynomial" : lambda d=2 : lambda x,y: (1 + x.dot(y)) ** d,
    "sigmoid" : lambda a=2,b=2: lambda x,y: np.tanh(a * x.dot(y) + b),
    "gauss" : lambda rho=10 :\
        (lambda x,y: np.exp(-0.5 * np.square(np.linalg.norm(x-y) / rho))),
}





def solve(x,y,kernel):
    x = np.array(x)
    y = np.array(y)
    n = len(y)
    assert(n == len(x))
    P = cvxopt.matrix([[0.0]*n]*n)
    for k in range(n):
        for l in range(n):
            P[k,l] = y[k] * y[l] * kernel(x[k],x[l]) if k >= l else P[l,k]
    q = cvxopt.matrix([-1.0] * n)
    G = cvxopt.spdiag([-1.0] * n)
    h = cvxopt.matrix([0.0] * n)
    A = cvxopt.matrix([[0.0]*1]*n)
    for k in range(n): A[0,k] = y[k]
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
    f = lambda n_x : 1.0 if kernel_dot_to_w(n_x) - theta > 0 else -1.0
    print("passed :" + str(sum([1 for i in range(n) if f(x[i]) == y[i]])) + " / " + str(len(y)))


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

def load_x_y(fileName):
    x,y = [],[]
    with open(fileName,"r") as f:
        for line in f.readlines():
            xk = line.split(" ")
            x.append([float(x) for x in xk[0:-1]])
            y.append(float(xk[-1]))
    return x,y

if __name__ == "__main__" :
    if "-h" in sys.argv or "--help" in sys.argv :
        print("#### support vector machine ####")
        print("python3 svm.py <filename> <method> --plot")
        print("method : gauss(default), polynomial, sigmoid, dot")
        #print("  method arg is optional ,the default is ...")
        #print("    gauss :: rho = 10")
        #print("    polynomial :: d = 2")
        #print("    sigmoid :: a = 2,b = 2")
        print("--plot : show plot graph (with matplotlib)")
        exit(0)
    shold_plot = "--plot" in sys.argv
    args = [x for x in sys.argv[1:] if x != "--plot"]
    if len(args) == 0 :
        print("please input filename !!")
        exit(1)
    filename = args[0]
    if len(args) == 1 : method = "gauss"
    else :
        method = args[1]
        if method not in kernels :
            print("method must be gauss, polynomial, sigmoid, or dot")
            exit(1)
    x,y = load_x_y(filename)
    if shold_plot : plot(x,y)
    solve(x,y,kernels[method]())

