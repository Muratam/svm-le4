import svmcore
from pprint import pprint
from docopt import docopt
import subprocess

__doc__ = """{f}
Usage:
    {f} <filename> [-m | --method <method>] [--cross <divide_num>] [--plot] [-p | --param <param>] [--cpp <executale>] [--show]
    {f} (-h | --help)
Options:
    --cross     do cross validation
    -p --param     assign parameter (ex: gauss kernel sigma)
    -m --method    {methods} (default:gauss)
    --cpp          execute in cpp (bia shell)
    --show         show progress (ex: crossvalidation parameter)
    --plot         show plotted graph (with matplotlib)
    -h --help      show this help.
""".format(f=__file__, methods=str(",".join(svmcore.kernels.keys())))


def parse_argv():
    args = docopt(__doc__)
    if not args["<method>"]:
        args["<method>"] = "gauss"
    else:
        if args["<method>"] not in svmcore.kernels:
            print("input valid method name!! ")
            exit(1)
    return args


def call_cpp(cpp_args):
    cpp_process = subprocess.Popen(cpp_args, stdout=subprocess.PIPE)
    res = []
    while True:
        line = cpp_process.stdout.readline()
        if line:
            res.append(line.decode()[:-1])
        if not line and cpp_process.poll() is not None:
            break
        #print(line.decode(), end="")
    return res

if __name__ == "__main__":
    args = parse_argv()
    div = int(args["<divide_num>"]) if args["<divide_num>"] else 10
    if args["--cpp"]:
        if not args["--cross"]:
            print("C++ supports is now only cross validation")
        cpp_res = call_cpp(
            [args["<executale>"], args["<filename>"],
             "-c", str(div), "", args["<method>"]])
        p, found = cpp_res
        print("p : {} | {}%".format(p, found))
    else:
        param_ranges, kernel = svmcore.kernels[args["<method>"]]
        x, y = svmcore.load_npx_npy(args["<filename>"])
        x = (x - x.min(0)) / (x.max(0) - x.min(0))  # normalize
        if args["--cross"]:
            p, found = svmcore.search_parameter(
                x, y, kernel, param_ranges, div,
                do_plot=args["--plot"], show=args["--show"])
            print("p : {} | {}%".format(p, found * 100))
        else:
            if args["<param>"]:
                f = svmcore.solve(x, y, kernel([float(args["<param>"])]))
            else:
                f = svmcore.solve(x, y, kernel(), True)
            if args["--plot"]:  # plot は二次元データのみ
                plot_f(f, x, y, plot_type3d="")
                plt.savefig("image/plotdata.png")
                plt.show()
