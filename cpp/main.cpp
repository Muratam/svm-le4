#include "svr.h"
#include "util.h"
using namespace std;

int print_usage(char *const thisName) {
  cout << "Usage :\n"
       << "  " << thisName
       << " <dataname> [gauss| polynomial | linear] [--cross <div_num>] "
          "[--param <param>] [--plot <filename>.dat] [--C <param_C>] [--eps "
          "<param_eps>]\n"
       << R"(Options :
  --cross      crossvalidation :: div_num
  --plot       plot function :: file name
  --param      defined parameter :: parameter
  --C          the parameter C  (default 1000)
  --eps        the parameter eps(default 0.01))";
  return -1;
}

int main(int argc, char *const argv[]) {
  if (argc < 2) {
    return print_usage(argv[0]);
  }
  vector<string> args;
  FOR(i, 1, argc) args.push_back(argv[i]);
  const string CROSS = "--cross", PLOT = "--plot", PARAM = "--param",
               PARAM_C = "--C", EPS = "--EPS";
  auto parsed = parse_args(args, {{CROSS, "10"},
                                  {PLOT, "result.dat"},
                                  {PARAM, ""},
                                  {PARAM_C, ""},
                                  {EPS, ""}});

  const auto kernel_kind = Kernel::strings2kernel_kind(args);
  vector<vector<double>> x;
  vector<double> y;
  Kernel::read_data(argv[1], x, y);
  Kernel::normalize(x);
  double eps = parsed.count(EPS) ? atof(parsed[EPS].c_str()) : 1e-2;
  if (parsed.count(CROSS)) {  // 交差検定
    auto pos = SVR::search_parameter(x, y, kernel_kind, eps, SVR::coefficient,
                                     atoi(parsed[CROSS].c_str()));
    auto kernel2 = Kernel(kernel_kind, {pow(2.0, pos.p_center)});
    SVR svr(x, y, kernel2, pow(2.0, pos.c_center), eps);
    if (parsed.count(PLOT)) {
      svr.plot_data(x, y, parsed[PLOT]);
    } else {
      cout << "RESULT\n";
      pos.print();
      svr.test(x, y);
    }
  } else {  // 普通にパラメータを指定して(プロット/テストする)
    double C = parsed.count(PARAM_C) ? atof(parsed[PARAM_C].c_str()) : 1e3;
    auto kernel = Kernel(kernel_kind, {atof(parsed[PARAM].c_str())});
    SVR svr(x, y, kernel, C, eps);
    if (parsed.count(PLOT)) {
      svr.plot_data(x, y, parsed[PLOT]);
    } else {
      svr.test(x, y);
    }
    return 0;
  }
}
