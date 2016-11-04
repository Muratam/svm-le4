#include "svr.h"
#include "util.h"
using namespace std;

int print_usage(char *const thisName) {
  cout << "Usage :\n"
       << "  " << thisName
       << " <dataname> [gauss (default)| polynomial | linear] "
          "[--plot <filename>.dat] [--cross <div_num>] "
          "[mean_square (default) | mean_abs | corrent_num | coefficient] "
          "[--p <param>] [--c <param_C>] "
          "[--eps <param_eps>]\n"
       << R"(Options :
  --plot       plot function :: file name
  --cross      crossvalidation :: div_num
  --p          defined parameter :: parameter
  --c          the parameter C  (default 1000)
  --eps        the parameter eps(default 0.01))"
       << endl;
  return -1;
}

int main(int argc, char *const argv[]) {
  if (argc < 2) {
    return print_usage(argv[0]);
  }
  vector<string> args;
  FOR(i, 1, argc) args.push_back(argv[i]);
  const string CROSS = "--cross", PLOT = "--plot", PARAM = "--p",
               PARAM_C = "--c", EPS = "--eps";
  auto parsed = parse_args(args, {{CROSS, "10"},
                                  {PLOT, "result.dat"},
                                  {PARAM, ""},
                                  {PARAM_C, ""},
                                  {EPS, ""}});
  const auto kernel_kind = Kernel::strings2kernel_kind(args);
  const auto cv_type = SVR::get_cross_validation(args);
  vector<vector<double>> x;
  vector<double> y;
  Kernel::read_data(argv[1], x, y);
  Kernel::normalize(x);
  const double eps = parsed.count(EPS) ? atof(parsed[EPS].c_str()) : 1e-2;
  if (parsed.count(CROSS)) {  // 交差検定
    if (parsed.count(PLOT)) {
      SVR::search_parameter(x, y, kernel_kind, eps, cv_type,
                            atoi(parsed[CROSS].c_str()), true);
    } else {
      const auto pos = SVR::search_parameter(
          x, y, kernel_kind, eps, cv_type, atoi(parsed[CROSS].c_str()), false);
      const auto kernel2 = Kernel(kernel_kind, {pow(2.0, pos.p_center)});
      SVR svr(x, y, kernel2, pow(2.0, pos.c_center), eps);
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
      svr.print_func();
      svr.test(x, y);
    }
    return 0;
  }
}
