#include "svr.h"
#include "util.h"
using namespace std;

int print_usage(char *const thisName) {
  cout << "Usage :" << endl
       << "    " << thisName
       << " <dataname> [gauss| polynomial | linear] [--cross <div_num>]\n"
          "            [--param <param>] [--plot <filename>.dat]\n"
          "            [--C <param_C>] [--eps <param_eps>]"
       << "Options :\n"
       << "    --cross      crossvalidation :: div_num\n"
       << "    --plot       plot function :: file name\n"
       << "    --param      defined parameter :: parameter\n"
       << "    --C          the parameter C\n"
       << "    --eps        the parameter eps\n"
       << endl;
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
                                  {PARAM, "10"},
                                  {PARAM_C, "1000"},
                                  {EPS, "0.01"}});

  const auto kernel_kind = Kernel::strings2kernel_kind(args);
  vector<vector<double>> x;
  vector<double> y;
  Kernel::read_data(argv[1], x, y);
  Kernel::normalize(x);
  // TODO: C,epsのコマンドライン
  double eps = atof(parsed[EPS].c_str());
  if (parsed.count(CROSS)) {  // 交差検定
    auto pos = SVR::search_parameter(x, y, kernel_kind, eps, SVR::mean_square,
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
    double C = atof(parsed[PARAM_C].c_str());
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
