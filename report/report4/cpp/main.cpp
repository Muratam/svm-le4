#include "svr.h"
#include "util.h"
using namespace std;

int print_usage(char *const thisName) {
  cout << "Usage :\n"
       << "  " << thisName
       << " <dataname> [gauss (default)| polynomial | linear] "
          "[--plot <filename>.dat] [--cross <div_num>] "
          "[mean_square (default) | mean_abs | corrent_num | coefficient] "
          "[--p <param>] [--c <param_C>] [--silent] [--plot-c-p <cp_filename>]"
          "[--eps <param_eps>] [--test <testdata>.dat] [non-normalize]\n"
       << R"(Options :
  --plot         plot function :: file name
  --cross        crossvalidation :: div_num
  --p            defined parameter :: parameter
  --c            the parameter C  (default 1000)
  --eps          the parameter eps(default 0.01)
  --plot-c-p     when cross validation, plot the result
  --test         when created SVR,test data (output => --plot file)
  --silent       do process silently
  non-normalize  then don't normalize x
  )" << endl;
  return -1;
}

int main(int argc, char *const argv[]) {
  if (argc < 2) {
    return print_usage(argv[0]);
  }
  vector<string> args;
  FOR(i, 1, argc) args.push_back(argv[i]);
  const string CROSS = "--cross", PLOT = "--plot", PARAM = "--p",
               SILENT = "--silent", PARAM_C = "--c", EPS = "--eps",
               TEST = "--test", PLOT_C_P = "--plot-c-p";
  auto parsed = parse_args(args, {{CROSS, "10"},
                                  {PLOT, "result.dat"},
                                  {PARAM, ""},
                                  {PARAM_C, ""},
                                  {EPS, ""},
                                  {TEST, "testdata.dat"},
                                  {PLOT_C_P, "c_p.dat"}});
  const auto kernel_kind = Kernel::strings2kernel_kind(args);
  const auto cv_type = SVR::get_cross_validation(args);
  const double eps = parsed.count(EPS) ? atof(parsed[EPS].c_str()) : 1e-2;
  const bool is_silent = ALL(find, args, SILENT) != args.end();
  vector<vector<double>> x;
  vector<double> y;
  Kernel::read_data(argv[1], x, y);
  if (ALL(find, args, "non-normalize") == args.end()) Kernel::normalize(x);
  auto do_svr = [&](SVR &svr) {
    if (parsed.count(TEST)) {
      vector<vector<double>> test_x;
      Kernel::read_x(parsed[TEST], test_x);
      svr.plot_data(test_x, parsed[PLOT], 0, is_silent);
    } else {
      if (parsed.count(PLOT)) {
        svr.plot_data(x, parsed[PLOT], 100, is_silent);
      } else {
        svr.print_func();
        svr.test(x, y);
      }
    }
  };
  if (parsed.count(CROSS)) {  // 交差検定
    const auto pos =
        SVR::search_parameter(x, y, kernel_kind, eps, cv_type,
                              atoi(parsed[CROSS].c_str()), is_silent);
    if (parsed.count(PLOT_C_P)) {
      std::ofstream ofile;
      ofile.open(parsed[PLOT_C_P], std::ios::out);
      ofile << pow(2.0, pos.c_center) << " " << pow(2.0, pos.p_center) << endl;
      ofile.close();
      if (not is_silent) {
        cout << "saved c,p as " << parsed[PLOT_C_P] << endl;
      }
      return 0;
    }
    const auto kernel2 = Kernel(kernel_kind, {pow(2.0, pos.p_center)});
    SVR svr(x, y, kernel2, pow(2.0, pos.c_center), eps);
    if (not is_silent) {
      cout << "RESULT\n";
      pos.print();
    }
    do_svr(svr);
  } else {  // 普通にパラメータを指定して(プロット/テストする)
    double C = parsed.count(PARAM_C) ? atof(parsed[PARAM_C].c_str()) : 1e3;
    auto kernel = Kernel(kernel_kind, {atof(parsed[PARAM].c_str())});
    SVR svr(x, y, kernel, C, eps);
    do_svr(svr);
    return 0;
  }
}
