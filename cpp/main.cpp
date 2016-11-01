#include "svr.h"
using namespace std;

auto parse_args(vector<string> args, vector<pair<string, string>> kvs) {
  unordered_map<string, string> res;
  for (auto kv : kvs) {
    auto seekedarg = ALL(find, args, get<0>(kv));
    if (seekedarg != args.end()) {
      res[get<0>(kv)] =
          seekedarg + 1 == args.end() ? get<1>(kv) : *(seekedarg + 1);
    }
  }
  return res;
}
int print_usage(char *const thisName) {
  cout << "Usage :" << endl
       << "    " << thisName
       << " <dataname> [gauss| polynomial | linear] [--cross <div_num>]\n    "
          "        [--param <param>] [--plot <filename>.dat]\n"
       << "Options :\n"
       //<< "    --cross      crossvalidation :: div_num\n"
       << "    --plot       plot function :: file name\n"
       // << "    --plot-all   plot all of progress :: direcoty name\n"
       << "    --param      defined parameter :: parameter\n"
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
               PLOT_ALL = "--plot-all";
  auto parsed = parse_args(
      args,
      {{CROSS, "10"}, {PLOT, "result.dat"}, {PARAM, "10"}, {PLOT_ALL, "data"}});

  const auto kernel_kind = Kernel::strings2kernel_kind(args);
  vector<vector<double>> x;
  vector<double> y;
  Kernel::read_data(argv[1], x, y);
  Kernel::normalize(x);
  // TODO: C,epsのコマンドライン
  auto eps = 1e-2;
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
    auto C = 1000;
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
