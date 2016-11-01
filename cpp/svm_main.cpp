#include "svm.h"
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
          "        [--param <param>] [--plot <filename>.dat] [--plot-all]\n"
       << "Options :\n"
       << "    --cross      crossvalidation :: div_num\n"
       << "    --plot       plot function :: file name\n"
       << "    --plot-all   plot all of progress :: direcoty name\n"
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

  if (parsed.count(CROSS)) {  // 交差検定

    const auto cp = SVM::search_parameter(
        x, y, kernel_kind,
        [&parsed, PLOT_ALL, &x, &y, &kernel_kind](auto cp) {
          if (!parsed.count(PLOT_ALL)) return;
          SVM svm(x, y, Kernel(kernel_kind, {pow(2, cp.center)}));
          auto filename = parsed[PLOT_ALL] + "/" +
                          to_string(pow(2, cp.center)) + "_" +
                          to_string((int)(cp.percent * 100)) + "per.dat";
          svm.plot_data(x, y, filename);
        },
        atof(parsed[CROSS].c_str()));
    if (parsed.count(PLOT)) {  // 結果をプロット
      SVM svm(x, y, Kernel(kernel_kind, {pow(2, cp.center)}));
      svm.plot_data(x, y, parsed[PLOT]);
    }
    cout << "RESULT :: " << pow(2, cp.center) << " | " << 100 * cp.percent
         << "% \n";

  } else {  // 普通にパラメータを指定して(プロット/テストする)

    SVM svm(x, y, Kernel(kernel_kind, {atof(parsed[PARAM].c_str())}));
    if (parsed.count(PLOT)) {
      svm.plot_data(x, y, parsed[PLOT]);
    } else {
      svm.test(x, y);
    }
  }

  return 0;
}
