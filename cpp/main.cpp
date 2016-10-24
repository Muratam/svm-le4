// clang++ -std=c++14 main.cpp quadProg/QuadProg++.cc
// <<argv[0]>> <dataname> [gauss| polynomial | linear] [--cross 5]
//                        [--param 10] [--save a.dat] [--plot [100]]";

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "quadProg/QuadProg++.hh"
#define REP(i, n) for (int i = 0; i < (n); ++i)
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define ALL(f, x, ...) \
  ([&](auto &ALL) { return (f)(begin(ALL), end(ALL), ##__VA_ARGS__); })(x)
using namespace std;

// 交差検定すらを数式にいれてはどうか
// png-jpgのハイブリッドな前処理を入れてみてはどうか

class Kernel {
 public:
  enum kind { linear, polynomial, gauss };
  const Kernel::kind m_kind;
  const vector<double> param;
  Kernel(const Kernel::kind k_type, const vector<double> param = {})
      : m_kind(k_type), param(param) {}
  double kernel(const vector<double> &x, const vector<double> &y) const {
    switch (m_kind) {
      case linear:
        return ALL(inner_product, x, y.begin(), 0.0);
      case polynomial:
        return pow(1.0 + ALL(inner_product, x, y.begin(), 0.0), param[0]);
      case gauss:
        double norm = 0;
        REP(i, x.size()) norm += (x[i] - y[i]) * (x[i] - y[i]);
        return exp(-0.5 * norm / param[0] / param[0]);
    }
  }
  struct search_range {  // 2 ** (center ± offset)
    double center, offset;
  };
  static search_range get_default_range(const Kernel::kind k_type) {
    switch (k_type) {
      case linear:
        return search_range({1.0, 1.0});
      case polynomial:
        return search_range({2.5, 2});
      case gauss:
        return search_range({-6, 7});
    }
  }
  static Kernel::kind strings2kernel_kind(vector<string> args) {
    unordered_map<string, Kernel::kind> kernels = {
        {"gauss", Kernel::gauss},
        {"polynomial", Kernel::polynomial},
        {"linear", Kernel::linear}};
    for (auto arg : args) {
      if (kernels.count(arg)) return kernels[arg];
    }
    return Kernel::gauss;
  }
};

class SVM {
 protected:
  struct Ok_ay_x {
    const double ay;
    const vector<double> x;
  };

 protected:
  vector<Ok_ay_x> oks;
  double theta;
  const Kernel kernel;
  double kernel_dot_to_w(const vector<double> &x) const {
    double sum = 0;
    for (auto co : this->oks) {
      sum += co.ay * kernel.kernel(co.x, x);
    }
    return sum;
  }

 public:
  SVM(const vector<vector<double>> &x, const vector<double> &y, Kernel kernel)
      : kernel(kernel) {
    const auto dim = 2;
    const auto n = y.size();
    assert(n == x.size());
    Matrix<double> P(n, n), G(n, n), A(n, 1);
    Vector<double> q(n), h(n), b(1), a(n);
    REP(k, n) REP(l, n) {
      if (k > l) {
        P[k][l] = P[l][k];
      } else {
        P[k][l] = y[k] * y[l] * kernel.kernel(x[k], x[l]);
      }
      if (k == l) P[k][l] += 1.0e-9;
    }
    REP(i, n) q[i] = -1;
    REP(i, n) REP(j, n) G[i][j] = i == j ? 1 : 0;
    REP(i, n) h[i] = 0;
    REP(i, n) A[i][0] = y[i];
    b[0] = 0;
    this->oks.clear();
    try {
      solve_quadprog(P, q, A, b, G, h, a);
    } catch (exception e) {
      // cout << "invalid A" << endl;
      return;
    }
    auto max_index = 0, max_val = 0;
    REP(i, n) {
      if (abs(a[i]) > 1e-5) {
        this->oks.push_back(Ok_ay_x({a[i] * y[i], x[i]}));
        if (abs(a[i]) > abs(max_val)) {
          max_val = a[i];
          max_index = i;
        }
      }
    }
    this->theta = kernel_dot_to_w(x[max_index]) - y[max_index];
  }
  double func(const vector<double> &x) const {
    return kernel_dot_to_w(x) - theta > 0 ? 1.0 : -1.0;
  }
  void test(const vector<vector<double>> &x, const vector<double> &y) const {
    auto sum = 0;
    REP(i, y.size()) sum += (func(x[i]) == y[i] ? 1 : 0);
    cout << sum << " / " << y.size() << endl;
  }
  void plot_data(const vector<vector<double>> &x, const vector<double> &y,
                 const string filename, const int plot_grid = 100) {
    if (filename == "") return;
    ofstream ofile;
    ofile.open(filename, std::ios::out);

    bool print_grid = plot_grid > 0;
    if (print_grid) {
      REP(i, plot_grid) {
        REP(j, plot_grid) {
          auto i_p = (double)i / plot_grid, j_p = (double)j / plot_grid;
          ofile << i_p << " " << j_p << " ";
          ofile << func({i_p, j_p}) << endl;
        }
      }
    } else {
      REP(i, x.size()) {
        REP(j, x[0].size()) { ofile << x[i][j] << " "; }
        ofile << func(x[i]) << endl;
      }
    }
    ofile.close();
    cout << "saved as " + filename << endl;
  }

 public:
  static void normalize(vector<vector<double>> &x) {
    //全て[0..1]の範囲にする
    REP(i, x[0].size()) {
      double max_val = -INT_MAX, min_val = INT_MAX;
      REP(j, x.size()) {
        max_val = max(max_val, x[j][i]);
        min_val = min(min_val, x[j][i]);
      }
      REP(j, x.size()) { x[j][i] = (x[j][i] - min_val) / (max_val - min_val); }
    }
  }
  static double cross_validation(const vector<vector<double>> &x,
                                 const vector<double> &y, const Kernel kernel,
                                 const int div = 10) {
    auto n = y.size();
    assert(n == x.size() and n >= div);
    auto passed_sum = 0;
    REP(i, div) {
      vector<vector<double>> train_x, test_x;
      vector<double> train_y, test_y;
      REP(j, n) {
        if (j % div == i) {
          test_x.push_back(x[j]);
          test_y.push_back(y[j]);
        } else {
          train_x.push_back(x[j]);
          train_y.push_back(y[j]);
        }
      }
      SVM svm(train_x, train_y, kernel);
      REP(j, test_x.size()) {
        passed_sum += svm.func(test_x[j]) == test_y[j] ? 1 : 0;
      }
    }
    return (double)passed_sum / n;
  }
  struct Center_Percent {
    double center, percent;
  };
  static Center_Percent search_parameter(
      const vector<vector<double>> &x, const vector<double> &y,
      const Kernel::kind kind, function<void(Center_Percent)> when_created_svm,
      int cross_validate_div = 10) {
    auto find_deep = [&](Kernel::search_range range, int div = 10) {
      vector<Center_Percent> cps(div + 1);
      vector<thread> threads;
      mutex cout_guard;
      REP(i, div + 1) {
        threads.push_back(thread([&, i]() {
          auto center =
              range.center + range.offset * (-1.0 + 2.0 * ((double)i / div));
          Kernel kernel({kind, {pow(2.0, center)}});
          auto percent = cross_validation(x, y, kernel, cross_validate_div);
          cps[i] = Center_Percent({center, percent});
          lock_guard<mutex> lk(cout_guard);
          cout << "2 ** " << center << " : " << percent * 100.0 << "%" << endl;
          when_created_svm(cps[i]);
        }));
      }
      for (auto &th : threads) {
        th.join();
      }
      Center_Percent maxcp({0, 0});
      for (auto &cp : cps) {
        if (cp.percent > maxcp.percent) {
          maxcp.percent = cp.percent;
          maxcp.center = cp.center;
        }
      }
      return maxcp;
    };
    if (cross_validate_div == 0) cross_validate_div = 10;
    assert(cross_validate_div < x.size());
    auto range = Kernel::get_default_range(kind);
    auto found = find_deep(range);
    while (true) {
      range.offset /= 10.0;
      range.center = found.center;
      auto pro_found = find_deep(range);
      if (abs(found.percent - pro_found.percent) <= 0.001) break;
      found = pro_found;
    }
    return found;
  }
  static void read_data(string filename, vector<vector<double>> &x,
                        vector<double> &y) {
    ifstream ifile;
    ifile.open(filename, std::ios::in);
    // x0 x1 .... xn y
    for (string line; getline(ifile, line) and cin.good();) {
      auto n = 0.0;
      auto xi = vector<double>();
      for (istringstream iss(line); iss.good();) {
        iss >> n;
        xi.push_back(n);
      }
      y.push_back(xi[xi.size() - 1]);
      xi.pop_back();
      x.push_back(xi);
    }
    ifile.close();
  }
};
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

int main(int argc, char *const argv[]) {
  vector<string> args;
  FOR(i, 1, argc) args.push_back(argv[i]);
  if (argc < 2) {
    cout << "Usage :" << endl
         << "    " << argv[0]
         << " <dataname> [gauss| polynomial | linear] [--cross <div_num>]\n    "
            "        [--param <param>] [--plot <filename>.dat]\n"
         << "Options :\n"
         << "    --cross      crossvalidation :: div_num\n"
         << "    --plot       plot function :: file name\n"
         << "    --plot-all   plot all of progress :: direcoty name\n"
         << "    --param      defined parameter :: parameter\n"
         << endl;
    return -1;
  }
  const auto kernel_kind = Kernel::strings2kernel_kind(args);
  vector<vector<double>> x;
  vector<double> y;
  SVM::read_data(argv[1], x, y);
  SVM::normalize(x);
  const string CROSS = "--cross", PLOT = "--plot", PARAM = "--param",
               PLOT_ALL = "--plot-all";
  auto parsed = parse_args(
      args,
      {{CROSS, "10"}, {PLOT, "result.dat"}, {PARAM, "10"}, {PLOT_ALL, "data"}});
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
