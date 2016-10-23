// clang++ -std=c++14 main.cpp quadProg/QuadProg++.cc
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
  static Center_Percent search_parameter(const vector<vector<double>> &x,
                                         const vector<double> &y,
                                         const Kernel::kind kind,
                                         const bool show_progress = false,
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
          if (show_progress) {
            lock_guard<mutex> lk(cout_guard);
            cout << "2 ** " << center << " : " << percent * 100.0 << "%"
                 << endl;
          }
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

int main(int argc, char *const argv[]) {
  vector<string> args;
  FOR(i, 1, argc) args.push_back(argv[i]);
  assert(argc >= 2);
  auto kernel_kind = Kernel::strings2kernel_kind(args);
  auto should_show = ALL(find, args, "--show") != args.end();
  auto args_cross_validation = ALL(find, args, "-c");
  auto should_cross_validation = args_cross_validation != args.end();
  int cross_validation_div = std::atoi((*(args_cross_validation + 1)).c_str());
  vector<vector<double>> x;
  vector<double> y;
  SVM::read_data(argv[1], x, y);
  SVM::normalize(x);
  if (should_cross_validation) {
    auto cp = SVM::search_parameter(x, y, kernel_kind, should_show,
                                    cross_validation_div);
    if (!should_show)
      cout << pow(2, cp.center) << "\n" << 100 * cp.percent << "\n";
    else
      cout << "RESULT :: " << pow(2, cp.center) << " | " << 100 * cp.percent
           << "% \n";
  } else {
    SVM(x, y, Kernel(kernel_kind, {10})).test(x, y);
  }
  return 0;
}
