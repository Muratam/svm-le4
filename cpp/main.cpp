#include <cassert>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "quadProg/QuadProg++.hh"
#define REP(i, n) for (int i = 0; i < (n); ++i)
using namespace std;

// readline /  python -> c++ -> python

struct Ok_ay_x {
  const double ay;
  const vector<double> x;
};

class Kernel {
 public:
  enum kernel_type { linear, polynomial, gauss };
  const kernel_type m_kernel_type;
  const vector<double> param;
  Kernel(kernel_type k_type, vector<double> param = {})
      : m_kernel_type(k_type), param(param) {}
  double kernel(const vector<double> &x, const vector<double> &y) const {
    switch (m_kernel_type) {
      case linear:
        return inner_product(x.begin(), x.end(), y.begin(), 0.0);
      case polynomial:
        return pow(1 + inner_product(x.begin(), x.end(), y.begin(), 0.0),
                   param[0]);
      case gauss:
        double norm = 0;
        REP(i, x.size()) norm += (x[i] - y[i]) * (x[i] - y[i]);
        return exp(-0.5 * norm / param[0] / param[0]);
    }
  }
  struct kernel_search_range {
    double center, offset;
  };
  static kernel_search_range get_default_range(kernel_type k_type) {
    switch (k_type) {
      case linear:
        assert(false);
        break;
      case polynomial:
        return kernel_search_range({2.5, 2});
      case gauss:
        return kernel_search_range({-6, 7});
    }
  }
};

class SVM {
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
        if (a[i] > max_val) {
          max_val = a[i];
          max_index = i;
        }
      } else {
        a[i] = 0.0;
      }
    }
    this->theta = kernel_dot_to_w(x[max_index]) - y[max_index];
  }
  double func(const vector<double> &x) const {
    return kernel_dot_to_w(x) - theta > 0 ? 1.0 : -1.0;
  }
  double test(const vector<vector<double>> &x, const vector<double> &y) const {
    auto sum = 0;
    REP(i, y.size()) sum += (func(x[i]) == y[i] ? 1 : 0);
    return (double)sum / y.size();
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
                                 const vector<double> &y, Kernel kernel,
                                 int div = 10) {
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
                                         Kernel::kernel_type kernel_type,
                                         int cross_validate_div = 10) {
    auto find_deep = [&](Kernel::kernel_search_range range, int div = 10) {
      // (2 ** (center - offset)) ~ (2 ** (center + offset )) を探す
      vector<Center_Percent> cps(div + 1);
      vector<thread> threads;
      mutex cout_guard;
      REP(i, div + 1) {
        threads.push_back(thread([&, i]() {
          auto center =
              range.center + range.offset * (-1.0 + 2.0 * ((double)i / div));
          Kernel kernel({kernel_type, {pow(2.0, center)}});
          auto percent = cross_validation(x, y, kernel, cross_validate_div);
          cps[i] = Center_Percent({center, percent});
          lock_guard<mutex> lk(cout_guard);
          cout << "2 ** " << center << " : " << percent * 100.0 << "%" << endl;
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
    auto range = Kernel::get_default_range(kernel_type);
    auto found = find_deep(range);
    while (true) {
      range.offset /= 10.0;
      range.center = found.center;
      auto pro_found = find_deep(range);
      if (abs(found.percent - found.percent) <= 0.0001) break;
      found = pro_found;
    }
    return found;
  }
};

int main(int argc, char *const argv[]) {
  vector<vector<double>> x;
  vector<double> y;
  while (true) {
    double x1, x2, y1;
    cin >> x1 >> x2 >> y1;
    if (x1 == x2 and x2 == y1 and y1 == 0) break;
    x.push_back(vector<double>({x1, x2}));
    y.push_back(y1);
  }
  SVM::normalize(x);
  SVM::search_parameter(x, y, Kernel::polynomial);
  return 0;
}
