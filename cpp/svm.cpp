#include "svm.h"
#include "kernel.h"
#include "quadProg/QuadProg++.hh"

using namespace std;

SVM::SVM(Kernel kernel) : kernel(kernel) {}

SVM::SVM(const vector<vector<double>> &x, const vector<double> &y,
         Kernel kernel)
    : kernel(kernel) {
  solve(x, y);
}

double SVM::kernel_dot_to_w(const vector<double> &x) const {
  double sum = 0;
  for (auto co : this->oks) {
    sum += co.ay * kernel.kernel(co.x, x);
  }
  return sum;
}

void SVM::solve(const vector<vector<double>> &x, const vector<double> &y) {
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

double SVM::func(const vector<double> &x) const {
  return kernel_dot_to_w(x) - theta > 0 ? 1.0 : -1.0;
}

void SVM::test(const vector<vector<double>> &x, const vector<double> &y) const {
  auto sum = 0;
  REP(i, y.size()) sum += (func(x[i]) == y[i] ? 1 : 0);
  cout << sum << " / " << y.size() << endl;
}

double SVM::cross_validation(const vector<vector<double>> &x,
                             const vector<double> &y, const Kernel kernel,
                             const int div) {
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
Center_Percent SVM::search_parameter(
    const vector<vector<double>> &x, const vector<double> &y,
    const Kernel::kind kind, function<void(Center_Percent)> when_created_svm,
    int cross_validate_div) {
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
    for (auto &th : threads) th.join();
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
