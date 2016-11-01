#include "svr.h"
#include "quadProg/QuadProg++.hh"
using namespace std;

double SVR::kernel_dot_to_w(const vector<double> &x) const {
  double sum = 0;
  for (auto co : this->oks) {
    sum += co.b * kernel.kernel(co.x, x);
  }
  return sum;
}

SVR::SVR(const vector<vector<double>> &x, const vector<double> &y,
         Kernel kernel, double C, double eps)
    : C(C), eps(eps), SVM(kernel) {
  solve(x, y);
}

void SVR::solve(const vector<vector<double>> &x, const vector<double> &y) {
  const auto r = y.size();
  assert(r == x.size());
  Matrix<double> P(r * 2, r * 2), G(r * 2, r * 4), A(r * 2, 1);
  Vector<double> q(r * 2), h(r * 4), b(1), d(r * 2);
  //全てゼロ初期化 ( *= 0; では微妙にダメっぽい)
  REP(i, r * 2) REP(j, r * 2) P[i][j] = 0;
  REP(i, r * 2) REP(j, r * 4) G[i][j] = 0;
  REP(i, r * 2) A[i][0] = 0;
  REP(i, r * 2) q[i] = 0;
  REP(i, r * 4) h[i] = 0;
  REP(i, 1) b[i] = 0;
  REP(i, r * 2) d[i] = 0;
  REP(k, r) REP(l, r) {
    if (k > l) {
      P[k][l] = P[l][k];
    } else {
      P[k][l] = kernel.kernel(x[k], x[l]);
    }
  }
  REP(i, r * 2) P[i][i] += 1e-9;
  REP(i, r) q[i] = -y[i];
  FOR(i, r, r * 2) q[i] = eps;
  REP(i, r) {
    auto i2 = i + r;
    G[i][i] = 1;
    G[i2][i] = 1;
    G[i][i + r] = -1;
    G[i2][i + r] = 1;
    G[i][i + 2 * r] = -1;
    G[i2][i + 2 * r] = -1;
    G[i][i + 3 * r] = 1;
    G[i2][i + 3 * r] = -1;
  }
  FOR(i, 2 * r, 4 * r) h[i] = 2 * C;
  REP(i, r) A[i][0] = 1;
  b[0] = 0;
  try {
    solve_quadprog(P, q, A, b, G, h, d);
  } catch (exception e) {
    // cout << "invalid matrix" << endl;
    return;
  }
  this->oks.clear();
  this->theta = 0;
  double EPS = eps;  // MEMO 内部浮動小数点処理用eps (epsに依存すべき？)
  double ZERO_EPS = 0.0;
  vector<double> thetas;
  Vector<double> alpha(r), alphastar(r);
  REP(i, r) alpha[i] = (d[i] + d[i + r]) / 2;
  REP(i, r) alphastar[i] = (-d[i] + d[i + r]) / 2;
  REP(i, r) alpha[i] = alpha[i] > EPS ? alpha[i] : 0;
  REP(i, r) alphastar[i] = alphastar[i] > EPS ? alphastar[i] : 0;
  // calc theta
  REP(i, r) {
    if (alpha[i] > ZERO_EPS or alphastar[i] > ZERO_EPS) {
      this->oks.push_back(Ok_b_x({d[i], x[i]}));
      double tmptheta = -y[i];
      if (ZERO_EPS < alpha[i] and alpha[i] < C - EPS) {
        tmptheta += eps;
        REP(j, r) {
          if (alpha[i] > ZERO_EPS or alphastar[i] > ZERO_EPS)
            tmptheta += d[j] * kernel.kernel(x[i], x[j]);
        }
      } else if (ZERO_EPS < alphastar[i] and alphastar[i] < C - EPS) {
        tmptheta -= eps;
        REP(j, r) {
          if (alpha[i] > ZERO_EPS or alphastar[i] > ZERO_EPS) {
            tmptheta += d[j] * kernel.kernel(x[i], x[j]);
          }
        }
      } else {
        continue;
      }
      thetas.push_back(tmptheta);
    }
  }
  if (thetas.size() == 0) {
    // cout << "No Theta..." << endl;
    // cout << alpha.size() << endl;
    return;
  }
  if (this->oks.size() == 0) {
    cout << "invalid data !!!" << endl;
    return;
  }
  ALL(sort, thetas);
  this->theta = thetas[thetas.size() / 2];  // 中央値をとる
  // cout << "ø : " << theta << endl;
  // print_vector("a", alpha);
  // print_vector("a*", alphastar);
  this->is_valid = true;
}

double SVR::func(const vector<double> &x) const {
  if (not this->is_valid) return 1e20;
  return kernel_dot_to_w(x) - theta;
}

void SVR::test(const vector<vector<double>> &x, const vector<double> &y) const {
  auto sum = 0;
  REP(i, y.size()) {
    auto diff = abs(func(x[i]) - y[i]);
    sum += diff < eps * 1.1 ? 1 : 0;
  }
  cout << sum << " / " << y.size() << endl;
}

double SVR::calc_diff(const vector<vector<double>> &test_x,
                      const vector<double> &test_y, const SVR &svr,
                      const cross_validation_type &cvtype, const double eps) {
  double diff = 0.0;
  switch (cvtype) {
    case correct_num:
      REP(j, test_x.size()) {
        diff += abs(svr.func(test_x[j]) - test_y[j]) < 0.1 ? 0 : 1;
      }
      break;
    case mean_abs:
      REP(j, test_x.size()) { diff += abs(svr.func(test_x[j]) - test_y[j]); }
      break;
    case mean_square:
      REP(j, test_x.size()) {
        double n_diff = svr.func(test_x[j]) - test_y[j];
        diff += n_diff * n_diff;
      }
      break;
    case coefficient:
      double d1 = 0.0, d2 = 0.0;
      double mean_test_y = 0.0;
      REP(j, test_x.size()) {
        double n_diff = svr.func(test_x[j]) - test_y[j];
        d1 += n_diff * n_diff;
      }
      REP(j, test_x.size()) mean_test_y += test_y[j];
      mean_test_y /= test_y.size();
      REP(j, test_x.size()) {
        double n_diff = mean_test_y - test_y[j];
        d2 += n_diff * n_diff;
      }
      diff += (1 - d1 / d2);
      break;
  }
  return diff;
}

double SVR::cross_validation(const vector<vector<double>> &x,
                             const vector<double> &y, const Kernel kernel,
                             const double C, const double eps,
                             const cross_validation_type cvtype,
                             const int div) {
  auto n = y.size();
  assert(n == x.size() and n >= div);
  double diff_res = 0.0;
  int successed_sum = 0;
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
    SVR svr(train_x, train_y, kernel, C, eps);
    if (not svr.get_is_valid()) continue;
    diff_res += calc_diff(test_x, test_y, svr, cvtype, eps);
    successed_sum++;
  }
  if (successed_sum == 0) return 1e20;
  return diff_res / n * div / successed_sum;
}
FindPos SVR::search_parameter(const vector<vector<double>> &x,
                              const vector<double> &y, const Kernel::kind kind,
                              const double eps,
                              const cross_validation_type cvtype,
                              int cross_validate_div) {
  auto find_deep = [&](const FindPos basepos, const int div = 10) {
    auto get_center = [div](double center, double offset, int index) {
      return center + offset * (-1.0 + 2.0 * ((double)index / div));
    };
    // TODO: 並列化高速化
    vector<thread> threads;
    vector<vector<FindPos>> finds(div + 1);
    REP(c, finds.size()) {
      finds[c] = vector<FindPos>(div + 1);
      threads.push_back(thread([&, c]() {
        REP(p, finds[c].size()) {
          FindPos pos;
          pos.c_center = get_center(basepos.c_center, basepos.c_offset, c);
          pos.p_center = get_center(basepos.p_center, basepos.p_offset, p);
          Kernel kernel({kind, {pow(2.0, pos.p_center)}});
          pos.error =
              SVR::cross_validation(x, y, kernel, pow(2.0, pos.c_center), eps,
                                    cvtype, cross_validate_div);
          finds[c][p] = pos;
        }
      }));
    }
    for (auto &th : threads) th.join();
    FindPos respos = basepos;
    REP(c, finds.size()) {
      REP(p, finds[c].size()) {
        if (finds[c][p].error < respos.error) {
          respos = finds[c][p];
        }
      }
    }
    respos.c_offset = basepos.c_offset / 10;
    respos.p_offset = basepos.p_offset / 10;
    return respos;
  };
  if (cross_validate_div == 0) cross_validate_div = 10;
  assert(cross_validate_div < x.size());
  auto range = Kernel::get_default_range(kind);
  FindPos nowpos = {5, 10, range.center, range.offset, 1e20};
  REP(i, 3) {  // 実は3回くらいでいいのでは
    FindPos found = find_deep(nowpos, 10 * sqrt(2.0 - (double)i / 3.0));
    cout << "FOUND :";
    found.print();
    if (abs(found.error) >= abs(nowpos.error)) break;
    nowpos = found;
  }
  return nowpos;
}
