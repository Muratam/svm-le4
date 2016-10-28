#include "svm.h"

class SVR : public SVM {
 protected:
  struct Ok_b_x {
    // b[] = a[] - aster[]
    // c[] = a[] + aster[]
    const double b;
    const vector<double> x;
  };
  const double eps;
  const double C;
  bool is_valid = false;
  vector<Ok_b_x> oks;

  virtual double kernel_dot_to_w(const vector<double> &x) const override {
    double sum = 0;
    for (auto co : this->oks) {
      sum += co.b * kernel.kernel(co.x, x);
    }
    return sum;
  }

 public:
  bool get_is_valid() { return is_valid; }
  SVR(const vector<vector<double>> &x, const vector<double> &y, Kernel kernel,
      double C = 1000, double eps = 1e-2)
      : C(C), eps(eps), SVM(kernel) {
    solve(x, y);
  }
  virtual void solve(const vector<vector<double>> &x,
                     const vector<double> &y) override {
    const auto r = y.size();
    assert(r == x.size());
    Matrix<double> P(r * 2, r * 2), G(r * 2, r * 4), A(r * 2, 1);
    Vector<double> q(r * 2), h(r * 4), b(1), d(r * 2);
    //全てゼロ初期化
    P *= 0;
    q *= 0;
    A *= 0;
    b *= 0;
    G *= 0;
    h *= 0;
    REP(k, r) REP(l, r) {
      if (k > l) {
        P[k][l] = P[l][k];
      } else {
        P[k][l] = kernel.kernel(x[k], x[l]);
      }
    }
    REP(i, r * 2) P[i][i] += 1.0e-9;
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
    solve_quadprog(P, q, A, b, G, h, d);

    this->oks.clear();
    this->theta = 0;
    auto EPS = eps;  // MEMO 内部浮動小数点処理用eps (epsに依存すべき？)
    vector<double> thetas;
    Vector<double> alpha(r), alphastar(r);
    REP(i, r) alpha[i] = (d[i] + d[i + r]) / 2;
    REP(i, r) alphastar[i] = (-d[i] + d[i + r]) / 2;
    REP(i, r) alpha[i] = alpha[i] > EPS ? alpha[i] : -0;
    REP(i, r) alphastar[i] = alphastar[i] > EPS ? alphastar[i] : -0;
    // calc theta
    REP(i, r) {
      if (alpha[i] > 0 or alphastar[i] > 0) {
        this->oks.push_back(Ok_b_x({d[i], x[i]}));
        double tmptheta = -y[i];
        if (0 < alpha[i] and alpha[i] < C - EPS) {
          tmptheta += eps;
          REP(j, r) {
            if (alpha[i] > 0 or alphastar[i] > 0)
              tmptheta += d[j] * kernel.kernel(x[i], x[j]);
          }
        } else if (0 < alphastar[i] and alphastar[i] < C - EPS) {
          tmptheta -= eps;
          REP(j, r) {
            if (alpha[i] > 0 or alphastar[i] > 0) {
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
      cout << "no theta!!!" << endl;
      return;
    }
    if (this->oks.size() == 0) {
      cout << "invalid data !!!" << endl;
      return;
    }
    ALL(sort, thetas);
    this->theta = thetas[thetas.size() / 2];  // 中央値をとる
    cout << "ø : " << theta << endl;
    // print_vector("a", alpha);
    // print_vector("a*", alphastar);
    this->is_valid = true;
  }
  virtual double func(const vector<double> &x) const override {
    if (not this->is_valid) return 0;
    return kernel_dot_to_w(x) - theta;
  }
  virtual void test(const vector<vector<double>> &x,
                    const vector<double> &y) const override {
    auto sum = 0;
    REP(i, y.size()) {
      auto diff = abs(func(x[i]) - y[i]);
      sum += diff < eps * 1.1 ? 1 : 0;
    }
    cout << sum << " / " << y.size() << endl;
  }
  enum cross_validation_type { mean_abs, mean_square, coefficient };
  static double cross_validation(const vector<vector<double>> &x,
                                 const vector<double> &y, const Kernel kernel,
                                 const double C, const double eps = 1e-2,
                                 const cross_validation_type cvtype = mean_abs,
                                 const int div = 10) {
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
      auto calc_diff = [](const auto &test_x, const auto &test_y,
                          const auto &svr, const auto &cvtype) {
        double diff = 0.0;
        switch (cvtype) {
          case mean_abs:
            REP(j, test_x.size()) {
              diff += abs(svr.func(test_x[j]) - test_y[j]);
            }
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
      };
      diff_res += calc_diff(test_x, test_y, svr, cvtype);
      successed_sum++;
    }
    if (successed_sum == 0) return 1e20;
    return diff_res / n * div / successed_sum;
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

  if (parsed.count(CROSS)) {  // 交差検定dd
  } else {  // 普通にパラメータを指定して(プロット/テストする)
    auto kernel = Kernel(kernel_kind, {atof(parsed[PARAM].c_str())});
    SVR svr(x, y, kernel, 1000, 0.01);
    if (parsed.count(PLOT)) {
      svr.plot_data(x, y, parsed[PLOT]);
    } else {
      svr.test(x, y);
      cout << SVR::cross_validation(x, y, kernel, 1000, 0.01, SVR::mean_square,
                                    5)
           << "\n";
    }

    return 0;
  }
}
