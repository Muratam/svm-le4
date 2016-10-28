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
  vector<Ok_b_x> oks;

  virtual double kernel_dot_to_w(const vector<double> &x) const override {
    double sum = 0;
    for (auto co : this->oks) {
      sum += co.b * kernel.kernel(co.x, x);
    }
    return sum;
  }

 public:
  SVR(const vector<vector<double>> &x, const vector<double> &y, Kernel kernel,
      double C = 1000, double eps = 1e-9)
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
    assert(thetas.size() >= 1);
    ALL(sort, thetas);
    cout << "theta: \n";
    REP(i, thetas.size()) cout << thetas[i] << " ";
    cout << endl;
    this->theta = thetas[thetas.size() / 2];  // 中央値をとる
    cout << "ø : " << theta << endl;
    print_vector("a", alpha);
    print_vector("a*", alphastar);
    cout << "n : \n";
    REP(i, oks.size()) { cout << oks[i].b << " "; }
    cout << endl;
  }
  virtual double func(const vector<double> &x) const override {
    return kernel_dot_to_w(x) - theta;
  }
  virtual void test(const vector<vector<double>> &x,
                    const vector<double> &y) const override {
    cout << "diff :\n";
    auto sum = 0;
    REP(i, y.size()) {
      auto diff = abs(func(x[i]) - y[i]);
      cout << diff << " ";
      sum += diff < eps * 1.1 ? 1 : 0;
    }
    cout << "\n" << sum << " / " << y.size() << endl;
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
  // Kernel::normalize(x);

  if (parsed.count(CROSS)) {  // 交差検定dd
  } else {  // 普通にパラメータを指定して(プロット/テストする)
    SVR svr(x, y, Kernel(kernel_kind, {atof(parsed[PARAM].c_str())}), 1000,
            0.01);
    if (parsed.count(PLOT)) {
      svr.plot_data(x, y, parsed[PLOT]);
    } else {
      svr.test(x, y);
    }

    return 0;
  }
}
