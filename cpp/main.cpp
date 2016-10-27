#include "svm.h"

class SVR : public SVM {
 protected:
  const double eps;
  const double C;

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
    // this->oks.clear();
    solve_quadprog(P, q, A, b, G, h, d);
    Vector<double> alpha(r), alphastar(r);
    REP(i, r) alpha[i] = (d[i] + d[i + r]) / 2;
    REP(i, r) alphastar[i] = (-d[i] + d[i + r]) / 2;
    print_vector("a", alpha);
    print_vector("a*", alphastar);
  }
  virtual double func(const vector<double> &x) const override {
    return 1 > 0 ? 1.0 : -1.0;
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
    SVR svr(x, y, Kernel(kernel_kind, {atof(parsed[PARAM].c_str())}), 1000,
            0.1);
    if (parsed.count(PLOT)) {
      svr.plot_data(x, y, parsed[PLOT]);
    } else {
      svr.test(x, y);
    }

    return 0;
  }
}
