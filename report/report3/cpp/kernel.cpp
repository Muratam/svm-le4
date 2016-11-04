#include "kernel.h"

using namespace std;
double Kernel::kernel(const vector<double> &x, const vector<double> &y) const {
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
string Kernel::to_string() const {
  switch (this->m_kind) {
    case linear:
      return "linear";
    case polynomial:
      return "polinomial";
    case gauss:
      return "gauss";
  }
}

Kernel::search_range Kernel::get_default_range(const Kernel::kind k_type) {
  switch (k_type) {
    case linear:
      return search_range({1.0, 1.0});
    case polynomial:
      return search_range({3, 2});
    case gauss:
      return search_range({-6, 7});
  }
}
Kernel::kind Kernel::strings2kernel_kind(vector<string> args) {
  unordered_map<string, Kernel::kind> kernels = {
      {"gauss", Kernel::gauss},
      {"polynomial", Kernel::polynomial},
      {"linear", Kernel::linear}};
  for (auto arg : args) {
    if (kernels.count(arg)) return kernels[arg];
  }
  return Kernel::gauss;
}
void Kernel::read_data(string filename, vector<vector<double>> &x,
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
void Kernel::normalize(vector<vector<double>> &x) {
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
