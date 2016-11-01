#pragma once
#include "kernel.h"
struct Center_Percent {
  double center, percent;
};
struct Ok_ay_x {
  const double ay;
  const vector<double> x;
};
class SVM {
 protected:
  vector<Ok_ay_x> oks;
  double theta;
  const Kernel kernel;
  virtual double kernel_dot_to_w(const vector<double> &x) const;

 public:
  SVM(Kernel kernel);
  SVM(const vector<vector<double>> &x, const vector<double> &y, Kernel kernel);
  virtual void solve(const vector<vector<double>> &x, const vector<double> &y);
  virtual double func(const vector<double> &x) const;
  virtual void test(const vector<vector<double>> &x,
                    const vector<double> &y) const;
  void plot_data(const vector<vector<double>> &x, const vector<double> &y,
                 const string filename, const int plot_grid = 100);

 public:
  static double cross_validation(const vector<vector<double>> &x,
                                 const vector<double> &y, const Kernel kernel,
                                 const int div = 10);
  static Center_Percent search_parameter(
      const vector<vector<double>> &x, const vector<double> &y,
      const Kernel::kind kind, function<void(Center_Percent)> when_created_svm,
      int cross_validate_div = 10);
};
