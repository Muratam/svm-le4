#pragma once
#include "kernel.h"
#include "plotable.h"
#include "util.h"

struct Center_Percent {
  double center, percent;
};
struct Ok_ay_x {
  const double ay;
  const vector<double> x;
};
class SVM : public Plotable {
 protected:
  vector<Ok_ay_x> oks;
  double theta;
  const Kernel kernel;
  double kernel_dot_to_w(const vector<double> &x) const;

 public:
  SVM(Kernel kernel);
  SVM(const vector<vector<double>> &x, const vector<double> &y, Kernel kernel);
  void solve(const vector<vector<double>> &x, const vector<double> &y);
  virtual double func(const vector<double> &x) const override;
  void test(const vector<vector<double>> &x, const vector<double> &y) const;

 public:
  static double cross_validation(const vector<vector<double>> &x,
                                 const vector<double> &y, const Kernel kernel,
                                 const int div = 10);
  static Center_Percent search_parameter(
      const vector<vector<double>> &x, const vector<double> &y,
      const Kernel::kind kind, function<void(Center_Percent)> when_created_svm,
      int cross_validate_div = 10);
};
