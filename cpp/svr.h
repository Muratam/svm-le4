#pragma once
#include "svm.h"
using namespace std;

// b[] = a[] - aster[]
// c[] = a[] + aster[]
struct Ok_b_x {
  const double b;
  const vector<double> x;
};
struct FindPos {  // p means kernel paraeter
  double c_center, c_offset;
  double p_center, p_offset;
  double error;
  void print() {
    std::cout << "c:" << c_center << " , p:" << p_center << " | " << error
              << std::endl;
  }
};

class SVR : public SVM {
 private:
  const double eps;
  const double C;
  bool is_valid = false;
  vector<Ok_b_x> oks;

 public:
  enum cross_validation_type {
    mean_abs,
    mean_square,
    coefficient,
    correct_num
  };

 private:
  virtual double kernel_dot_to_w(const vector<double> &x) const override;

 public:
  bool get_is_valid() { return is_valid; }

  SVR(const vector<vector<double>> &x, const vector<double> &y, Kernel kernel,
      double C = 1000, double eps = 1e-2);

  virtual void solve(const vector<vector<double>> &x,
                     const vector<double> &y) override;

  virtual double func(const vector<double> &x) const override;

  virtual void test(const vector<vector<double>> &x,
                    const vector<double> &y) const override;

  static double calc_diff(const vector<vector<double>> &test_x,
                          const vector<double> &test_y, const SVR &svr,
                          const cross_validation_type &cvtype,
                          const double eps);

  static double cross_validation(const vector<vector<double>> &x,
                                 const vector<double> &y, const Kernel kernel,
                                 const double C, const double eps = 1e-2,
                                 const cross_validation_type cvtype = mean_abs,
                                 const int div = 10);
  static FindPos search_parameter(const vector<vector<double>> &x,
                                  const vector<double> &y,
                                  const Kernel::kind kind,
                                  const double eps = 1e-2,
                                  const cross_validation_type cvtype = mean_abs,
                                  int cross_validate_div = 8);
};
