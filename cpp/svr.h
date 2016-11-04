#pragma once
#include "kernel.h"
#include "plotable.h"
#include "util.h"

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
  void print() const {
    std::cout << "c:2**" << c_center << " , p:2**" << p_center << " | " << error
              << "  (c:" << std::pow(2.0, c_center)
              << " , p:" << std::pow(2.0, p_center) << ")\n";
  }
  static constexpr double max_error = 1e20;
};

class SVR : public Plotable {
 private:
  const double eps;
  const double C;
  bool is_valid = false;
  vector<Ok_b_x> oks;
  double theta;
  const Kernel kernel;

 public:
  enum cross_validation_type {
    mean_abs,
    mean_square,
    coefficient,
    correct_num
  };

 private:
  virtual double kernel_dot_to_w(const vector<double> &x) const;

 public:
  bool get_is_valid() { return is_valid; }

  SVR(const vector<vector<double>> &x, const vector<double> &y, Kernel kernel,
      double C = 1000, double eps = 1e-2);

  void solve(const vector<vector<double>> &x, const vector<double> &y);

  virtual double func(const vector<double> &x) const override;

  void print_func() const;

  void test(const vector<vector<double>> &x, const vector<double> &y) const;

  static cross_validation_type get_cross_validation(vector<string> args);

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
                                  int cross_validate_div = 8,
                                  bool plot = false);
};
