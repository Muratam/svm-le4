#pragma once
#include "quadProg/QuadProg++.hh"
#include "util.h"

class Kernel {
 public:
  struct search_range {
    // 2 ** (center ± offset)
    double center, offset;
  };
  enum kind { linear, polynomial, gauss };
  const Kernel::kind m_kind;
  const vector<double> param;

  Kernel(const Kernel::kind k_type, const vector<double> param = {})
      : m_kind(k_type), param(param) {}

  double kernel(const vector<double> &x, const vector<double> &y) const;

  static search_range get_default_range(const Kernel::kind k_type);

  static Kernel::kind strings2kernel_kind(vector<string> args);

  static void read_data(string filename, vector<vector<double>> &x,
                        vector<double> &y);

  static void normalize(vector<vector<double>> &x);
};