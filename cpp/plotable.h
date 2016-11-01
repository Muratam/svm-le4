#pragma once
#include "util.h"

class Plotable {
 private:
  virtual double func(const vector<double> &x) const = 0;

 public:
  void plot_data(const vector<vector<double>> &x, const vector<double> &y,
                 const string filename, const int plot_grid = 100) {
    if (filename == "") return;
    std::ofstream ofile;
    ofile.open(filename, std::ios::out);

    bool print_grid = plot_grid > 0;
    if (print_grid) {
      REP(i, plot_grid) {
        REP(j, plot_grid) {
          auto i_p = (double)i / plot_grid, j_p = (double)j / plot_grid;
          ofile << i_p << " " << j_p << " ";
          ofile << func({i_p, j_p}) << endl;
        }
      }
    } else {
      REP(i, x.size()) {
        REP(j, x[0].size()) { ofile << x[i][j] << " "; }
        ofile << func(x[i]) << endl;
      }
    }
    ofile.close();
    cout << "saved as " + filename << endl;
  }
};