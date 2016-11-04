#pragma once
#include "util.h"

class Plotable {
 private:
  virtual double func(const vector<double> &x) const = 0;

 public:
  void plot_data(const vector<vector<double>> &x, const vector<double> &y,
                 const string filename, const int plot_grid = 100) {
    if (x.size() == 0) {
      cout << "invalid sample data... can't plot ..." << endl;
      return;
    }
    if (filename == "") return;
    std::ofstream ofile;
    ofile.open(filename, std::ios::out);
    bool print_grid = plot_grid > 0;
    if (print_grid) {          // グリッドでプロットする
      if (x[0].size() == 1) {  // 一次元の場合
        auto linear_grid = plot_grid * plot_grid;
        REP(i, linear_grid) {
          auto i_p = (double)i / linear_grid;
          ofile << i_p << " " << func({i_p}) << endl;
        }
      } else if (x[0].size() == 2) {  // デフォルトで二次元
        REP(i, plot_grid) {
          REP(j, plot_grid) {
            auto i_p = (double)i / plot_grid, j_p = (double)j / plot_grid;
            ofile << i_p << " " << j_p << " ";
            ofile << func({i_p, j_p}) << endl;
          }
        }
      } else {
        cout << "grid plot can only 1 or 2 dimension" << endl;
        return;
      }
    } else {  // x,yを使ってプロットする
      REP(i, x.size()) {
        REP(j, x[0].size()) { ofile << x[i][j] << " "; }
        ofile << func(x[i]) << endl;
      }
    }
    ofile.close();
    cout << "saved as " + filename << endl;
  }
};