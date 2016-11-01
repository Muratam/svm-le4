#pragma once
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define REP(i, n) for (int i = 0; i < (n); ++i)
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define ALL(f, x, ...) \
  ([&](auto &ALL) { return (f)(begin(ALL), end(ALL), ##__VA_ARGS__); })(x)
using std::vector;
using std::unordered_map;
using std::string;
using std::cout;
using std::endl;
using std::function;

unordered_map<string, string> parse_args(vector<string> args,
                                         vector<std::pair<string, string>> kvs);